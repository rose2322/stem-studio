// Stem Studio · 브라우저 AI 음원 분리 (Demucs htdemucs, onnxruntime-web)
import { DemucsProcessor, CONSTANTS } from "./lib/demucs-web/index.js";
import { STEMS, sliceForTest, verifySeparation, encodeWav, encodeMp3 } from "./analysis.js";
import { separateOfficial, runSegment, TRACKS } from "./separate.js";

const ORT_URL = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.21.0/dist/ort.all.mjs";
const MODEL_URL = CONSTANTS.DEFAULT_MODEL_URL;
const SR = CONSTANTS.SAMPLE_RATE;
const $ = s => document.querySelector(s);
const log = [];
const report = (line) => { log.push(line); $("#report").textContent = log.join("\n"); };
const status = (t, cls = "") => { $("#status").className = "small " + cls; $("#status").textContent = t; };
const bar = p => ($("#bar").style.width = `${Math.round(p * 100)}%`);
const sleep = ms => new Promise(r => setTimeout(r, ms));
const fmt = s => (s < 60 ? `${Math.round(s)}초` : `${Math.floor(s / 60)}분 ${Math.round(s % 60)}초`);

const env = { webgpu: false };

// ① 기기 점검 -----------------------------------------------------------
async function diagnose() {
  const lines = [];
  report(`[기기] ${navigator.userAgent}`);
  report(`[기기] CPU 코어 ${navigator.hardwareConcurrency || "?"} · 메모리 ${navigator.deviceMemory ? navigator.deviceMemory + "GB 이상" : "?"}`);
  report(`[기기] HTTPS ${isSecureContext} · 멀티스레드(crossOriginIsolated) ${crossOriginIsolated}`);
  if (navigator.gpu) {
    try {
      const ad = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
      if (ad) {
        env.webgpu = true;
        const info = ad.info || {};
        const lim = ad.limits || {};
        report(`[WebGPU] 사용 가능 · ${info.vendor || ""} ${info.architecture || ""} · maxBuffer ${Math.round((lim.maxBufferSize || 0) / 1048576)}MB · maxStorageBinding ${Math.round((lim.maxStorageBufferBindingSize || 0) / 1048576)}MB`);
        lines.push(`<span class="ok">✔ WebGPU 사용 가능</span> (${info.vendor || "GPU"} ${info.architecture || ""})`);
      } else { report("[WebGPU] 어댑터 없음"); lines.push(`<span class="warn">✘ WebGPU 어댑터 없음 → CPU로 처리 (매우 느림)</span>`); }
    } catch (e) { report("[WebGPU] 오류 " + e.message); lines.push(`<span class="warn">✘ WebGPU 오류 → CPU로 처리</span>`); }
  } else { report("[WebGPU] 브라우저 미지원"); lines.push(`<span class="warn">✘ 이 브라우저는 WebGPU 미지원 → 크롬 최신 버전 권장</span>`); }
  lines.push(crossOriginIsolated ? `<span class="ok">✔ CPU 멀티스레드 가능</span>` : `<span class="warn">△ CPU 단일 스레드 (WebGPU가 되면 상관없음)</span>`);
  let cached = false;
  try { cached = !!(await (await caches.open("stem-models")).match(MODEL_URL)); } catch {}
  lines.push(cached ? `<span class="ok">✔ AI 모델이 이미 저장되어 있음</span>` : `AI 모델 미저장 (첫 실행 때 약 172MB 다운로드)`);
  report(`[모델] 저장됨 ${cached}`);
  $("#diag").innerHTML = lines.join("<br>");
}

// 모델 받기 (Cache Storage에 저장해서 다음부터는 다운로드 생략) ------------------
async function getModelBuffer() {
  let cache = null;
  try { cache = await caches.open("stem-models"); } catch {}
  if (cache) {
    const hit = await cache.match(MODEL_URL);
    if (hit) { status("저장된 AI 모델 불러오는 중…"); return await hit.arrayBuffer(); }
  }
  const res = await fetch(MODEL_URL);
  if (!res.ok) throw new Error(`모델 다운로드 실패 HTTP ${res.status}`);
  const total = +res.headers.get("Content-Length") || 172e6;
  const reader = res.body.getReader(); const chunks = []; let got = 0, last = 0;
  for (;;) {
    const { done, value } = await reader.read(); if (done) break;
    chunks.push(value); got += value.length;
    if (got - last > 2e6) { last = got; bar(got / total * 0.3); status(`AI 모델 다운로드 ${Math.round(got / 1048576)} / ${Math.round(total / 1048576)}MB (처음 한 번만)`); }
  }
  const buf = new Uint8Array(got); let o = 0; for (const c of chunks) { buf.set(c, o); o += c.length; }
  report(`[모델] 다운로드 ${Math.round(got / 1048576)}MB`);
  if (cache) { try { await cache.put(MODEL_URL, new Response(buf.slice(), { headers: { "Content-Type": "application/octet-stream" } })); report("[모델] 폰에 저장 완료"); } catch (e) { report("[모델] 저장 실패(용량?) " + e.message); } }
  return buf.buffer;
}

let ort = null, processor = null, engineUsed = null, modelBufKeep = null;
async function loadEngine(choice, onProgress) {
  if (!ort) {
    status("AI 실행 엔진 불러오는 중…");
    ort = await import(ORT_URL);
    ort.env.wasm.numThreads = crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 4) : 1;
    if (ort.env.webgpu) ort.env.webgpu.powerPreference = "high-performance";
  }
  const modelBuf = await getModelBuffer();
  modelBufKeep = modelBuf;
  const order = choice === "wasm" ? ["wasm"] : choice === "webgpu" ? ["webgpu"] : (env.webgpu ? ["webgpu", "wasm"] : ["wasm"]);
  let lastErr = null;
  for (const ep of order) {
    try {
      status(`AI 모델 준비 중… (${ep === "webgpu" ? "WebGPU" : "CPU"})`);
      const p = new DemucsProcessor({
        ort, onProgress,
        sessionOptions: { executionProviders: [ep], enableCpuMemArena: false, enableMemPattern: false },
      });
      const t = performance.now();
      await p.loadModel(modelBuf.slice(0));
      report(`[엔진] ${ep} 준비 완료 ${((performance.now() - t) / 1000).toFixed(1)}초`);
      engineUsed = ep; return p;
    } catch (e) { lastErr = e; report(`[엔진] ${ep} 실패: ${e.message}`); }
  }
  throw lastErr || new Error("AI 엔진을 준비하지 못했어요");
}

// ② 파일 ---------------------------------------------------------------
let file = null;
$("#file").addEventListener("change", e => {
  file = e.target.files[0] || null;
  $("#fileinfo").textContent = file ? `${file.name} · ${(file.size / 1048576).toFixed(1)}MB` : "선택된 파일 없음";
  $("#run").disabled = !file;
});

async function decode(f) {
  const ctx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: SR });
  try {
    const ab = await ctx.decodeAudioData(await f.arrayBuffer());
    const left = ab.getChannelData(0).slice();
    const right = ab.numberOfChannels > 1 ? ab.getChannelData(1).slice() : left.slice();
    return { left, right, duration: ab.duration, sampleRate: ab.sampleRate, channels: ab.numberOfChannels };
  } finally { ctx.close(); }
}

let result = null, input = null, baseName = "song";
$("#run").addEventListener("click", async () => {
  $("#run").disabled = true; $("#verify").style.display = "none"; $("#tracks").innerHTML = ""; bar(0);
  let wake = null;
  try { wake = await navigator.wakeLock?.request("screen"); } catch {}
  const t0 = performance.now();
  try {
    baseName = file.name.replace(/\.[^.]+$/, "");
    status("원곡 읽는 중…");
    const audio = await decode(file);
    report(`[원곡] ${file.name} · ${audio.duration.toFixed(1)}초 · ${audio.channels}ch · ${audio.sampleRate}Hz`);
    input = $("#testMode").checked ? sliceForTest(audio.left, audio.right, SR, 30) : { left: audio.left, right: audio.right, startSec: 0 };
    const secs = input.left.length / SR;
    report(`[처리 구간] ${input.startSec.toFixed(0)}초부터 ${secs.toFixed(1)}초`);

    let tSep = 0;
    const onProgress = ({ progress, currentSegment, totalSegments }) => {
      progress = Math.min(progress, 1); currentSegment = Math.min(currentSegment, totalSegments);
      const el = (performance.now() - tSep) / 1000;
      const eta = progress > 0 ? el / progress - el : 0;
      bar(0.3 + 0.7 * progress);
      status(`AI 분리 중… ${currentSegment}/${totalSegments} 구간 · ${Math.round(progress * 100)}%\n경과 ${fmt(el)} · 남은 시간 약 ${fmt(eta)}\n화면을 켜 둔 채로 이 탭에 머물러 주세요.`);
    };
    if (!processor) processor = await loadEngine($("#engine").value, onProgress);
    else processor.onProgress = onProgress;

    const mode = $("#quality").value;
    const modeName = { lib: "기본(기존 성공 버전)", official: "공식 Demucs 방식", hq: "고음질(겹침50%·shift2)" }[mode];
    report(`[음질 모드] ${modeName}`);
    status(`AI 분리 시작 (${engineUsed === "webgpu" ? "WebGPU" : "CPU"} · ${modeName})…`);
    await sleep(50);
    tSep = performance.now();
    if (mode === "lib") result = await processor.separate(input.left, input.right);
    else result = await separateOfficial(processor.session, ort, input.left, input.right, {
      overlap: mode === "hq" ? 0.5 : 0.25, shifts: mode === "hq" ? 2 : 1, onProgress,
      onShifts: s => report(`[shift] ${s.map(x => (x / SR).toFixed(2) + "초").join(", ")}`),
    });
    const sepSec = (performance.now() - tSep) / 1000;
    report(`[분리] 완료 ${sepSec.toFixed(1)}초 · 엔진 ${engineUsed} · 곡 1초당 ${(sepSec / secs).toFixed(2)}초 걸림`);
    report(`[분리] 3분 곡 예상: 약 ${fmt(sepSec / secs * 180)}`);

    const v = verifySeparation(input, result);
    showVerify(v, sepSec, secs);
    v.checks.forEach(c => report(`[검증] ${c.ok ? "통과" : "실패"} · ${c.name} · ${c.value}`));
    v.rows.forEach(r => report(`[검증] ${r.label}: 비율 ${r.share.toFixed(1)}% · 크기 ${r.rmsDb.toFixed(1)}dB · 원곡과 상관 ${r.corrIn.toFixed(2)}`));
    report(`[판정] ${v.pass ? "실제 분리 성공" : "분리 결과 이상"}`);
    showTracks();
    $("#cmpBox").style.display = engineUsed === "webgpu" ? "block" : "none";
    bar(1);
    status(`완료! 전체 ${fmt((performance.now() - t0) / 1000)} 걸렸어요. 아래에서 트랙별로 들어보세요.`, v.pass ? "ok" : "bad");
  } catch (e) {
    console.error(e);
    report(`[오류] ${e && e.message || e}`);
    if (e && e.stack) report(String(e.stack).split("\n").slice(0, 4).join("\n"));
    status("오류: " + (e && e.message || e) + "\n아래 리포트를 복사해서 보내주세요.", "bad");
  } finally {
    try { await wake?.release(); } catch {}
    $("#run").disabled = !file;
  }
});

function showVerify(v, sepSec, secs) {
  const el = $("#verify"); el.style.display = "block";
  el.innerHTML = `<strong>③ 자동 검증 결과: <span class="${v.pass ? "ok" : "bad"}">${v.pass ? "실제 AI 분리 성공" : "분리 결과 이상"}</span></strong>
  <p class="small">처리 ${fmt(sepSec)} (곡 ${fmt(secs)}) · 엔진 ${engineUsed === "webgpu" ? "WebGPU" : "CPU"}</p>
  <table>${v.checks.map(c => `<tr><td class="${c.ok ? "ok" : "bad"}">${c.ok ? "✔" : "✘"}</td><td>${c.name}</td><td>${c.value}</td></tr>`).join("")}</table>
  <br><table><tr><th>트랙</th><th>소리 비율</th><th>원곡과 닮음</th></tr>
  ${v.rows.map(r => `<tr><td style="color:${r.color}">${r.label}</td><td>${r.share.toFixed(1)}%</td><td>${r.corrIn.toFixed(2)}</td></tr>`).join("")}</table>
  <p class="small">숫자 검사와 별개로, 귀로 들어서 드럼·베이스·악기·보컬이 각각 들리는지 꼭 확인해 주세요.</p>`;
}

function blobUrl(bytes, type) { return URL.createObjectURL(new Blob([bytes], { type })); }
function saveBytes(bytes, name, type) {
  const a = document.createElement("a"); a.href = blobUrl(bytes, type); a.download = name;
  document.body.appendChild(a); a.click(); a.remove(); setTimeout(() => URL.revokeObjectURL(a.href), 10000);
}

function showTracks() {
  $("#tracks").innerHTML = STEMS.map((s, i) => `<div class="track" style="--c:${s.color}">
    <strong>${s.label}</strong>
    <audio controls src="${blobUrl(encodeWav(result[s.key].left, result[s.key].right, SR), "audio/wav")}"></audio>
    <div class="row"><button class="mini ghost" data-mp3="${i}">MP3 320kbps 저장</button><button class="mini ghost" data-wav="${i}">WAV 저장</button><span class="small" id="mp3s${i}"></span></div>
  </div>`).join("");
}

$("#tracks").addEventListener("click", async e => {
  const w = e.target.dataset.wav, m = e.target.dataset.mp3;
  if (w !== undefined) { const s = STEMS[+w]; saveBytes(encodeWav(result[s.key].left, result[s.key].right, SR), `${baseName}_${s.label}.wav`, "audio/wav"); }
  if (m !== undefined) {
    const s = STEMS[+m], btn = e.target; btn.disabled = true;
    try {
      const t = performance.now();
      const mp3 = await encodeMp3(window.lamejs, result[s.key].left, result[s.key].right, SR, 320, p => ($(`#mp3s${m}`).textContent = `변환 ${Math.round(p * 100)}%`));
      $(`#mp3s${m}`).textContent = `${(mp3.length / 1048576).toFixed(1)}MB`;
      report(`[MP3] ${s.label} ${(mp3.length / 1024).toFixed(0)}KB · ${((performance.now() - t) / 1000).toFixed(1)}초`);
      saveBytes(mp3, `${baseName}_${s.label}.mp3`, "audio/mpeg");
    } catch (err) { $(`#mp3s${m}`).textContent = "MP3 실패"; report("[MP3] 실패 " + err.message); }
    finally { btn.disabled = false; }
  }
});

$("#copy").addEventListener("click", async () => {
  try { await navigator.clipboard.writeText(log.join("\n")); $("#copy").textContent = "복사됨 ✔"; }
  catch { $("#copy").textContent = "복사 실패 – 아래 글자를 길게 눌러 복사하세요"; }
  setTimeout(() => ($("#copy").textContent = "리포트 복사"), 2500);
});

window.addEventListener("error", e => report("[오류] " + e.message));
window.addEventListener("unhandledrejection", e => report("[오류] " + (e.reason && e.reason.message || e.reason)));
diagnose();

// WebGPU 정밀도 검사: 같은 7.8초 구간을 WebGPU와 CPU(WASM)로 각각 돌려 결과 차이를 잰다 ---------
$("#cmp").addEventListener("click", async () => {
  const btn = $("#cmp"); btn.disabled = true;
  let cpu = null;
  try {
    const SEG = CONSTANTS.TRAINING_SAMPLES;
    const off = Math.max(0, Math.floor((input.left.length - SEG) / 2));
    const segL = new Float32Array(SEG), segR = new Float32Array(SEG);
    segL.set(input.left.subarray(off, off + SEG)); segR.set(input.right.subarray(off, off + SEG));
    status("정밀도 검사 1/2: WebGPU로 계산 중…");
    let t = performance.now();
    const g = await runSegment(processor.session, ort, segL, segR);
    const tg = (performance.now() - t) / 1000;
    status("정밀도 검사 2/2: CPU로 같은 구간 계산 중… (1~5분 걸릴 수 있어요)");
    cpu = await ort.InferenceSession.create(modelBufKeep.slice(0), { executionProviders: ["wasm"], enableCpuMemArena: false, enableMemPattern: false, graphOptimizationLevel: "basic" });
    t = performance.now();
    const c = await runSegment(cpu, ort, segL, segR);
    const tc = (performance.now() - t) / 1000;
    const lines = TRACKS.map((name, k) => {
      let e = 0, p = 0;
      for (const ch of ["left", "right"]) for (let i = 0; i < SEG; i++) { const d = g[k][ch][i] - c[k][ch][i]; e += d * d; p += c[k][ch][i] ** 2; }
      return { name: STEMS[k].label, snr: 10 * Math.log10(p / (e || 1e-30)) };
    });
    const worst = Math.min(...lines.map(l => l.snr));
    lines.forEach(l => report(`[정밀도] ${l.name}: WebGPU와 CPU 결과 차이 ${l.snr.toFixed(1)} dB`));
    report(`[정밀도] WebGPU ${tg.toFixed(1)}초 · CPU ${tc.toFixed(1)}초 (7.8초 구간)`);
    const ok = worst >= 40;
    report(`[정밀도 판정] ${ok ? "WebGPU 계산은 CPU와 사실상 같음 → 음질 저하 원인 아님" : worst >= 20 ? "작은 차이 있음 → 들어서 비교 필요" : "차이 큼 → WebGPU 계산이 음질을 떨어뜨리고 있음"}`);
    status(`정밀도 검사 완료: 가장 큰 차이 ${worst.toFixed(1)} dB (40dB 이상이면 같다고 봐도 됨)`, ok ? "ok" : "bad");
  } catch (e) {
    report("[정밀도] 실패 " + (e && e.message || e)); status("정밀도 검사 실패: " + (e && e.message || e), "bad");
  } finally { try { await cpu?.release(); } catch {} btn.disabled = false; }
});
