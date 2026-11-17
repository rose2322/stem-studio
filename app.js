// Stem Studio · 모델 비교 테스트 (htdemucs / SCNet XL / BS-RoFormer SW) · onnxruntime-web
import { DemucsProcessor, CONSTANTS } from "./lib/demucs-web/index.js";
import { STEMS, sliceForTest, verifySeparation, encodeWav, encodeMp3 } from "./analysis.js";
import { separateOfficial } from "./separate.js";
import { SPECTRAL_MODELS, separateSpectral, foldToFour, readOnnxMetadata, checkSpectralMetadata } from "./spectral.js";

const ORT_URLS = [
  "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.26.0/dist/ort.all.min.mjs",
  "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.26.0/dist/ort.all.mjs",
  "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/ort.all.mjs",
  "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.21.0/dist/ort.all.mjs",
];
const SR = CONSTANTS.SAMPLE_RATE;
const MODELS = {
  htdemucs: { name: "htdemucs (기존 성공 버전)", url: CONSTANTS.DEFAULT_MODEL_URL, bytes: 172e6, license: "가중치 라이선스 명시 없음 (Meta 공개)" },
  scnet_xl_wide_v5: SPECTRAL_MODELS.scnet_xl_wide_v5,
  bs_roformer_sw: SPECTRAL_MODELS.bs_roformer_sw,
};
const $ = s => document.querySelector(s);
const log = [];
const report = line => { log.push(line); $("#report").textContent = log.join("\n"); };
const status = (t, cls = "") => { $("#status").className = "small " + cls; $("#status").textContent = t; };
const bar = p => ($("#bar").style.width = `${Math.round(p * 100)}%`);
const sleep = ms => new Promise(r => setTimeout(r, ms));
const fmt = s => (s < 60 ? `${Math.round(s)}초` : `${Math.floor(s / 60)}분 ${Math.round(s % 60)}초`);
const MB = b => `${Math.round(b / 1048576)}MB`;
const env = { webgpu: false, f16: false };

// ① 기기 점검 ---------------------------------------------------------------
async function diagnose() {
  const lines = [];
  report(`[기기] ${navigator.userAgent}`);
  report(`[기기] CPU 코어 ${navigator.hardwareConcurrency || "?"} · 메모리 ${navigator.deviceMemory ? navigator.deviceMemory + "GB 이상" : "?"} · 멀티스레드 ${crossOriginIsolated}`);
  if (navigator.gpu) {
    try {
      const ad = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
      if (ad) {
        env.webgpu = true; env.f16 = ad.features.has("shader-f16");
        const info = ad.info || {}, lim = ad.limits || {};
        report(`[WebGPU] ${info.vendor || ""} ${info.architecture || ""} · shader-f16 ${env.f16} · maxBuffer ${MB(lim.maxBufferSize || 0)} · maxStorageBinding ${MB(lim.maxStorageBufferBindingSize || 0)}`);
        lines.push(`<span class="ok">✔ WebGPU 사용 가능</span> (${info.vendor || "GPU"} ${info.architecture || ""})`);
        lines.push(env.f16 ? `<span class="ok">✔ GPU 반정밀도(fp16) 지원 → BS-RoFormer 351MB 버전 사용</span>` : `<span class="warn">△ fp16 미지원 → BS-RoFormer는 700MB 버전 필요</span>`);
      } else lines.push(`<span class="bad">✘ WebGPU 어댑터 없음 → 새 모델은 실행 불가</span>`);
    } catch (e) { report("[WebGPU] 오류 " + e.message); lines.push(`<span class="bad">✘ WebGPU 오류</span>`); }
  } else lines.push(`<span class="bad">✘ WebGPU 미지원 브라우저 → 크롬 최신 버전 필요</span>`);
  try {
    const est = await navigator.storage?.estimate?.();
    if (est) { report(`[저장공간] 사용 ${MB(est.usage)} / 허용 ${MB(est.quota)}`); lines.push(`저장공간 여유 약 ${MB(est.quota - est.usage)}`); }
  } catch {}
  $("#diag").innerHTML = lines.join("<br>");
  updateModelInfo();
}

function modelFile(key) {
  const m = MODELS[key];
  if (!m.files) return { url: m.url, bytes: m.bytes, precision: "fp32", approx: true };
  const precision = m.fp16NeedsShaderF16 && !env.f16 ? "fp32" : "fp16";
  return { ...m.files[precision], precision };
}
async function updateModelInfo() {
  const key = $("#model").value, m = MODELS[key], f = modelFile(key);
  let cached = false;
  try { cached = !!(await (await caches.open("stem-models")).match(f.url)); } catch {}
  $("#modelinfo").innerHTML = `파일 ${MB(f.bytes)} (${f.precision}) · ${cached ? '<span class="ok">폰에 저장됨</span>' : '<span class="warn">처음 한 번 다운로드 필요</span>'}<br>라이선스: ${m.license}`;
}
$("#model").addEventListener("change", updateModelInfo);

// 모델 파일 받기 (Cache Storage에 저장) ---------------------------------------------
async function getModelBytes(url, expected) {
  let cache = null;
  try { cache = await caches.open("stem-models"); } catch {}
  const hit = cache && (await cache.match(url));
  if (hit) { status("저장된 모델 불러오는 중…"); return new Uint8Array(await hit.arrayBuffer()); }
  const res = await fetch(url);
  if (!res.ok) throw new Error(`모델 다운로드 실패 HTTP ${res.status}`);
  const total = +res.headers.get("Content-Length") || expected || 172e6;
  const reader = res.body.getReader(), chunks = []; let got = 0, last = 0;
  for (;;) {
    const { done, value } = await reader.read(); if (done) break;
    chunks.push(value); got += value.length;
    if (got - last > 3e6) { last = got; bar((got / total) * 0.3); status(`모델 다운로드 ${MB(got)} / ${MB(total)} (처음 한 번만)`); }
  }
  if (expected && Math.abs(got - expected) > 1024) throw new Error(`모델 파일 크기가 달라요 (${got} / ${expected}) – 다운로드가 끊겼을 수 있어요`);
  const buf = new Uint8Array(got); let o = 0; for (const c of chunks) { buf.set(c, o); o += c.length; }
  chunks.length = 0;
  report(`[모델] 다운로드 ${MB(got)}`);
  if (cache) { try { await cache.put(url, new Response(buf, { headers: { "Content-Type": "application/octet-stream" } })); report("[모델] 폰에 저장 완료"); } catch (e) { report("[모델] 저장 실패(용량?) " + e.message); } }
  return buf;
}

let ort = null;
async function loadOrt() {
  if (ort) return ort;
  status("AI 실행 엔진 불러오는 중…");
  for (const u of ORT_URLS) {
    try { ort = await import(u); report(`[엔진] onnxruntime-web ${u.split("@")[1] || u}`); break; } catch (e) { report(`[엔진] 실패 ${u.split("@")[1]}: ${e.message}`); }
  }
  if (!ort) throw new Error("onnxruntime-web을 불러오지 못했어요");
  ort.env.wasm.numThreads = crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 4) : 1;
  if (ort.env.webgpu) ort.env.webgpu.powerPreference = "high-performance";
  return ort;
}

// 한 번에 모델 하나만 메모리에 올린다
let loaded = { key: null, session: null, processor: null, device: null, ep: null };
async function loadModel(key) {
  if (loaded.key === key) return loaded;
  if (loaded.session) { try { await loaded.session.release(); } catch {} report(`[메모리] ${loaded.key} 해제`); }
  loaded = { key: null, session: null, processor: null, device: null, ep: null };
  await loadOrt();
  const f = modelFile(key), spectral = !!MODELS[key].files;
  if (spectral && !env.webgpu) throw new Error("이 모델은 WebGPU가 꼭 필요해요 (CPU로는 메모리가 부족해요)");
  const bytes = await getModelBytes(f.url, f.approx ? 0 : f.bytes);
  const t = performance.now();
  if (spectral) {
    const meta = checkSpectralMetadata(readOnnxMetadata(bytes), MODELS[key]);
    report(`[모델] ${key} 규격 확인 · 계산정밀도 ${meta.compute_precision || "?"} · 저장정밀도 ${meta.weight_precision || "?"} · 라이선스표기 ${meta.license || "없음"}`);
    status(`${MODELS[key].name} 준비 중… (GPU에 올리는 데 1~2분 걸릴 수 있어요)`);
    const session = await ort.InferenceSession.create(bytes, { executionProviders: ["webgpu"], graphOptimizationLevel: "all" });
    const device = (await ort.env.webgpu?.device) ?? null;
    if (meta.compute_precision === "fp16" && device && !device.features.has("shader-f16")) { await session.release(); throw new Error("이 GPU는 fp16 계산을 지원하지 않아요"); }
    loaded = { key, session, device, ep: "webgpu" };
  } else {
    status("htdemucs 준비 중…");
    const order = env.webgpu ? ["webgpu", "wasm"] : ["wasm"];
    let err;
    for (const ep of order) {
      try {
        const p = new DemucsProcessor({ ort, sessionOptions: { executionProviders: [ep], enableCpuMemArena: false, enableMemPattern: false } });
        await p.loadModel(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength));
        loaded = { key, session: p.session, processor: p, device: null, ep }; err = null; break;
      } catch (e) { err = e; report(`[엔진] ${ep} 실패: ${e.message}`); }
    }
    if (err) throw err;
  }
  report(`[모델] ${MODELS[key].name} 준비 ${((performance.now() - t) / 1000).toFixed(1)}초 · ${loaded.ep} · 파일 ${MB(f.bytes)}`);
  return loaded;
}

async function jsMemory() {
  try {
    if (crossOriginIsolated && performance.measureUserAgentSpecificMemory) {
      const r = await Promise.race([performance.measureUserAgentSpecificMemory(), sleep(15000).then(() => null)]);
      if (r) return r.bytes;
    }
  } catch {}
  return performance.memory?.usedJSHeapSize ?? null;
}

// ② 파일 ---------------------------------------------------------------------
let file = null, input = null, inputKey = "", baseName = "song";
$("#file").addEventListener("change", e => {
  file = e.target.files[0] || null;
  $("#fileinfo").textContent = file ? `${file.name} · ${(file.size / 1048576).toFixed(1)}MB` : "선택된 파일 없음";
  $("#run").disabled = !file;
  input = null; runs.length = 0; renderRuns();
});
async function decode(f) {
  const ctx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: SR });
  try {
    const ab = await ctx.decodeAudioData(await f.arrayBuffer());
    const left = ab.getChannelData(0).slice(), right = ab.numberOfChannels > 1 ? ab.getChannelData(1).slice() : left.slice();
    return { left, right, duration: ab.duration, channels: ab.numberOfChannels };
  } finally { ctx.close(); }
}

// ③ 실행 ---------------------------------------------------------------------
const runs = [];
$("#run").addEventListener("click", async () => {
  $("#run").disabled = true; bar(0);
  let wake = null; try { wake = await navigator.wakeLock?.request("screen"); } catch {}
  const key = $("#model").value, t0 = performance.now();
  try {
    const want = `${file.name}|${file.size}|${$("#testMode").checked}`;
    if (!input || inputKey !== want) {
      baseName = file.name.replace(/\.[^.]+$/, "");
      status("원곡 읽는 중…");
      const a = await decode(file);
      input = $("#testMode").checked ? sliceForTest(a.left, a.right, SR, 30) : { left: a.left, right: a.right, startSec: 0 };
      inputKey = want; runs.length = 0;
      report(`[원곡] ${file.name} · ${a.duration.toFixed(1)}초 · 처리 구간 ${input.startSec.toFixed(0)}초부터 ${(input.left.length / SR).toFixed(1)}초`);
    }
    const secs = input.left.length / SR;
    const m = await loadModel(key);
    let tSep = 0;
    const onProgress = ({ progress, currentSegment, totalSegments }) => {
      const el = (performance.now() - tSep) / 1000, eta = progress > 0 ? el / progress - el : 0;
      bar(0.3 + 0.7 * progress);
      status(`${MODELS[key].name} 분리 중… ${currentSegment}/${totalSegments} 구간\n경과 ${fmt(el)} · 남은 시간 약 ${fmt(eta)}\n화면을 켜 둔 채로 이 탭에 머물러 주세요.`);
    };
    status(`${MODELS[key].name} 분리 시작…`); await sleep(50);
    tSep = performance.now();
    let stems, extras = null;
    if (key === "htdemucs") stems = await separateOfficial(m.session, ort, input.left, input.right, { overlap: 0.25, shifts: 1, onProgress });
    else {
      const raw = await separateSpectral(m.session, ort, MODELS[key], input.left, input.right, { overlap: 0.25, shifts: 0, onProgress, gpuDevice: m.device });
      stems = foldToFour(raw);
      if (raw.guitar) extras = { GUITAR: raw.guitar, PIANO: raw.piano, "OTHER(기타·피아노 제외)": raw.other };
    }
    const sepSec = (performance.now() - tSep) / 1000;
    const memAfter = await jsMemory();
    const v = verifySeparation(input, stems);
    const run = { key, name: MODELS[key].name, sepSec, secs, v, stems, extras, mem: memAfter, file: modelFile(key) };
    runs.push(run);
    report(`[결과] ${run.name} · 분리 ${sepSec.toFixed(1)}초 (곡 1초당 ${(sepSec / secs).toFixed(2)}초) · 3분 곡 예상 ${fmt((sepSec / secs) * 180)} · ${v.pass ? "검증 통과" : "검증 실패"}`);
    if (memAfter) report(`[메모리] 브라우저 JS 메모리 ${MB(memAfter)} (GPU 메모리는 측정 불가) · 모델 파일 ${MB(run.file.bytes)}`);
    v.checks.forEach(c => report(`[검증] ${c.ok ? "통과" : "실패"} · ${c.name} · ${c.value}`));
    v.rows.forEach(r => report(`[검증] ${r.label}: 비율 ${r.share.toFixed(1)}% · 원곡과 상관 ${r.corrIn.toFixed(2)}`));
    renderRuns();
    bar(1);
    status(`완료! ${run.name} · ${fmt((performance.now() - t0) / 1000)} 걸렸어요. 아래에서 모델별로 비교해 들어보세요.`, v.pass ? "ok" : "bad");
  } catch (e) {
    console.error(e);
    report(`[오류] ${MODELS[key].name}: ${e && e.message || e}`);
    if (e && e.stack) report(String(e.stack).split("\n").slice(0, 3).join("\n"));
    status("오류: " + (e && e.message || e) + "\n리포트를 복사해서 보내주세요.", "bad");
  } finally {
    try { await wake?.release(); } catch {}
    $("#run").disabled = !file;
  }
});

// ④ 비교 화면 -------------------------------------------------------------------
const urls = [];
function wavUrl(t) { const u = URL.createObjectURL(new Blob([encodeWav(t.left, t.right, SR)], { type: "audio/wav" })); urls.push(u); return u; }
function renderRuns() {
  urls.splice(0).forEach(u => URL.revokeObjectURL(u));
  if (!runs.length) { $("#compare").innerHTML = ""; return; }
  const table = `<table><tr><th>모델</th><th>분리 시간</th><th>3분 곡 예상</th><th>모델 파일</th><th>JS 메모리</th><th>검증</th></tr>${runs.map(r =>
    `<tr><td>${r.name}</td><td>${fmt(r.sepSec)}</td><td>${fmt((r.sepSec / r.secs) * 180)}</td><td>${MB(r.file.bytes)}</td><td>${r.mem ? MB(r.mem) : "-"}</td><td class="${r.v.pass ? "ok" : "bad"}">${r.v.pass ? "통과" : "실패"}</td></tr>`).join("")}</table>`;
  const byStem = STEMS.map((s, si) => `<div class="track" style="--c:${s.color}"><strong>${s.label}</strong>${runs.map((r, ri) =>
    `<div class="small" style="margin-top:8px">${r.name}</div><audio controls preload="none" src="${wavUrl(r.stems[s.key])}"></audio>
     <div class="row"><button class="mini ghost" data-r="${ri}" data-s="${si}">MP3 320kbps 저장</button><span class="small" id="m${ri}_${si}"></span></div>`).join("")}</div>`).join("");
  const extras = runs.filter(r => r.extras).map(r => `<details class="panel"><summary>${r.name} · INST 세부 트랙 (기타 / 피아노 / 나머지)</summary>${Object.entries(r.extras).map(([n, t]) =>
    `<div class="small" style="margin-top:8px">${n}</div><audio controls preload="none" src="${wavUrl(t)}"></audio>`).join("")}</details>`).join("");
  $("#compare").innerHTML = `<div class="panel"><strong>⑤ 모델 비교 (같은 구간)</strong>${table}<p class="small">같은 30초를 모델별로 돌린 결과예요. 트랙마다 모델을 바꿔 들으며 비교해 주세요. 소리 섞임(블리드), 뭉개짐, 금속성 잡음을 들어보세요.</p></div>${byStem}${extras}`;
}
$("#compare").addEventListener("click", async e => {
  const ri = e.target.dataset.r, si = e.target.dataset.s;
  if (ri === undefined) return;
  const r = runs[+ri], s = STEMS[+si], btn = e.target, out = $(`#m${ri}_${si}`); btn.disabled = true;
  try {
    const mp3 = await encodeMp3(window.lamejs, r.stems[s.key].left, r.stems[s.key].right, SR, 320, p => (out.textContent = `변환 ${Math.round(p * 100)}%`));
    out.textContent = `${(mp3.length / 1048576).toFixed(1)}MB`;
    const a = document.createElement("a"); a.href = URL.createObjectURL(new Blob([mp3], { type: "audio/mpeg" }));
    a.download = `${baseName}_${s.label}_${r.key}.mp3`; document.body.appendChild(a); a.click(); a.remove();
  } catch (err) { out.textContent = "실패"; report("[MP3] " + err.message); } finally { btn.disabled = false; }
});

$("#clear").addEventListener("click", async () => {
  try { await caches.delete("stem-models"); status("저장된 모델을 모두 지웠어요."); report("[저장공간] 모델 캐시 삭제"); updateModelInfo(); } catch (e) { status("삭제 실패: " + e.message, "bad"); }
});
$("#copy").addEventListener("click", async () => {
  try { await navigator.clipboard.writeText(log.join("\n")); $("#copy").textContent = "복사됨 ✔"; }
  catch { $("#copy").textContent = "복사 실패 – 아래 글자를 길게 눌러 복사하세요"; }
  setTimeout(() => ($("#copy").textContent = "리포트 복사"), 2500);
});
window.addEventListener("error", e => report("[오류] " + e.message));
window.addEventListener("unhandledrejection", e => report("[오류] " + (e.reason && e.reason.message || e.reason)));
diagnose();
