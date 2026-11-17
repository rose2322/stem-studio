// 스펙트로그램 입출력 모델(BS-RoFormer, SCNet) 브라우저 실행기
// 규격 출처: unblend (MIT, github.com/Ryan5453/unblend) onnx.md "RoFormer" / "SCNet" / "Full tracks"
import { fft, ifft } from "./lib/demucs-web/fft.js";

const SR = 44100;

/* ---------------- 모델 목록 ---------------- */
const HF = "https://huggingface.co/Ryan5453/unblend/resolve/30b749fd691312b1e4b3f8fe79df0b51cde836cd/";
export const SPECTRAL_MODELS = {
  bs_roformer_sw: {
    name: "BS-RoFormer SW (6트랙→4트랙)",
    nfft: 2048, hop: 512, seg: 588800, input: 588800, window: "hann", normalized: false,
    sources: ["bass", "drums", "other", "vocals", "guitar", "piano"],
    files: {
      fp16: { url: HF + "bs_roformer_sw_fp16.onnx", bytes: 351215164 },
      fp32: { url: HF + "bs_roformer_sw_fp32.onnx", bytes: 700419210 },
    },
    fp16NeedsShaderF16: true,
    license: "가중치 라이선스 없음(권리 불명확) · 개인 비상업 테스트용",
  },
  scnet_xl_wide_v5: {
    name: "SCNet XL (4트랙)",
    nfft: 4096, hop: 1024, seg: 485100, input: 486400, window: "rect", normalized: true,
    sources: ["drums", "bass", "other", "vocals"],
    files: {
      fp16: { url: HF + "scnet_xl_wide_v5_fp16.onnx", bytes: 114857118 },
      fp32: { url: HF + "scnet_xl_wide_v5_fp32.onnx", bytes: 221482626 },
    },
    fp16NeedsShaderF16: false, // 가중치만 fp16 저장, 계산은 fp32
    license: "가중치 라이선스 없음 · MUSDB18(비상업 연구용)로만 학습",
  },
};

/** INST = other + guitar + piano 처럼 화면용 4트랙으로 합친다 */
export function foldToFour(stems) {
  const inst = { left: stems.other.left.slice(), right: stems.other.right.slice() };
  for (const k of ["guitar", "piano"]) if (stems[k]) for (let i = 0; i < inst.left.length; i++) { inst.left[i] += stems[k].left[i]; inst.right[i] += stems[k].right[i]; }
  return { drums: stems.drums, bass: stems.bass, other: inst, vocals: stems.vocals };
}

/* ---------------- STFT / iSTFT (torch.stft / torch.istft center=True, reflect) ---------------- */
const winCache = new Map();
function getWindow(n, type) {
  const k = n + type;
  if (!winCache.has(k)) {
    const w = new Float32Array(n);
    for (let i = 0; i < n; i++) w[i] = type === "rect" ? 1 : 0.5 * (1 - Math.cos((2 * Math.PI * i) / n));
    winCache.set(k, w);
  }
  return winCache.get(k);
}
function reflectIndex(i, len) {
  const p = 2 * (len - 1); i = ((i % p) + p) % p; return i >= len ? p - i : i;
}

/** 한 채널 → 실수/허수 [F*T] (F 우선, 인덱스 = b*T + f) */
export function stftChannel(x, cfg) {
  const { nfft, hop, window, normalized } = cfg;
  const n = x.length, T = Math.floor(n / hop) + 1, F = nfft / 2 + 1, half = nfft / 2;
  const w = getWindow(nfft, window), scale = normalized ? 1 / Math.sqrt(nfft) : 1;
  const re = new Float32Array(F * T), im = new Float32Array(F * T);
  const fr = new Float32Array(nfft), oR = new Float32Array(nfft), oI = new Float32Array(nfft);
  for (let t = 0; t < T; t++) {
    const s = t * hop - half;
    for (let i = 0; i < nfft; i++) fr[i] = x[reflectIndex(s + i, n)] * w[i];
    fft(oR, oI, fr, nfft);
    for (let b = 0; b < F; b++) { re[b * T + t] = oR[b] * scale; im[b * T + t] = oI[b] * scale; }
  }
  return { re, im, F, T };
}

/** 실수/허수 [F*T] → 길이 length 신호 */
export function istftChannel(re, im, F, T, length, cfg) {
  const { nfft, hop, window, normalized } = cfg;
  const w = getWindow(nfft, window), half = nfft / 2, scale = normalized ? Math.sqrt(nfft) : 1;
  const full = (T - 1) * hop + nfft;
  const y = new Float32Array(full), ws = new Float32Array(full);
  const inR = new Float32Array(nfft), inI = new Float32Array(nfft), oR = new Float32Array(nfft), oI = new Float32Array(nfft);
  for (let t = 0; t < T; t++) {
    for (let b = 0; b < F; b++) { inR[b] = re[b * T + t] * scale; inI[b] = im[b * T + t] * scale; }
    inI[0] = 0; inI[F - 1] = 0; // irfft 는 DC/나이퀴스트의 허수부를 무시
    for (let b = 1; b < F - 1; b++) { inR[nfft - b] = inR[b]; inI[nfft - b] = -inI[b]; }
    ifft(oR, oI, inR, inI, nfft);
    const s = t * hop;
    for (let i = 0; i < nfft; i++) { y[s + i] += oR[i] * w[i]; ws[s + i] += w[i] * w[i]; }
  }
  const out = new Float32Array(length);
  for (let i = 0; i < length; i++) { const j = i + half; out[i] = ws[j] > 1e-11 ? y[j] / ws[j] : 0; }
  return out;
}

/* ---------------- 구간 1개 실행 ---------------- */
export async function runSpectralSegment(session, ort, cfg, segL, segR, gpuDevice = null) {
  // SCNet: 논리 구간 뒤에 0을 덧붙여 그래프 길이에 맞춘다 (실제 음원으로 채우면 안 됨)
  const L = new Float32Array(cfg.input), R = new Float32Array(cfg.input);
  L.set(segL.subarray(0, cfg.seg)); R.set(segR.subarray(0, cfg.seg));
  const a = stftChannel(L, cfg), b = stftChannel(R, cfg);
  const { F, T } = a, plane = F * T;
  const sr = new Float32Array(2 * plane), si = new Float32Array(2 * plane);
  sr.set(a.re, 0); sr.set(b.re, plane); si.set(a.im, 0); si.set(b.im, plane);
  const dims = [1, 2, F, T];
  const feeds = { spec_real: new ort.Tensor("float32", sr, dims), spec_imag: new ort.Tensor("float32", si, dims) };
  const run = () => session.run(feeds);
  const res = await withGpuErrorScopes(gpuDevice, run);
  const oR = res.out_spec_real, oI = res.out_spec_imag;
  if (!oR || !oI) throw new Error("모델 출력 이름이 예상과 달라요: " + Object.keys(res).join(","));
  const S = oR.dims[1];
  if (oR.dims[2] !== 2 || oR.dims[3] !== F || oR.dims[4] !== T) throw new Error(`출력 모양이 이상해요: ${oR.dims}`);
  const dR = oR.data, dI = oI.data;
  const out = [];
  for (let s = 0; s < S; s++) {
    const ch = [];
    for (let c = 0; c < 2; c++) {
      const off = (s * 2 + c) * plane;
      const x = istftChannel(dR.subarray(off, off + plane), dI.subarray(off, off + plane), F, T, cfg.input, cfg);
      ch.push(x.subarray(0, cfg.seg));
    }
    for (let i = 0; i < cfg.seg; i += 997) if (!Number.isFinite(ch[0][i]) || !Number.isFinite(ch[1][i])) throw new Error("GPU 계산 결과에 숫자가 아닌 값(NaN)이 있어요 – GPU 메모리/정밀도 문제");
    out.push({ left: ch[0], right: ch[1] });
  }
  for (const k in res) res[k].dispose?.();
  feeds.spec_real.dispose?.(); feeds.spec_imag.dispose?.();
  return out;
}

/* ---------------- 곡 전체 (unblend/demucs apply_model 과 같은 방식) ---------------- */
function triangle(seg) {
  const w = new Float32Array(seg), half = Math.floor(seg / 2), peak = Math.max(half, seg - half);
  for (let i = 0; i < seg; i++) w[i] = (i < half ? i + 1 : seg - i) / peak;
  return w;
}
function paddedChunk(src, offset, length, target) {
  const delta = target - length, start = offset - Math.floor(delta / 2), out = new Float32Array(target);
  const a = Math.max(0, start), b = Math.min(src.length, start + target);
  if (b > a) out.set(src.subarray(a, b), a - start);
  return { data: out, trim: Math.floor(delta / 2) };
}
async function splitApply(runSeg, L, R, seg, nSrc, overlap, tick) {
  const len = L.length, stride = Math.floor((1 - overlap) * seg), w = triangle(seg);
  const outs = Array.from({ length: nSrc }, () => ({ left: new Float32Array(len), right: new Float32Array(len) }));
  const sumW = new Float32Array(len);
  for (let off = 0; off < len; off += stride) {
    const clen = Math.min(seg, len - off);
    const cl = paddedChunk(L, off, clen, seg), cr = paddedChunk(R, off, clen, seg);
    const res = await runSeg(cl.data, cr.data);
    for (let s = 0; s < nSrc; s++) for (let i = 0; i < clen; i++) {
      outs[s].left[off + i] += w[i] * res[s].left[cl.trim + i];
      outs[s].right[off + i] += w[i] * res[s].right[cl.trim + i];
    }
    for (let i = 0; i < clen; i++) sumW[off + i] += w[i];
    tick();
  }
  for (const o of outs) for (let i = 0; i < len; i++) { o.left[i] /= sumW[i]; o.right[i] /= sumW[i]; }
  return outs;
}
export function spectralSegmentCount(len, cfg, overlap = 0.25, shifts = 0) {
  const stride = Math.floor((1 - overlap) * cfg.seg);
  return shifts > 0 ? shifts * Math.ceil((len + SR / 2) / stride) : Math.ceil(len / stride);
}
export async function separateSpectral(session, ort, cfg, left, right, opts = {}) {
  const overlap = opts.overlap ?? 0.25, shifts = opts.shifts ?? 0, random = opts.random || Math.random;
  const len = left.length, nSrc = cfg.sources.length;
  const total = Math.max(1, spectralSegmentCount(len, cfg, overlap, shifts));
  let done = 0;
  const tick = () => { done++; opts.onProgress?.({ progress: Math.min(1, done / total), currentSegment: Math.min(done, total), totalSegments: total }); };
  const runSeg = (a, b) => runSpectralSegment(session, ort, cfg, a, b, opts.gpuDevice || null);
  let outs;
  if (shifts <= 0) outs = await splitApply(runSeg, left, right, cfg.seg, nSrc, overlap, tick);
  else {
    const S = Math.floor(SR / 2);
    const pL = new Float32Array(len + 2 * S), pR = new Float32Array(len + 2 * S);
    pL.set(left, S); pR.set(right, S);
    outs = Array.from({ length: nSrc }, () => ({ left: new Float32Array(len), right: new Float32Array(len) }));
    for (let r = 0; r < shifts; r++) {
      const o = Math.floor(random() * (S + 1));
      const res = await splitApply(runSeg, pL.subarray(o, len + S), pR.subarray(o, len + S), cfg.seg, nSrc, overlap, tick);
      const k = S - o;
      for (let s = 0; s < nSrc; s++) for (let i = 0; i < len; i++) { outs[s].left[i] += res[s].left[k + i] / shifts; outs[s].right[i] += res[s].right[k + i] / shifts; }
    }
  }
  const result = {};
  cfg.sources.forEach((name, s) => (result[name] = outs[s]));
  return result;
}

/* ---------------- 안전장치 (unblend 참고, MIT) ---------------- */
/** ONNX 파일의 metadata_props 만 읽기 (그래프 본문은 건너뜀) */
export function readOnnxMetadata(bytes) {
  const meta = {}, dec = new TextDecoder(); let pos = 0;
  const varint = end => { let v = 0, sc = 1; while (pos < end) { const b = bytes[pos++]; v += (b & 0x7f) * sc; if (b < 0x80) return v; sc *= 128; if (sc > 2 ** 63) return null; } return null; };
  const skip = (wt, end) => { if (wt === 0) return varint(end) !== null; if (wt === 1) pos += 8; else if (wt === 5) pos += 4; else if (wt === 2) { const l = varint(end); if (l === null) return false; pos += l; } else return false; return pos <= end; };
  while (pos < bytes.length) {
    const tag = varint(bytes.length); if (tag === null) break;
    const field = Math.floor(tag / 8), wt = tag % 8;
    if (field !== 14 || wt !== 2) { if (!skip(wt, bytes.length)) break; continue; }
    const l = varint(bytes.length); if (l === null || pos + l > bytes.length) break;
    const end = pos + l; let key, val = "";
    while (pos < end) {
      const et = varint(end); if (et === null) break; const ef = Math.floor(et / 8);
      if (et % 8 === 2 && (ef === 1 || ef === 2)) { const sz = varint(end); if (sz === null || pos + sz > end) break; const t = dec.decode(bytes.subarray(pos, pos + sz)); pos += sz; if (ef === 1) key = t; else val = t; }
      else if (!skip(et % 8, end)) break;
    }
    pos = end; if (key !== undefined) meta[key] = val;
  }
  return meta;
}
export function checkSpectralMetadata(meta, cfg) {
  const p = [];
  const eq = (k, v) => { if (meta[k] === undefined) p.push(`${k} 없음`); else if (String(meta[k]) !== String(v)) p.push(`${k}=${meta[k]} (예상 ${v})`); };
  eq("sample_rate", SR); eq("stft_n_fft", cfg.nfft); eq("stft_hop_length", cfg.hop); eq("segment_samples", cfg.input);
  eq("stft_normalized", cfg.normalized ? "true" : "false"); eq("stft_window", cfg.window === "rect" ? "none" : "hann");
  let src = ""; try { src = JSON.parse(meta.sources || "[]").join(","); } catch {}
  if (src !== cfg.sources.join(",")) p.push(`sources=${meta.sources}`);
  if (p.length) throw new Error("모델 파일 규격이 코드와 달라요: " + p.join("; "));
  return meta;
}
/** WebGPU 오류(메모리 부족, 셰이더 실패)는 ORT가 조용히 넘기므로 직접 잡는다 */
export async function withGpuErrorScopes(device, fn) {
  if (!device) return fn();
  const filters = ["validation", "out-of-memory", "internal"];
  for (const f of filters) device.pushErrorScope(f);
  let err, value;
  try { value = await fn(); }
  finally {
    for (let i = filters.length - 1; i >= 0; i--) {
      const e = await device.popErrorScope().catch(() => null);
      if (e && !err) err = `WebGPU ${filters[i]} 오류: ${e.message}`;
    }
  }
  if (err) throw new Error(err);
  return value;
}
