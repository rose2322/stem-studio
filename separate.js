// 공식 Demucs(PyTorch) apply_model 과 같은 방식으로 곡 전체를 분리한다.
//  - 구간 길이 7.8초(343,980샘플), 겹침 비율 overlap
//  - 삼각형 가중치로 구간을 이어 붙임 (구간 가운데 = 문맥이 충분한 부분을 더 믿음)
//  - 마지막 짧은 구간도 앞쪽 실제 음원을 문맥으로 채워서 처리 (0으로 채우지 않음)
//  - shift trick: 0~0.5초 무작위로 밀어서 여러 번 분리한 뒤 평균
//  - 곡 전체 평균/표준편차로 정규화 후 되돌림 (demucs.separate 와 같되, 평균(DC)은 4트랙에 나눠 더해 합=원곡 유지)
import { CONSTANTS, prepareModelInput, standaloneMask, standaloneIspec } from "./lib/demucs-web/index.js";

const SEG = CONSTANTS.TRAINING_SAMPLES;
const SR = CONSTANTS.SAMPLE_RATE;
export const TRACKS = ["drums", "bass", "other", "vocals"];

/** 7.8초 구간 1개를 모델에 넣고 4트랙 (left,right) 를 돌려준다 */
export async function runSegment(session, ort, segL, segR) {
  const input = prepareModelInput(segL, segR);
  const feeds = {};
  feeds[session.inputNames[0]] = new ort.Tensor("float32", input.waveform, [1, 2, SEG]);
  if (session.inputNames.length > 1) feeds[session.inputNames[1]] = new ort.Tensor("float32", input.magSpec, [1, 4, CONSTANTS.MODEL_SPEC_BINS, CONSTANTS.MODEL_SPEC_FRAMES]);
  const res = await session.run(feeds);
  let timeData = null, timeShape = null, freqData = null;
  for (const name of session.outputNames) {
    const t = res[name];
    if (t.dims.length === 4 && t.dims[2] === 2) { timeData = t.data; timeShape = t.dims; }
    else if (t.dims.length === 5 && t.dims[2] === 4) freqData = t.data;
  }
  if (!timeData) throw new Error("모델 출력(시간 영역)을 찾지 못했어요");
  const n = timeShape[3];
  const specs = freqData ? standaloneMask(freqData) : null;
  const out = [];
  for (let t = 0; t < 4; t++) {
    const L = new Float32Array(SEG), R = new Float32Array(SEG);
    const f = specs ? standaloneIspec(specs[t], SEG) : null;
    for (let i = 0; i < SEG && i < n; i++) {
      L[i] = timeData[(t * 2) * n + i] + (f ? f.left[i] : 0);
      R[i] = timeData[(t * 2 + 1) * n + i] + (f ? f.right[i] : 0);
    }
    out.push({ left: L, right: R });
  }
  for (const k in res) res[k].dispose?.();
  return out;
}

/** 길이 targetLen 으로, 가운데 정렬해서 실제 음원으로 패딩 (TensorChunk.padded 와 동일) */
function paddedChunk(src, offset, length, targetLen) {
  const delta = targetLen - length;
  const start = offset - Math.floor(delta / 2);
  const out = new Float32Array(targetLen);
  const a = Math.max(0, start), b = Math.min(src.length, start + targetLen);
  if (b > a) out.set(src.subarray(a, b), a - start);
  return { data: out, trim: Math.floor(delta / 2) };
}

function triangleWeight(seg, power = 1) {
  const w = new Float32Array(seg), half = Math.floor(seg / 2);
  for (let i = 0; i < half; i++) w[i] = i + 1;
  for (let i = half; i < seg; i++) w[i] = seg - i;
  let max = 0; for (let i = 0; i < seg; i++) max = Math.max(max, w[i]);
  for (let i = 0; i < seg; i++) w[i] = Math.pow(w[i] / max, power);
  return w;
}

/** shift 없이 구간 분할 + 삼각 가중치로 이어 붙이기 (apply_model split=True) */
async function splitApply(runSeg, L, R, overlap, tick) {
  const len = L.length, stride = Math.floor((1 - overlap) * SEG);
  const w = triangleWeight(SEG);
  const outs = TRACKS.map(() => ({ left: new Float32Array(len), right: new Float32Array(len) }));
  const sumW = new Float32Array(len);
  for (let off = 0; off < len; off += stride) {
    const clen = Math.min(SEG, len - off);
    const cl = paddedChunk(L, off, clen, SEG), cr = paddedChunk(R, off, clen, SEG);
    const res = await runSeg(cl.data, cr.data);
    for (let t = 0; t < 4; t++) {
      const o = outs[t], r = res[t];
      for (let i = 0; i < clen; i++) {
        o.left[off + i] += w[i] * r.left[cl.trim + i];
        o.right[off + i] += w[i] * r.right[cl.trim + i];
      }
    }
    for (let i = 0; i < clen; i++) sumW[off + i] += w[i];
    tick();
  }
  for (const o of outs) for (let i = 0; i < len; i++) { o.left[i] /= sumW[i]; o.right[i] /= sumW[i]; }
  return outs;
}

export function countSegments(len, overlap, shifts) {
  const stride = Math.floor((1 - overlap) * SEG);
  const maxShift = Math.floor(0.5 * SR);
  const segFor = n => Math.ceil(n / stride);
  return shifts > 0 ? shifts * segFor(len + maxShift) : segFor(len); // 근사치 (진행률 표시용)
}

/**
 * 곡 전체 분리.
 * opts: { overlap=0.25, shifts=1, onProgress({progress,currentSegment,totalSegments}), random=Math.random }
 */
export async function separateOfficial(session, ort, left, right, opts = {}) {
  const overlap = opts.overlap ?? 0.25, shifts = opts.shifts ?? 1, random = opts.random || Math.random;
  const len = left.length;

  // demucs.separate: ref = 모노 평균, (wav - mean) / std
  let mean = 0; for (let i = 0; i < len; i++) mean += (left[i] + right[i]) / 2; mean /= len;
  let v = 0; for (let i = 0; i < len; i++) { const d = (left[i] + right[i]) / 2 - mean; v += d * d; }
  const std = Math.sqrt(v / Math.max(1, len - 1)) || 1;
  const L = new Float32Array(len), R = new Float32Array(len);
  for (let i = 0; i < len; i++) { L[i] = (left[i] - mean) / std; R[i] = (right[i] - mean) / std; }

  const total = Math.max(1, countSegments(len, overlap, shifts));
  let done = 0;
  const tick = () => { done++; opts.onProgress?.({ progress: Math.min(1, done / total), currentSegment: Math.min(done, total), totalSegments: total }); };
  const runSeg = (a, b) => runSegment(session, ort, a, b);

  let outs;
  if (shifts <= 0) {
    outs = await splitApply(runSeg, L, R, overlap, tick);
  } else {
    const maxShift = Math.floor(0.5 * SR);
    const padL = new Float32Array(len + 2 * maxShift), padR = new Float32Array(len + 2 * maxShift);
    padL.set(L, maxShift); padR.set(R, maxShift);
    outs = TRACKS.map(() => ({ left: new Float32Array(len), right: new Float32Array(len) }));
    const used = [];
    for (let s = 0; s < shifts; s++) {
      const offset = Math.floor(random() * (maxShift + 1));
      used.push(offset);
      // TensorChunk(padded_mix, offset, length + max_shift - offset)
      const sl = padL.subarray(offset, len + maxShift);
      const sr = padR.subarray(offset, len + maxShift);
      const res = await splitApply(runSeg, sl, sr, overlap, tick);
      const k = maxShift - offset;
      for (let t = 0; t < 4; t++) for (let i = 0; i < len; i++) {
        outs[t].left[i] += res[t].left[k + i] / shifts;
        outs[t].right[i] += res[t].right[k + i] / shifts;
      }
    }
    opts.onShifts?.(used);
  }

  const result = {};
  TRACKS.forEach((name, t) => {
    const o = outs[t];
    for (let i = 0; i < len; i++) { o.left[i] = o.left[i] * std + mean / 4; o.right[i] = o.right[i] * std + mean / 4; }
    result[name] = o;
  });
  return result;
}
