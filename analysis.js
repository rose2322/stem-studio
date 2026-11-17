// Stem Studio · 분리 결과 검증과 파일 만들기 (브라우저/Node 공용, 외부 의존성 없음)

export const STEMS = [
  { key: "drums", label: "DRUMS", color: "#b9ff4d" },
  { key: "bass", label: "BASS", color: "#ff923f" },
  { key: "other", label: "INST", color: "#4aa8ff" },
  { key: "vocals", label: "VOCALS", color: "#d77cff" },
];

/** 테스트용 구간 자르기: 곡 앞부분 인트로를 피해 30초 지점부터 seconds초 */
export function sliceForTest(left, right, sr, seconds) {
  const len = Math.min(left.length, Math.round(seconds * sr));
  const maxStart = left.length - len;
  const start = Math.max(0, Math.min(Math.round(30 * sr), maxStart));
  return { left: left.slice(start, start + len), right: right.slice(start, start + len), startSec: start / sr };
}

function energy(a) { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * a[i]; return s; }
function dot(a, b) { let s = 0; const n = Math.min(a.length, b.length); for (let i = 0; i < n; i++) s += a[i] * b[i]; return s; }
const db = x => (x > 0 ? 10 * Math.log10(x) : -Infinity);

/**
 * 진짜로 분리됐는지 숫자로 판정한다.
 *  - 복원 SNR: 4개 트랙을 더하면 원곡이 되어야 한다 (모델이 소리를 지어내지 않았는지)
 *  - 트랙끼리 상관: 서로 다른 소리여야 한다 (원곡을 4번 복사한 가짜 결과를 걸러냄)
 *  - 에너지 비율: 한 트랙에 소리가 몰리거나 전부 무음이면 실패
 */
export function verifySeparation(input, stems) {
  const n = input.left.length;
  const inE = energy(input.left) + energy(input.right);
  const rows = STEMS.map(s => {
    const t = stems[s.key];
    if (!t || t.left.length < n * 0.99) throw new Error(`${s.label} 트랙 길이가 맞지 않아요`);
    const e = energy(t.left) + energy(t.right);
    const corrIn = (dot(t.left, input.left) + dot(t.right, input.right)) / Math.sqrt(e * inE || 1);
    return { ...s, energy: e, rmsDb: db(e / (2 * n)), corrIn };
  });
  const total = rows.reduce((a, r) => a + r.energy, 0) || 1;
  rows.forEach(r => (r.share = (100 * r.energy) / total));

  let errE = 0;
  for (const ch of ["left", "right"]) {
    for (let i = 0; i < n; i++) {
      let sum = 0;
      for (const s of STEMS) sum += stems[s.key][ch][i] || 0;
      const d = input[ch][i] - sum;
      errE += d * d;
    }
  }
  const reconSnrDb = db(inE / (errE || 1e-20));

  let maxPairCorr = 0, maxPair = "";
  for (let i = 0; i < rows.length; i++) for (let j = i + 1; j < rows.length; j++) {
    const a = stems[rows[i].key], b = stems[rows[j].key];
    const c = (dot(a.left, b.left) + dot(a.right, b.right)) / Math.sqrt(rows[i].energy * rows[j].energy || 1);
    if (Math.abs(c) > Math.abs(maxPairCorr)) { maxPairCorr = c; maxPair = `${rows[i].label}–${rows[j].label}`; }
  }
  const maxShare = Math.max(...rows.map(r => r.share));
  const silent = rows.filter(r => r.rmsDb < -80).map(r => r.label);

  const checks = [
    { name: "4개 트랙 합 = 원곡 (복원 SNR ≥ 10dB)", ok: reconSnrDb >= 10, value: `${reconSnrDb.toFixed(1)} dB` },
    { name: "트랙끼리 서로 다른 소리 (최대 상관 < 0.9)", ok: Math.abs(maxPairCorr) < 0.9, value: `${maxPairCorr.toFixed(2)} (${maxPair})` },
    { name: "한 트랙에 소리가 몰리지 않음 (최대 비율 < 95%)", ok: maxShare < 95, value: `${maxShare.toFixed(1)}%` },
    { name: "입력 원곡이 무음이 아님", ok: db(inE / (2 * n)) > -60, value: `${db(inE / (2 * n)).toFixed(1)} dB` },
  ];
  return { rows, reconSnrDb, maxPairCorr, maxPair, silent, checks, pass: checks.every(c => c.ok) };
}

function toInt16(x) { const v = Math.max(-1, Math.min(1, x)); return v < 0 ? v * 0x8000 : v * 0x7fff; }

/** 16bit 스테레오 WAV */
export function encodeWav(left, right, sr) {
  const n = left.length, buf = new ArrayBuffer(44 + n * 4), v = new DataView(buf);
  const w = (o, s) => [...s].forEach((c, i) => v.setUint8(o + i, c.charCodeAt(0)));
  w(0, "RIFF"); v.setUint32(4, 36 + n * 4, true); w(8, "WAVE"); w(12, "fmt ");
  v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, 2, true);
  v.setUint32(24, sr, true); v.setUint32(28, sr * 4, true); v.setUint16(32, 4, true); v.setUint16(34, 16, true);
  w(36, "data"); v.setUint32(40, n * 4, true);
  let o = 44;
  for (let i = 0; i < n; i++) { v.setInt16(o, toInt16(left[i]), true); v.setInt16(o + 2, toInt16(right[i]), true); o += 4; }
  return new Uint8Array(buf);
}

/** lamejs로 스테레오 MP3 (기본 320kbps). 폰이 멈추지 않게 중간중간 쉬어 간다. */
export async function encodeMp3(lamejs, left, right, sr, kbps = 320, onProgress = () => {}) {
  const enc = new lamejs.Mp3Encoder(2, sr, kbps);
  const block = 1152 * 20, out = [];
  const l16 = new Int16Array(block), r16 = new Int16Array(block);
  for (let i = 0; i < left.length; i += block) {
    const m = Math.min(block, left.length - i);
    for (let k = 0; k < m; k++) { l16[k] = toInt16(left[i + k]); r16[k] = toInt16(right[i + k]); }
    const chunk = enc.encodeBuffer(l16.subarray(0, m), r16.subarray(0, m));
    if (chunk.length) out.push(new Uint8Array(chunk));
    if ((i / block) % 10 === 0) { onProgress(i / left.length); await new Promise(r => setTimeout(r, 0)); }
  }
  const end = enc.flush();
  if (end.length) out.push(new Uint8Array(end));
  onProgress(1);
  const size = out.reduce((a, c) => a + c.length, 0), res = new Uint8Array(size);
  let o = 0; for (const c of out) { res.set(c, o); o += c.length; }
  return res;
}
