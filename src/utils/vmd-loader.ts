export interface VMDBoneFrame {
  frame: number;
  rotation: [number, number, number, number];
  translation: [number, number, number];
  interpolation: Uint8Array;
}

export interface VMDMorphFrame {
  frame: number;
  weight: number;
}

export interface VMDCameraFrame {
  frame: number;
  distance: number;
  target: [number, number, number];
  rotation: [number, number, number];
  fov: number;
  interpolation: Uint8Array;
}

export interface VMDData {
  name: string;
  boneFrames: Map<string, VMDBoneFrame[]>;
  morphFrames: Map<string, VMDMorphFrame[]>;
  cameraFrames: VMDCameraFrame[];
  maxFrame: number;
}

const FRAME_RATE = 30.0;

export function normalizeBoneName(name: string): string {
  let out = "";
  for (let i = 0; i < name.length; i++) {
    const c = name.charCodeAt(i);
    if (c >= 0xff01 && c <= 0xff5e) {
      out += String.fromCharCode(c - 0xfee0);
    } else if (c === 0x3000) {
      out += " ";
    } else {
      out += name[i];
    }
  }
  return out;
}

export async function loadVMD(url: string): Promise<VMDData> {
  const resp = await fetch(url);
  if (!resp.ok) throw new Error(`Failed to fetch VMD: ${url}`);
  return parseVMD(await resp.arrayBuffer());
}

export function parseVMD(buffer: ArrayBuffer): VMDData {
  const view = new DataView(buffer);
  let off = 0;

  const u8 = () => { const v = view.getUint8(off); off += 1; return v; };
  const u32 = () => { const v = view.getUint32(off, true); off += 4; return v; };
  const f32 = () => { const v = view.getFloat32(off, true); off += 4; return v; };

  let decoder: TextDecoder;
  try { decoder = new TextDecoder("shift-jis"); } catch { decoder = new TextDecoder("utf-8"); }

  const readName = (len: number): string => {
    const buf = new Uint8Array(buffer, off, len);
    off += len;
    let end = len;
    for (let i = 0; i < len; i++) { if (buf[i] === 0) { end = i; break; } }
    try { return decoder.decode(buf.slice(0, end)); } catch { return String.fromCharCode(...buf.slice(0, end)); }
  };

  const readAscii = (len: number): string => {
    const buf = new Uint8Array(buffer, off, len);
    off += len;
    return String.fromCharCode(...buf);
  };

  const header = readAscii(30);
  if (!header.startsWith("Vocaloid Motion Data")) throw new Error("Invalid VMD header");
  const modelName = readName(20);

  const boneCount = u32();
  const boneFrames = new Map<string, VMDBoneFrame[]>();

  for (let i = 0; i < boneCount; i++) {
    const boneName = normalizeBoneName(readName(15));
    const frame = u32();
    const tx = f32(), ty = f32(), tz = f32();
    const rx = f32(), ry = f32(), rz = f32(), rw = f32();
    const interp = new Uint8Array(64);
    for (let j = 0; j < 64; j++) interp[j] = u8();

    let arr = boneFrames.get(boneName);
    if (!arr) { arr = []; boneFrames.set(boneName, arr); }
    arr.push({ frame, rotation: [rx, ry, rz, rw], translation: [tx, ty, tz], interpolation: interp });
  }

  const morphCount = u32();
  const morphFrames = new Map<string, VMDMorphFrame[]>();

  for (let i = 0; i < morphCount; i++) {
    const morphName = normalizeBoneName(readName(15));
    const frame = u32();
    const weight = f32();

    let arr = morphFrames.get(morphName);
    if (!arr) { arr = []; morphFrames.set(morphName, arr); }
    arr.push({ frame, weight });
  }

  for (const frames of boneFrames.values()) frames.sort((a, b) => a.frame - b.frame);
  for (const frames of morphFrames.values()) frames.sort((a, b) => a.frame - b.frame);

  let maxFrame = 0;
  for (const frames of boneFrames.values()) {
    if (frames.length > 0) maxFrame = Math.max(maxFrame, frames[frames.length - 1].frame);
  }
  for (const frames of morphFrames.values()) {
    if (frames.length > 0) maxFrame = Math.max(maxFrame, frames[frames.length - 1].frame);
  }

  const cameraFrames: VMDCameraFrame[] = [];
  if (off + 4 <= buffer.byteLength) {
    const cameraCount = u32();
    for (let i = 0; i < cameraCount; i++) {
      const frame = u32();
      const distance = f32();
      const tx = f32(), ty = f32(), tz = f32();
      const rx = f32(), ry = f32(), rz = f32();
      const interp = new Uint8Array(24);
      for (let j = 0; j < 24; j++) interp[j] = u8();
      const fov = u32();
      u8();
      cameraFrames.push({ frame, distance, target: [tx, ty, tz], rotation: [rx, ry, rz], fov, interpolation: interp });
    }
    cameraFrames.sort((a, b) => a.frame - b.frame);
    if (cameraFrames.length > 0) maxFrame = Math.max(maxFrame, cameraFrames[cameraFrames.length - 1].frame);
  }

  return { name: modelName, boneFrames, morphFrames, cameraFrames, maxFrame };
}

export function vmdDuration(data: VMDData): number {
  return data.maxFrame / FRAME_RATE;
}

export function evalBezier(ax: number, ay: number, bx: number, by: number, t: number): number {
  if (ax === ay && bx === by) return t;
  let lo = 0.0, hi = 1.0;
  for (let i = 0; i < 16; i++) {
    const mid = (lo + hi) * 0.5;
    const u = 1.0 - mid;
    const x = 3.0 * u * u * mid * ax + 3.0 * u * mid * mid * bx + mid * mid * mid;
    if (x < t) lo = mid; else hi = mid;
  }
  const s = (lo + hi) * 0.5;
  const u = 1.0 - s;
  return 3.0 * u * u * s * ay + 3.0 * u * s * s * by + s * s * s;
}

export function sampleVMDBone(frames: VMDBoneFrame[], frameNum: number, out: { rotation: [number, number, number, number]; translation: [number, number, number] }):
  void {

  if (frames.length === 0) { out.rotation[0]=0; out.rotation[1]=0; out.rotation[2]=0; out.rotation[3]=1; out.translation[0]=0; out.translation[1]=0; out.translation[2]=0; return; }
  if (frames.length === 1) { out.rotation[0]=frames[0].rotation[0]; out.rotation[1]=frames[0].rotation[1]; out.rotation[2]=frames[0].rotation[2]; out.rotation[3]=frames[0].rotation[3]; out.translation[0]=frames[0].translation[0]; out.translation[1]=frames[0].translation[1]; out.translation[2]=frames[0].translation[2]; return; }
  if (frameNum <= frames[0].frame) { out.rotation[0]=frames[0].rotation[0]; out.rotation[1]=frames[0].rotation[1]; out.rotation[2]=frames[0].rotation[2]; out.rotation[3]=frames[0].rotation[3]; out.translation[0]=frames[0].translation[0]; out.translation[1]=frames[0].translation[1]; out.translation[2]=frames[0].translation[2]; return; }
  const last = frames[frames.length - 1];
  if (frameNum >= last.frame) { out.rotation[0]=last.rotation[0]; out.rotation[1]=last.rotation[1]; out.rotation[2]=last.rotation[2]; out.rotation[3]=last.rotation[3]; out.translation[0]=last.translation[0]; out.translation[1]=last.translation[1]; out.translation[2]=last.translation[2]; return; }

  let lo = 0, hi = frames.length - 1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (frames[mid].frame <= frameNum) lo = mid; else hi = mid;
  }

  const f0 = frames[lo], f1 = frames[hi];
  const dt = f1.frame - f0.frame;
  if (dt <= 0) { out.rotation[0]=f0.rotation[0]; out.rotation[1]=f0.rotation[1]; out.rotation[2]=f0.rotation[2]; out.rotation[3]=f0.rotation[3]; out.translation[0]=f0.translation[0]; out.translation[1]=f0.translation[1]; out.translation[2]=f0.translation[2]; return; }

  const t = (frameNum - f0.frame) / dt;
  const ip = f0.interpolation;

  const tx = evalBezier(ip[0] / 127.0, ip[4] / 127.0, ip[8] / 127.0, ip[12] / 127.0, t);
  const ty = evalBezier(ip[1] / 127.0, ip[5] / 127.0, ip[9] / 127.0, ip[13] / 127.0, t);
  const tz = evalBezier(ip[2] / 127.0, ip[6] / 127.0, ip[10] / 127.0, ip[14] / 127.0, t);
  const tr = evalBezier(ip[3] / 127.0, ip[7] / 127.0, ip[11] / 127.0, ip[15] / 127.0, t);

  const px = f0.translation[0] + (f1.translation[0] - f0.translation[0]) * tx;
  const py = f0.translation[1] + (f1.translation[1] - f0.translation[1]) * ty;
  const pz = f0.translation[2] + (f1.translation[2] - f0.translation[2]) * tz;

  const q0 = f0.rotation, q1 = f1.rotation;
  let dot = q0[0] * q1[0] + q0[1] * q1[1] + q0[2] * q1[2] + q0[3] * q1[3];
  const sign = dot >= 0 ? 1 : -1;
  dot = Math.abs(dot);

  if (dot > 0.9995) {
    const rx = q0[0] + (q1[0] * sign - q0[0]) * tr;
    const ry = q0[1] + (q1[1] * sign - q0[1]) * tr;
    const rz = q0[2] + (q1[2] * sign - q0[2]) * tr;
    const rw = q0[3] + (q1[3] * sign - q0[3]) * tr;
    const len = Math.sqrt(rx * rx + ry * ry + rz * rz + rw * rw) || 1;
    out.rotation[0] = rx / len; out.rotation[1] = ry / len; out.rotation[2] = rz / len; out.rotation[3] = rw / len;
    out.translation[0] = px; out.translation[1] = py; out.translation[2] = pz;
    return;
  }

  const theta = Math.acos(dot);
  const sinTheta = Math.sin(theta);
  const s0 = Math.abs(sinTheta) < 1e-6 ? 0.5 : Math.sin((1.0 - tr) * theta) / sinTheta;
  const s1 = Math.abs(sinTheta) < 1e-6 ? 0.5 : Math.sin(tr * theta) / sinTheta;
  const rx = s0 * q0[0] + s1 * q1[0] * sign;
  const ry = s0 * q0[1] + s1 * q1[1] * sign;
  const rz = s0 * q0[2] + s1 * q1[2] * sign;
  const rw = s0 * q0[3] + s1 * q1[3] * sign;
  const len = Math.sqrt(rx * rx + ry * ry + rz * rz + rw * rw) || 1;
  out.rotation[0] = rx / len; out.rotation[1] = ry / len; out.rotation[2] = rz / len; out.rotation[3] = rw / len;
  out.translation[0] = px; out.translation[1] = py; out.translation[2] = pz;
}

export function sampleVMDMorph(frames: VMDMorphFrame[], frameNum: number): number {
  if (frames.length === 0) return 0;
  if (frames.length === 1) return frames[0].weight;
  if (frameNum <= frames[0].frame) return frames[0].weight;
  if (frameNum >= frames[frames.length - 1].frame) return frames[frames.length - 1].weight;

  let lo = 0, hi = frames.length - 1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (frames[mid].frame <= frameNum) lo = mid; else hi = mid;
  }

  const f0 = frames[lo], f1 = frames[hi];
  const dt = f1.frame - f0.frame;
  if (dt <= 0) return f0.weight;
  const t = (frameNum - f0.frame) / dt;
  return f0.weight + (f1.weight - f0.weight) * t;
}
