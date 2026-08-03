import { decodeTGA } from "./tga-loader";

export function create1x1Texture(device: GPUDevice, r: number, g: number, b: number, a: number, label: string): GPUTexture {
  const tex = device.createTexture({ label, size: [1, 1], format: "rgba8unorm-srgb", usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
  device.queue.writeTexture({ texture: tex }, new Uint8Array([r, g, b, a]), { bytesPerRow: 4 }, [1, 1]);
  return tex;
}

export function createToonRampTexture(device: GPUDevice): GPUTexture {
  const h = 64;
  const data = new Uint8Array(h * 4);
  for (let y = 0; y < h; y++) {
    const v = y / (h - 1);
    const t = Math.min(1, Math.max(0, (v - 0.5) / 0.1));
    const s = t * t * (3 - 2 * t);
    data[y * 4 + 0] = Math.round(255 - s * (255 - 196));
    data[y * 4 + 1] = Math.round(255 - s * (255 - 186));
    data[y * 4 + 2] = Math.round(255 - s * (255 - 205));
    data[y * 4 + 3] = 255;
  }
  const tex = device.createTexture({ label: "toon-ramp", size: [1, h], format: "rgba8unorm-srgb", usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
  device.queue.writeTexture({ texture: tex }, data, { bytesPerRow: 4 }, [1, h]);
  return tex;
}

export async function loadTextureImage(device: GPUDevice, url: string, label: string): Promise<GPUTexture | null> {
  try {
    if (url.toLowerCase().endsWith(".tga")) {
      const resp = await fetch(url);
      if (!resp.ok) return null;
      const buffer = await resp.arrayBuffer();
      const tga = decodeTGA(buffer);
      const tex = device.createTexture({
        label,
        size: [tga.width, tga.height],
        format: "rgba8unorm-srgb",
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
      });
      const tgaBytes = new Uint8Array(tga.data.buffer.slice(tga.data.byteOffset, tga.data.byteOffset + tga.data.byteLength));
      device.queue.writeTexture({ texture: tex }, tgaBytes as unknown as GPUAllowSharedBufferSource, { bytesPerRow: tga.width * 4 }, [tga.width, tga.height]);
      return tex;
    }
    const resp = await fetch(url);
    if (!resp.ok) return null;
    const blob = await resp.blob();
    const bitmap = await createImageBitmap(blob);
    const tex = device.createTexture({ label, size: [bitmap.width, bitmap.height], format: "rgba8unorm-srgb", usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT });
    device.queue.copyExternalImageToTexture({ source: bitmap }, { texture: tex }, [bitmap.width, bitmap.height]);
    bitmap.close();
    return tex;
  } catch { return null; }
}