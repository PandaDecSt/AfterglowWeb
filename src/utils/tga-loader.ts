export interface TGAImage {
  width: number;
  height: number;
  data: Uint8Array;
  hasAlpha: boolean;
}

export function decodeTGA(buffer: ArrayBuffer): TGAImage {
  const view = new DataView(buffer);
  const length = buffer.byteLength;

  const idLength = view.getUint8(0);
  const colorMapType = view.getUint8(1);
  const imageType = view.getUint8(2);
  const pixelDepth = view.getUint8(16);
  const imageDescriptor = view.getUint8(17);

  const width = view.getUint16(12, true);
  const height = view.getUint16(14, true);

  let offset = 18 + idLength;

  if (colorMapType === 1) {
    const cmLength = view.getUint16(5, true);
    const cmEntrySize = view.getUint8(7);
    offset += cmLength * Math.ceil(cmEntrySize / 8);
  }

  const bpp = pixelDepth;
  const bytePerPixel = bpp / 8;
  const pixelCount = width * height;
  const hasAlpha = bpp === 32;
  const data = new Uint8Array(pixelCount * 4);

  if (imageType === 2) {
    for (let i = 0; i < pixelCount; i++) {
      const srcOff = offset + i * bytePerPixel;
      if (srcOff + bytePerPixel > length) break;
      const dstOff = i * 4;
      if (bpp === 32) {
        data[dstOff] = view.getUint8(srcOff + 2);
        data[dstOff + 1] = view.getUint8(srcOff + 1);
        data[dstOff + 2] = view.getUint8(srcOff);
        data[dstOff + 3] = view.getUint8(srcOff + 3);
      } else if (bpp === 24) {
        data[dstOff] = view.getUint8(srcOff + 2);
        data[dstOff + 1] = view.getUint8(srcOff + 1);
        data[dstOff + 2] = view.getUint8(srcOff);
        data[dstOff + 3] = 255;
      }
    }
  } else if (imageType === 10) {
    let pixelIndex = 0;
    let srcOff = offset;

    while (pixelIndex < pixelCount && srcOff < length) {
      const header = view.getUint8(srcOff++);
      const count = (header & 0x7F) + 1;

      if (header & 0x80) {
        if (srcOff + bytePerPixel > length) break;
        const r = bpp >= 24 ? view.getUint8(srcOff + 2) : 0;
        const g = bpp >= 24 ? view.getUint8(srcOff + 1) : 0;
        const b = bpp >= 24 ? view.getUint8(srcOff) : 0;
        const a = hasAlpha ? view.getUint8(srcOff + 3) : 255;
        srcOff += bytePerPixel;

        for (let j = 0; j < count && pixelIndex < pixelCount; j++, pixelIndex++) {
          const dstOff = pixelIndex * 4;
          data[dstOff] = r;
          data[dstOff + 1] = g;
          data[dstOff + 2] = b;
          data[dstOff + 3] = a;
        }
      } else {
        for (let j = 0; j < count && pixelIndex < pixelCount; j++, pixelIndex++) {
          if (srcOff + bytePerPixel > length) break;
          const dstOff = pixelIndex * 4;
          data[dstOff] = bpp >= 24 ? view.getUint8(srcOff + 2) : 0;
          data[dstOff + 1] = bpp >= 24 ? view.getUint8(srcOff + 1) : 0;
          data[dstOff + 2] = bpp >= 24 ? view.getUint8(srcOff) : 0;
          data[dstOff + 3] = hasAlpha ? view.getUint8(srcOff + 3) : 255;
          srcOff += bytePerPixel;
        }
      }
    }
  }

  const topOrigin = !(imageDescriptor & 0x20);
  if (!topOrigin) {
    const rowBytes = width * 4;
    const row = new Uint8Array(rowBytes);
    for (let y = 0; y < Math.floor(height / 2); y++) {
      const topOff = y * rowBytes;
      const botOff = (height - 1 - y) * rowBytes;
      row.set(data.subarray(topOff, topOff + rowBytes));
      data.copyWithin(topOff, botOff, botOff + rowBytes);
      data.set(row, botOff);
    }
  }

  return { width, height, data, hasAlpha };
}

export async function loadTGA(url: string): Promise<TGAImage> {
  const response = await fetch(url);
  const buffer = await response.arrayBuffer();
  return decodeTGA(buffer);
}