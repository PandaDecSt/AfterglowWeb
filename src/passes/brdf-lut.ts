import { BRDF_LUT_BAKE_WGSL, BRDF_LUT_SIZE } from "../shader/dfg-lut";
import { LTC_MAG_LUT_DATA, LTC_MAG_LUT_SIZE } from "../shader/ltc-mag-lut";

export { BRDF_LUT_SIZE };

export class BrdfLut {
  texture!: GPUTexture;
  view!: GPUTextureView;

  bake(device: GPUDevice): void {
    this.texture?.destroy();

    if (BRDF_LUT_SIZE !== LTC_MAG_LUT_SIZE) {
      throw new Error("BRDF LUT bake requires DFG size == LTC size (both 64).");
    }

    const ltcTemp = device.createTexture({
      label: "LTC mag LUT (bake input)",
      size: [LTC_MAG_LUT_SIZE, LTC_MAG_LUT_SIZE],
      format: "rg16float",
      usage: GPUTextureUsage.COPY_DST | GPUTextureUsage.TEXTURE_BINDING,
    });

    const n = LTC_MAG_LUT_DATA.length;
    const half = new Uint16Array(n);
    const f32 = new Float32Array(1);
    const u32 = new Uint32Array(f32.buffer);
    for (let i = 0; i < n; i++) {
      f32[0] = LTC_MAG_LUT_DATA[i];
      const x = u32[0];
      const sign = (x >>> 16) & 0x8000;
      let exp = ((x >>> 23) & 0xff) - 127 + 15;
      const mant = x & 0x7fffff;
      if (exp <= 0) {
        half[i] = sign;
      } else if (exp >= 31) {
        half[i] = sign | 0x7c00;
      } else {
        half[i] = sign | (exp << 10) | (mant >>> 13);
      }
    }
    device.queue.writeTexture(
      { texture: ltcTemp },
      half,
      { bytesPerRow: LTC_MAG_LUT_SIZE * 4, rowsPerImage: LTC_MAG_LUT_SIZE },
      { width: LTC_MAG_LUT_SIZE, height: LTC_MAG_LUT_SIZE, depthOrArrayLayers: 1 },
    );

    this.texture = device.createTexture({
      label: "BRDF LUT (DFG + LTC packed)",
      size: [BRDF_LUT_SIZE, BRDF_LUT_SIZE],
      format: "rgba8unorm",
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    });
    this.view = this.texture.createView();

    const module = device.createShaderModule({ label: "BRDF LUT bake", code: BRDF_LUT_BAKE_WGSL });
    const pipeline = device.createRenderPipeline({
      label: "BRDF LUT bake pipeline",
      layout: "auto",
      vertex: { module, entryPoint: "vs" },
      fragment: { module, entryPoint: "fs", targets: [{ format: "rgba8unorm" }] },
      primitive: { topology: "triangle-list" },
    });

    const bakeBindGroup = device.createBindGroup({
      label: "BRDF LUT bake bind group",
      layout: pipeline.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: ltcTemp.createView() }],
    });

    const enc = device.createCommandEncoder({ label: "BRDF LUT bake encoder" });
    const pass = enc.beginRenderPass({
      colorAttachments: [
        { view: this.view, clearValue: { r: 0, g: 0, b: 0, a: 1 }, loadOp: "clear", storeOp: "store" },
      ],
    });
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bakeBindGroup);
    pass.draw(3, 1, 0, 0);
    pass.end();
    device.queue.submit([enc.finish()]);

    ltcTemp.destroy();
  }

  destroy(): void {
    this.texture?.destroy();
  }
}