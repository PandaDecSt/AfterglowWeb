# 2026-08-04 Deferred PBR 渲染管线修复与经验记录

## 问题一：黑屏（深度全部被判为天空）

### 现象
GBuffer + Lighting 后输出纯黑（实际是极暗的 `(0.02, 0.02, 0.04)`）。

### 根因
GBuffer 顶点着色器写入深度拷贝纹理时：

```wgsl
// 错误：@builtin(position).z 在 WebGPU 中已经是窗口/NDC 深度 [0,1]
out.depthCopy = vec4<f32>(in.position.z / in.position.w, 0, 0, 0);
```

WebGPU 中 `@builtin(position)` 在 **fragment stage 已经是透视除法后的窗口坐标**，`.z` ∈ [0,1]（已经做过 /w）。再除以 `.w`（此时为 1.0 但 shader 里实际是视觉深度的倒数含义，随距离变小）会把这个值**膨胀**，导致所有像素 depth ≥ 1.0，被 lighting shader 的 `depth >= 1.0` 判定为天空，走了暗色天空分支。

### 修复
```wgsl
out.depthCopy = vec4<f32>(in.position.z, 0, 0, 0);
```

### 定位方法
写诊断模式（天空红 / 物体绿）确认 ALL 像素走天空分支，从而锁定是深度判断永真，而非采样/混合问题。

### 教训
1. **WebGPU fragment 的 `@builtin(position)` 已是投影后的窗口深度 [0,1]，不要再次除以 `.w`**。OpenGL 习惯 (`gl_FragCoord.z` 同理) 与 WebGPU 差异不大，但若同时用到 `in.position.w` 需明确它已被清除，不要误当 clip z。
2. 排障时用**二分量诊断着色器（天空一色、物体另一色）**快速区分是"分支判断错误"还是"数据没传对"。

## 问题二：bind group layout 不匹配（layout:"auto" 剔除未用 binding）

### 现象
`deferred-lighting` pipeline 报 `binding index 1 not present in the bind group layout`。

### 根因
用 `layout: "auto"` 创建 pipeline 时，Chrome 会根据 shader 实际用到的 binding 生成 layout；当 shader 某分支持换（如关 CSM/SSAO 分支）用不到 binding 1（lights），layout 里就没有 binding 1，但代码仍按固定 BGL 建 bind group → 校验失败、pipeline 缓存命中错误 layout。

### 修复
`deferred-lighting.ts` 改为**显式 `GPUBindGroupLayout`** 创建 pipeline：

```typescript
const bgl = this.device.createBindGroupLayout({ entries: [ ... ] });
pipeline = this.device.createRenderPipeline({ layout: this.device.createPipelineLayout({ bindGroupLayouts: [bgl] }), ... });
```

并在开关状态变化时重建 pipeline：

```typescript
private cachedHasShadow = false;
if (this.useCSM !== this.cachedHasShadow) { this.cachedHasShadow = useCSM; this.createLightingPipeline(); }
```

### 教训
3. **不要对"有可选 binding 与 shader 分支绑定"的 pipeline 用 `layout: "auto"`**。auto 会根据当前编译的变体剔除未用 binding，导致固定 BindGroup 失配。显式声明 BGL + 状态变化时重建 pipeline 最稳。

## 问题三：depth24plus-stencil8 无法用 depth24plus 子格式 view 采样

### 现象
GBuffer 深度纹理用 `depth24plus-stencil8` 创建，post-process 需要以 depth 采样绑定。尝试：

```typescript
// 第一次：直接 createView({format:"depth24plus"})
// → The texture view format (Depth24Plus) is not compatible with the texture format (Depth24PlusStencil8)

// 第二次：createTexture 加 viewFormats:["depth24plus"]
// → The texture view format (Depth24Plus) is not texture view format compatible
//    with the texture format (Depth24PlusStencil8) on this device
```

当前设备实现不允许 `depth24plus-stencil8` 的任何 sub-view。

### 修复
既然 demo 没用 stencil，直接改纯深度格式：

```typescript
static readonly DEPTH_FORMAT: GPUTextureFormat = "depth24plus";          // 不再用 stencil
this.depthView = this.depthTexture.createView();
this.depthSampledView = this.depthView;                                  // 默认 view 就是 depth24plus，可直接绑定 texture_depth_2d
```

且同步**移除 depthStencilAttachment 中的 stencil 字段**（非 stencil 格式禁止提供 stencilLoadOp/stencilStoreOp/stencilClearValue）。

### 教训
4. **需要给 post-processing 采样深度时，直接用纯 `depth24plus` 深度纹理，别用 `depth24plus-stencil8`**。后者要么靠 `viewFormats` 声明子 view（兼容性因设备而异，当前环境不支持），要么只能在 stub 里 stencil 部分。若确需 stencil，就要在 createTexture 的 `viewFormats` 里统一声明并在所有地方用子 view。

## 问题四：GUI 只有 Metallic/Roughness 有反应

### 现象
Deferred PBR demo 的 Bloom/Exposure/Saturation/Vignette 等参数调整无任何效果。

### 根因
`bloomPass` 与 `postProcessPass` 在 `init()` 中创建但**从未接入渲染管线**——`createPasses()` 最后一个步骤把 lighting 结果直接 blit 到屏幕，bloom/post-process 全被跳过。

### 修复（接线）
1. `gbuffer.ts`：depth 纹理 usage 加 `TEXTURE_BINDING`，提供可直接采样的 `depthSampledView`。
2. `post-process.ts`：`execute()` 参数由 `depthTexture: GPUTexture` 改为 `depthView: GPUTextureView`；bind group 缓存键改为 `(sceneTexture, depthView)`。
3. `deferred-pbr.ts`：
   - `createPasses()` 末尾改为：`bloomPass.execute(lightingRT)` → `bloomPass.combine(..., bloomResult.view, bloomCombineRT.view, bloomIntensity)` → `postProcessPass.execute(bloomCombineRT.texture, gbuffer.depthSampledView, screenView, [w,h], time)`。
   - `update()` 中每帧喂 postProcess 的 `cameraPos`、`invVP`（mat4.inverse(viewProj)）。CSI 代码重用了同一份 invViewProj。
   - 删除了临时 inline blit shader（该 inline blit 曾把 `tex`/`samp` 写进同一个 shader 字符串并拼 `w`/`h` 浮点）。

### 教训
5. **创建了 pass 不代表接入了管线**。排查"调参无反应"时，先确认 pass 是否真的被 `execute` 调用，再怀疑 shader 参数映射。本类问题用字符串搜索 `passName.execute` 的调用点即可一锤定音。
6. post-process 用到的相机数据（cameraPos、invVP）必须每帧更新，且 invVP 应复用 lighting/CSM 已经算好的矩阵，避免重复求逆。
7. inline blit shader 里 `vec2<f32>(${w}.0, ${h}.0)` 这种字符串拼接的浮点模板写法易出错，接入正式 post-process 后一并移除。