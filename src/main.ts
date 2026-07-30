import { GPUContext } from "./core/device";
import { EngineContext } from "./core/engine";
import { Renderer } from "./core/renderer";
import { Camera } from "./scene/camera";
import { PerfHUD } from "./ui/perf-hud";
import { DebugPanel } from "./ui/debug-panel";
import {
  Demo,
  FractalNoiseDemo,
  ParticleDemo,
  BoidDemo,
  ToonDemo,
  ArcToonDemo,
  ShowcaseDemo,
  IndirectDrawDemo,
  PBRShadowDemo,
  MeshGenDemo,
  GLBViewerDemo,
  GrassDemo,
  WaterDemo,
  ShellFurDemo,
  GlitchedMosaicsDemo,
  GreedySnakeDemo,
  MeteorographDemo,
  ToonTexturedDemo,
  TerrainHydrologyDemo,
  ACESPipelineDemo,
  HairEyeShadowDemo,
} from "./demos";

const canvas = document.getElementById("canvas") as HTMLCanvasElement;
const errorOverlay = document.getElementById("error-overlay")!;

function showError(msg: string) {
  errorOverlay.textContent = msg;
  errorOverlay.style.display = "block";
}

const demoFactories: { label: string; create: () => Demo }[] = [
  { label: "GLB Viewer (Qin_DL)", create: () => new GLBViewerDemo() },
  { label: "PBR + Shadow", create: () => new PBRShadowDemo() },
  { label: "Toon (Multi-Material)", create: () => new ToonDemo() },
  { label: "Arc Toon (ILM+SSS)", create: () => new ArcToonDemo() },
  { label: "Grass", create: () => new GrassDemo() },
  { label: "Water", create: () => new WaterDemo() },
  { label: "ShellFur", create: () => new ShellFurDemo() },
  { label: "Particles", create: () => new ParticleDemo() },
  { label: "Boid", create: () => new BoidDemo() },
  { label: "MeshGen (CS→Indirect)", create: () => new MeshGenDemo() },
  { label: "IndirectDraw (10k Culling)", create: () => new IndirectDrawDemo() },
  { label: "GlitchedMosaics", create: () => new GlitchedMosaicsDemo() },
  { label: "GreedySnake", create: () => new GreedySnakeDemo() },
  { label: "FractalNoise", create: () => new FractalNoiseDemo() },
  { label: "Meteorograph (Weather)", create: () => new MeteorographDemo() },
  { label: "Toon Textured (Ramp)", create: () => new ToonTexturedDemo() },
  { label: "Terrain Hydrology", create: () => new TerrainHydrologyDemo() },
  { label: "ACES Pipeline", create: () => new ACESPipelineDemo() },
  { label: "Hair/Eye Shadow", create: () => new HairEyeShadowDemo() },
  { label: "Showcase (Full Pipeline)", create: () => new ShowcaseDemo() },
];

async function main() {
  let ctx: GPUContext;
  try {
    ctx = await GPUContext.create(canvas);
  } catch (e) {
    showError(String(e));
    return;
  }

  ctx.resize(window.innerWidth, window.innerHeight);
  window.addEventListener("resize", () =>
    ctx.resize(window.innerWidth, window.innerHeight)
  );

  const engine = new EngineContext(ctx);
  const camera = new Camera(canvas);
  const renderer = new Renderer(ctx);

  engine.modules.registerModule("pbr", PBR_MODULE);

  let gpuInfo = "WebGPU";
  try {
    const adapter = await navigator.gpu.requestAdapter();
    if (adapter) {
      const a = adapter as any;
      const info = a.info ?? (a.requestAdapterInfo ? await a.requestAdapterInfo() : null);
      if (info) {
        gpuInfo = `${info.vendor || "GPU"} | ${info.architecture || ""} | ${info.description || "WebGPU"}`;
      }
    }
  } catch { /* fallback */ }

  const hud = new PerfHUD(gpuInfo);

  engine.shaderReload.createEditor(document.body);
  const panel = new DebugPanel(engine.shaderReload, camera);

  let demoFolder: ReturnType<typeof panel.addFolder> | null = null;
  let currentDemo: Demo | null = null;

  function registerDemoShaders(demo: Demo) {
    engine.shaderReload.clear();
    if (!demo.getShaderStages) return;

    const stages = demo.getShaderStages();
    for (const stage of stages) {
      engine.shaderReload.register(stage.label, stage.code, stage.type);
      if (demo.onShaderReload) {
        engine.shaderReload.onReload(stage.label, (src) => {
          return demo.onShaderReload!(src.label, src.code);
        });
      }
    }
    panel.refreshShaderList();
  }

  function switchDemo(index: number) {
    renderer.stop();
    renderer.clearPasses();

    if (currentDemo) {
      currentDemo.destroy();
      currentDemo = null;
    }
    if (demoFolder) {
      demoFolder.destroy();
      demoFolder = null;
    }
    engine.shaderReload.closeEditor();

    const demo = demoFactories[index].create();
    const result = demo.init(ctx, camera, engine);

    const setupDemo = () => {
      registerDemoShaders(demo);

      const passes = demo.createPasses();
      for (const pass of passes) {
        renderer.addPass(pass);
      }

      if (demo.registerGUI) {
        if (demoFolder) { demoFolder.destroy(); demoFolder = null; }
        demoFolder = panel.addFolder(demo.label);
        demo.registerGUI(demoFolder);
      }

      renderer.resetTime();
      renderer.start();
    };

    if (result instanceof Promise) {
      result.then(setupDemo);
    } else {
      setupDemo();
    }
    currentDemo = demo;
  }

  // Wire demo lifecycle into renderer hooks
  renderer.onUpdate = (renderCtx) => {
    if (!currentDemo) return;
    hud.begin();
    currentDemo.update(renderCtx.time, renderCtx.deltaTime);
  };

  renderer.onPostSubmit = () => {
    if (!currentDemo) return;
    hud.end(currentDemo.stats?.());
  };

  window.addEventListener("keydown", (e) => {
    if (e.key === "h" || e.key === "H") hud.toggle();
    if (e.key === "g" || e.key === "G") panel.show(panel.hidden);
  });

  const selector = document.createElement("div");
  Object.assign(selector.style, {
    position: "fixed",
    top: "12px",
    left: "12px",
    zIndex: "9999",
    display: "flex",
    gap: "6px",
    flexWrap: "wrap",
    maxWidth: "400px",
  });

  demoFactories.forEach((d, i) => {
    const btn = document.createElement("button");
    btn.textContent = d.label;
    Object.assign(btn.style, {
      padding: "6px 12px",
      background: i === 0 ? "#4a9eff" : "#2d2d3d",
      color: "#fff",
      border: "1px solid #555",
      borderRadius: "4px",
      cursor: "pointer",
      fontSize: "12px",
      fontFamily: "monospace",
    });
    btn.addEventListener("mouseenter", () => {
      btn.style.background = "#4a9eff";
    });
    btn.addEventListener("mouseleave", () => {
      btn.style.background = btn.dataset.active ? "#4a9eff" : "#2d2d3d";
    });
    btn.addEventListener("click", () => {
      selector.querySelectorAll("button").forEach((b) => {
        b.style.background = "#2d2d3d";
        delete b.dataset.active;
      });
      btn.style.background = "#4a9eff";
      btn.dataset.active = "1";
      switchDemo(i);
    });
    selector.appendChild(btn);
  });

  document.body.appendChild(selector);

  const hint = document.createElement("div");
  hint.textContent =
    "[H] HUD  [G] GUI  [Drag] Orbit  [Scroll] Zoom  [Ctrl+Enter] Apply Shader  [Esc] Close Editor";
  Object.assign(hint.style, {
    position: "fixed",
    bottom: "12px",
    right: "12px",
    zIndex: "9999",
    fontFamily: "monospace",
    fontSize: "10px",
    color: "#666",
    pointerEvents: "none",
  });
  document.body.appendChild(hint);

  switchDemo(0);

  console.log("[AfterglowWeb] Engine started. Press H for HUD, G for GUI.");
}

const PBR_MODULE = `
const PI: f32 = 3.14159265359;

fn distributionGGX(N: vec3<f32>, H: vec3<f32>, roughness: f32) -> f32 {
  let a = roughness * roughness;
  let a2 = a * a;
  let NdotH = max(dot(N, H), 0.0);
  let NdotH2 = NdotH * NdotH;
  let denom = NdotH2 * (a2 - 1.0) + 1.0;
  return a2 / (PI * denom * denom);
}

fn geometrySchlickGGX(NdotV: f32, roughness: f32) -> f32 {
  let r = roughness + 1.0;
  let k = (r * r) / 8.0;
  return NdotV / (NdotV * (1.0 - k) + k);
}

fn geometrySmith(N: vec3<f32>, V: vec3<f32>, L: vec3<f32>, roughness: f32) -> f32 {
  return geometrySchlickGGX(max(dot(N, V), 0.0), roughness) * geometrySchlickGGX(max(dot(N, L), 0.0), roughness);
}

fn fresnelSchlick(cosTheta: f32, F0: vec3<f32>) -> vec3<f32> {
  return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

fn cookTorrance(N: vec3<f32>, V: vec3<f32>, L: vec3<f32>, baseColor: vec3<f32>, metallic: f32, roughness: f32) -> vec3<f32> {
  let H = normalize(V + L);
  let F0 = mix(vec3<f32>(0.04), baseColor, metallic);
  let NDF = distributionGGX(N, H, roughness);
  let G = geometrySmith(N, V, L, roughness);
  let F = fresnelSchlick(max(dot(H, V), 0.0), F0);
  let numerator = NDF * G * F;
  let denominator = 4.0 * max(dot(N, V), 0.0) * max(dot(N, L), 0.0) + 0.0001;
  let specular = numerator / denominator;
  let kD = (vec3<f32>(1.0) - F) * (1.0 - metallic);
  let NdotL = max(dot(N, L), 0.0);
  return (kD * baseColor / PI + specular) * NdotL;
}
`;

main();
