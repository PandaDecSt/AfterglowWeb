import type { DemoStats } from "../demos/types";

export class PerfHUD {
  private container: HTMLDivElement;
  private fpsEl: HTMLSpanElement;
  private msEl: HTMLSpanElement;
  private gpuEl: HTMLDivElement;
  private statsEl: HTMLDivElement;
  private graphCanvas: HTMLCanvasElement;
  private graphCtx: CanvasRenderingContext2D;

  private frameTimes: number[] = [];
  private maxSamples = 120;
  private lastTime = performance.now();
  private lastFrameEnd = performance.now();
  private frameCount = 0;
  private fpsAccum = 0;
  private currentFps = 0;
  private visible = true;

  constructor(gpuInfo: string) {
    this.container = document.createElement("div");
    Object.assign(this.container.style, {
      position: "fixed",
      bottom: "12px",
      left: "12px",
      zIndex: "9999",
      background: "rgba(10, 10, 18, 0.88)",
      border: "1px solid #333",
      borderRadius: "6px",
      padding: "10px 14px",
      fontFamily: "monospace",
      fontSize: "11px",
      color: "#ccc",
      minWidth: "220px",
      backdropFilter: "blur(4px)",
      pointerEvents: "none",
      lineHeight: "1.6",
    });

    this.fpsEl = document.createElement("span");
    this.fpsEl.style.cssText = "font-size:20px;font-weight:bold;color:#4f4;";
    this.msEl = document.createElement("span");
    this.msEl.style.cssText = "margin-left:8px;color:#aaa;font-size:12px;";

    const row1 = document.createElement("div");
    row1.appendChild(this.fpsEl);
    row1.appendChild(this.msEl);

    this.graphCanvas = document.createElement("canvas");
    this.graphCanvas.width = 200;
    this.graphCanvas.height = 40;
    this.graphCanvas.style.cssText =
      "display:block;margin:6px 0;border:1px solid #333;border-radius:3px;background:#0a0a12;";
    this.graphCtx = this.graphCanvas.getContext("2d")!;

    this.gpuEl = document.createElement("div");
    this.gpuEl.style.cssText = "color:#6af;font-size:10px;margin-bottom:4px;word-break:break-all;";
    this.gpuEl.textContent = gpuInfo;

    this.statsEl = document.createElement("div");
    this.statsEl.style.cssText = "border-top:1px solid #333;padding-top:4px;margin-top:4px;";

    this.container.appendChild(row1);
    this.container.appendChild(this.graphCanvas);
    this.container.appendChild(this.gpuEl);
    this.container.appendChild(this.statsEl);
    document.body.appendChild(this.container);
  }

  begin() {
    this.lastTime = performance.now();
  }

  end(stats?: DemoStats) {
    const now = performance.now();
    const renderTime = now - this.lastTime;
    const frameInterval = now - this.lastFrameEnd;
    this.lastFrameEnd = now;

    this.frameTimes.push(renderTime);
    if (this.frameTimes.length > this.maxSamples) this.frameTimes.shift();

    this.frameCount++;
    this.fpsAccum += frameInterval;
    this.msEl.textContent = `${renderTime.toFixed(2)} ms`;
    if (this.fpsAccum >= 1000) {
      this.currentFps = Math.round((this.frameCount / this.fpsAccum) * 1000);
      this.frameCount = 0;
      this.fpsAccum = 0;
      this.fpsEl.textContent = `${this.currentFps}`;
      this.fpsEl.style.color =
        this.currentFps >= 55 ? "#4f4" : this.currentFps >= 30 ? "#ff4" : "#f44";
    }

    this.drawGraph();
    this.updateStats(stats);
  }

  private drawGraph() {
    const ctx = this.graphCtx;
    const w = this.graphCanvas.width;
    const h = this.graphCanvas.height;
    ctx.clearRect(0, 0, w, h);

    ctx.strokeStyle = "#333";
    ctx.beginPath();
    const y16 = h - (16.67 / 50) * h;
    ctx.moveTo(0, y16);
    ctx.lineTo(w, y16);
    ctx.stroke();

    const n = this.frameTimes.length;
    const barW = w / this.maxSamples;
    for (let i = 0; i < n; i++) {
      const t = this.frameTimes[i];
      const barH = Math.min((t / 50) * h, h);
      const x = i * barW;
      ctx.fillStyle = t < 16.7 ? "#4f4" : t < 33 ? "#ff4" : "#f44";
      ctx.fillRect(x, h - barH, Math.max(barW - 0.5, 1), barH);
    }
  }

  private updateStats(stats?: DemoStats) {
    if (!stats) {
      this.statsEl.textContent = "";
      return;
    }
    const lines: string[] = [];
    if (stats.drawCalls !== undefined) lines.push(`Draw Calls: ${stats.drawCalls}`);
    if (stats.triangles !== undefined) lines.push(`Triangles: ${stats.triangles.toLocaleString()}`);
    if (stats.instances !== undefined) lines.push(`Instances: ${stats.instances.toLocaleString()}`);
    if (stats.computeDispatches !== undefined) lines.push(`Compute: ${stats.computeDispatches}`);
    if (stats.custom) {
      for (const [k, v] of Object.entries(stats.custom)) {
        lines.push(`${k}: ${v}`);
      }
    }
    this.statsEl.innerHTML = lines
      .map((l) => `<div>${l}</div>`)
      .join("");
  }

  toggle() {
    this.visible = !this.visible;
    this.container.style.display = this.visible ? "block" : "none";
  }

  destroy() {
    this.container.remove();
  }
}
