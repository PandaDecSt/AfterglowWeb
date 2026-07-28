export interface ShaderSource {
  label: string;
  code: string;
  stage?: string;
}

type ReloadCallback = (source: ShaderSource) => boolean | void;

export class ShaderHotReload {
  private sources = new Map<string, ShaderSource>();
  private listeners = new Map<string, ReloadCallback[]>();
  private editor: HTMLTextAreaElement | null = null;
  private statusBar: HTMLDivElement | null = null;
  private currentLabel = "";

  register(label: string, code: string, stage?: string) {
    this.sources.set(label, { label, code, stage });
  }

  unregister(label: string) {
    this.sources.delete(label);
    this.listeners.delete(label);
  }

  clear() {
    this.sources.clear();
    this.listeners.clear();
  }

  get(label: string): string | undefined {
    return this.sources.get(label)?.code;
  }

  getAll(): string[] {
    return [...this.sources.keys()];
  }

  getStage(label: string): string | undefined {
    return this.sources.get(label)?.stage;
  }

  onReload(label: string, cb: ReloadCallback) {
    const list = this.listeners.get(label) ?? [];
    list.push(cb);
    this.listeners.set(label, list);
  }

  removeReloadCallbacks(label: string) {
    this.listeners.delete(label);
  }

  update(label: string, code: string): boolean {
    const source = this.sources.get(label);
    if (!source) return false;
    source.code = code;
    const cbs = this.listeners.get(label) ?? [];
    let ok = true;
    for (const cb of cbs) {
      const result = cb(source);
      if (result === false) ok = false;
    }
    this.setStatus(ok ? `Reloaded: ${label}` : `Error in: ${label}`, ok);
    return ok;
  }

  createEditor(container: HTMLElement) {
    this.editor = document.createElement("textarea");
    Object.assign(this.editor.style, {
      position: "fixed",
      bottom: "0",
      left: "0",
      width: "100%",
      height: "220px",
      background: "#0d1117",
      color: "#c9d1d9",
      fontFamily: "'Cascadia Code', 'Fira Code', monospace",
      fontSize: "13px",
      lineHeight: "1.5",
      padding: "12px",
      border: "none",
      borderTop: "1px solid #30363d",
      resize: "vertical",
      outline: "none",
      zIndex: "10000",
      display: "none",
      tabSize: "2",
    });
    container.appendChild(this.editor);

    this.statusBar = document.createElement("div");
    Object.assign(this.statusBar.style, {
      position: "fixed",
      bottom: "220px",
      left: "0",
      width: "100%",
      height: "24px",
      background: "#161b22",
      color: "#8b949e",
      fontFamily: "monospace",
      fontSize: "11px",
      lineHeight: "24px",
      padding: "0 12px",
      borderTop: "1px solid #30363d",
      zIndex: "10001",
      display: "none",
      pointerEvents: "none",
    });
    container.appendChild(this.statusBar);

    this.editor.addEventListener("keydown", (e) => {
      if (e.ctrlKey && e.key === "Enter") {
        e.preventDefault();
        this.applyCurrent();
      }
      if (e.key === "Escape") {
        this.closeEditor();
      }
      if (e.key === "Tab") {
        e.preventDefault();
        const start = this.editor!.selectionStart;
        const end = this.editor!.selectionEnd;
        this.editor!.value =
          this.editor!.value.substring(0, start) + "  " + this.editor!.value.substring(end);
        this.editor!.selectionStart = this.editor!.selectionEnd = start + 2;
      }
    });
  }

  openEditor(label: string) {
    if (!this.editor) return;
    const source = this.sources.get(label);
    if (!source) return;
    this.currentLabel = label;
    this.editor.value = source.code;
    this.editor.style.display = "block";
    if (this.statusBar) {
      this.statusBar.style.display = "block";
      this.setStatus(`Editing: ${label}  [Ctrl+Enter] Apply  [Esc] Close`, true);
    }
    this.editor.focus();
  }

  closeEditor() {
    if (this.editor) this.editor.style.display = "none";
    if (this.statusBar) this.statusBar.style.display = "none";
    this.currentLabel = "";
  }

  get isOpen() {
    return this.editor?.style.display !== "none";
  }

  private setStatus(msg: string, ok: boolean) {
    if (!this.statusBar) return;
    this.statusBar.textContent = msg;
    this.statusBar.style.color = ok ? "#3fb950" : "#f85149";
    if (!ok) {
      setTimeout(() => {
        if (this.statusBar) this.statusBar.style.color = "#8b949e";
      }, 3000);
    }
  }

  private applyCurrent() {
    if (!this.editor || !this.currentLabel) return;
    this.update(this.currentLabel, this.editor.value);
  }
}
