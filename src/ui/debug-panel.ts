import GUI from "lil-gui";
import { ShaderHotReload } from "../shader/hotreload";
import { Camera } from "../scene/camera";

export class DebugPanel {
  private gui: GUI;
  private shaderReload: ShaderHotReload;
  private shaderFolder: GUI;
  private shaderState: { shader: string; open: () => void; close: () => void };
  private shaderController: ReturnType<GUI["add"]> | null = null;

  constructor(shaderReload: ShaderHotReload, camera: Camera) {
    this.shaderReload = shaderReload;
    this.gui = new GUI({ title: "AfterglowRender" });

    this.shaderFolder = this.gui.addFolder("Shader Editor");
    this.shaderState = {
      shader: "",
      open: () => this.shaderReload.openEditor(this.shaderState.shader),
      close: () => this.shaderReload.closeEditor(),
    };
    this.refreshShaderList();
    this.shaderFolder.add(this.shaderState, "open").name("Open Editor");
    this.shaderFolder.add(this.shaderState, "close").name("Close Editor");

    const camFolder = this.gui.addFolder("Camera");
    camFolder.add(camera, "fov", 20, 120, 1).name("FOV");
    camFolder.add(camera, "near", 0.01, 1, 0.01).name("Near");
    camFolder.add(camera, "far", 10, 500, 1).name("Far");
  }

  refreshShaderList() {
    if (this.shaderController) {
      this.shaderController.destroy();
      this.shaderController = null;
    }
    const shaders = this.shaderReload.getAll();
    if (shaders.length > 0) {
      if (!shaders.includes(this.shaderState.shader)) {
        this.shaderState.shader = shaders[0];
      }
      this.shaderController = this.shaderFolder
        .add(this.shaderState, "shader", shaders)
        .name("Stage");
    }
  }

  addFolder(title: string): GUI {
    return this.gui.addFolder(title);
  }

  get root(): GUI {
    return this.gui;
  }

  show(visible: boolean) {
    this.gui.show(visible);
  }

  get hidden() {
    return this.gui._hidden;
  }

  destroy() {
    this.gui.destroy();
  }
}
