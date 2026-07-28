export class ShaderModuleSystem {
  private modules = new Map<string, string>();
  private resolvedCache = new Map<string, string>();

  registerModule(name: string, code: string) {
    this.modules.set(name, code);
    this.resolvedCache.delete(name);
  }

  resolve(code: string, visited = new Set<string>()): string {
    const includeRegex = /#include\s+"([^"]+)"/g;
    let result = code;
    let match: RegExpExecArray | null;

    while ((match = includeRegex.exec(code)) !== null) {
      const moduleName = match[1];
      if (visited.has(moduleName)) {
        console.warn(`[ShaderModule] Circular include detected: ${moduleName}`);
        continue;
      }
      visited.add(moduleName);

      const moduleCode = this.modules.get(moduleName);
      if (!moduleCode) {
        console.warn(`[ShaderModule] Module not found: ${moduleName}`);
        continue;
      }

      const resolved = this.resolve(moduleCode, visited);
      result = result.replace(match[0], resolved);
      includeRegex.lastIndex = 0;
    }

    return result;
  }

  resolveAndCompile(device: GPUDevice, label: string, code: string): GPUShaderModule {
    const resolved = this.resolve(code);
    return device.createShaderModule({ label, code: resolved });
  }

  hotReload(label: string, code: string) {
    this.registerModule(label, code);
    this.resolvedCache.clear();
  }
}
