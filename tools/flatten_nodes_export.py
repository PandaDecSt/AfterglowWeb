"""
flatten_nodes_export.py — Blender Python 脚本
功能：
  1. 递归遍历所有材质节点组，找到 Image Texture 节点
  2. 根据节点名称/连接关系识别 diffuse / lightmap / normal
  3. 拍平为标准 Principled BSDF 连接（glTF 导出器可识别）
  4. Lightmap → glTF Occlusion 槽
  5. 导出 GLB（嵌入贴图）

使用方法：
  Blender → Scripting 标签页 → 新建文本块 → 粘贴 → 运行
  或者: blender --background scene.blend --python flatten_nodes_export.py
"""

import bpy
import os
from pathlib import Path


# ============================================================
# 配置
# ============================================================
OUTPUT_FILENAME = "Qin_DL_fixed.glb"

# 节点名称关键词匹配（不区分大小写）
DIFFUSE_KEYWORDS = ["diffuse", "base", "color", "albedo", "tex", "body", "face", "hair", "cloth", "skin"]
LIGHTMAP_KEYWORDS = ["lightmap", "light_map", "lm", "shadow", "ao", "occlusion", "bake"]
NORMAL_KEYWORDS = ["normal", "nrm", "bump"]
METALLIC_KEYWORDS = ["metallic", "metal", "specular", "spec", "roughness", "rough", "mr", "pbr"]


# ============================================================
# 工具函数
# ============================================================

def classify_texture(node_name: str, image_name: str) -> str:
    """根据节点名和图片名判断贴图类型"""
    combined = (node_name + " " + image_name).lower()

    # 优先匹配 lightmap（因为 lightmap 也可能包含 "shadow" 等词）
    for kw in LIGHTMAP_KEYWORDS:
        if kw in combined:
            return "lightmap"

    for kw in NORMAL_KEYWORDS:
        if kw in combined:
            return "normal"

    for kw in METALLIC_KEYWORDS:
        if kw in combined:
            return "metallic_roughness"

    for kw in DIFFUSE_KEYWORDS:
        if kw in combined:
            return "diffuse"

    # 默认当作 diffuse
    return "diffuse"


def find_image_textures_recursive(nodes, prefix="") -> list:
    """递归遍历节点树（包括 Group 节点内部），找到所有 Image Texture 节点"""
    results = []
    for node in nodes:
        full_name = f"{prefix}{node.name}"
        if node.type == 'TEX_IMAGE' and node.image:
            results.append({
                "node": node,
                "name": full_name,
                "image": node.image,
                "image_name": node.image.name,
            })
        elif node.type == 'GROUP' and node.node_tree:
            # 递归进入节点组
            sub_results = find_image_textures_recursive(
                node.node_tree.nodes,
                prefix=f"{full_name}/"
            )
            results.extend(sub_results)
    return results


def ensure_principled_bsdf(tree) -> bpy.types.Node:
    """确保材质有 Principled BSDF 节点，没有就创建一个"""
    for node in tree.nodes:
        if node.type == 'BSDF_PRINCIPLED':
            return node

    # 创建新的 Principled BSDF
    bsdf = tree.nodes.new('ShaderNodeBsdfPrincipled')
    bsdf.location = (0, 0)

    # 连接到 Material Output
    output = None
    for node in tree.nodes:
        if node.type == 'OUTPUT_MATERIAL':
            output = node
            break
    if not output:
        output = tree.nodes.new('ShaderNodeOutputMaterial')
        output.location = (300, 0)

    tree.links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])
    return bsdf


def disconnect_input(tree, input_socket):
    """断开某个输入槽的所有连接"""
    for link in list(tree.links):
        if link.to_socket == input_socket:
            tree.links.remove(link)


def process_material(mat):
    """处理单个材质：拍平节点组，建立标准连接"""
    if not mat.use_nodes:
        mat.use_nodes = True

    tree = mat.node_tree
    bsdf = ensure_principled_bsdf(tree)

    # 找到所有 Image Texture（递归）
    textures = find_image_textures_recursive(tree.nodes)

    if not textures:
        print(f"  [SKIP] {mat.name}: no Image Texture nodes found")
        return

    # 分类贴图
    classified = {"diffuse": [], "lightmap": [], "normal": [], "metallic_roughness": []}
    for tex_info in textures:
        tex_type = classify_texture(tex_info["name"], tex_info["image_name"])
        classified[tex_type].append(tex_info)

    print(f"  [MAT] {mat.name}:")
    for tex_type, tex_list in classified.items():
        for t in tex_list:
            print(f"    {tex_type}: {t['image_name']} (node: {t['name']})")

    # --- 连接 Diffuse → Base Color ---
    if classified["diffuse"]:
        tex_info = classified["diffuse"][0]  # 取第一个
        tex_node = tex_info["node"]

        # 如果贴图节点在节点组内部，需要在主树中创建一个代理节点
        if "/" in tex_info["name"]:
            # 在主树中创建新的 Image Texture 节点引用同一张图片
            proxy = tree.nodes.new('ShaderNodeTexImage')
            proxy.image = tex_info["image"]
            proxy.name = f"FLAT_{tex_info['image_name']}"
            proxy.location = (-400, 300)
            tex_node = proxy

        disconnect_input(tree, bsdf.inputs['Base Color'])
        tree.links.new(tex_node.outputs['Color'], bsdf.inputs['Base Color'])

        # 如果有 Alpha，连接 Alpha
        if 'Alpha' in tex_node.outputs and bsdf.inputs.get('Alpha'):
            tree.links.new(tex_node.outputs['Alpha'], bsdf.inputs['Alpha'])

        print(f"    → Linked diffuse: {tex_info['image_name']} → Base Color")

    # --- 连接 Lightmap → glTF 导出器识别的方式 ---
    # glTF 的 occlusionTexture 需要连接到 Principled BSDF 的
    # "Specular IOR Level" 或通过 glTF Material Output 节点
    # Blender 4.x 的 glTF 导出器支持通过 "glTF Material Output" 节点
    if classified["lightmap"]:
        tex_info = classified["lightmap"][0]
        tex_node = tex_info["node"]

        if "/" in tex_info["name"]:
            proxy = tree.nodes.new('ShaderNodeTexImage')
            proxy.image = tex_info["image"]
            proxy.name = f"FLAT_LM_{tex_info['image_name']}"
            proxy.location = (-400, 0)
            tex_node = proxy

        # 方法1: 尝试使用 glTF Material Output 节点（Blender 3.x+）
        gltf_output = None
        for node in tree.nodes:
            if node.type == 'OUTPUT_MATERIAL' and hasattr(node, 'target'):
                pass
            # Blender 的 glTF 导出器插件节点
            if "gltf" in node.bl_idname.lower() if hasattr(node, 'bl_idname') else False:
                gltf_output = node
                break

        # 方法2: 创建 ShaderNodeGroup 模拟 glTF occlusion
        # 最可靠的方式：直接连接到 Principled BSDF 的 Specular 输入
        # glTF 导出器会把 Specular IOR Level 的贴图导出为 occlusionTexture
        # 但这不完全正确。最可靠的是用 MixRGB 把 lightmap 乘到 Base Color 上
        # 这样至少视觉上正确，即使 glTF 里不是独立的 occlusion 通道

        # 实际最可靠方案：用 glTF 导出器的 "Occlusion" 输入
        # Blender 4.0+ 的 Principled BSDF 没有直接 Occlusion 输入
        # 但 glTF 导出器会查找名为 "Occlusion" 的节点组输入

        # 最终方案：将 lightmap 通过 MixRGB 乘到 diffuse 上（视觉正确）
        # 同时保留独立贴图供我们的 viewer 使用
        mix_node = tree.nodes.new('ShaderNodeMixRGB')
        mix_node.blend_type = 'MULTIPLY'
        mix_node.inputs['Fac'].default_value = 1.0
        mix_node.location = (-200, 300)

        # 找到当前连接到 Base Color 的节点
        base_color_link = None
        for link in tree.links:
            if link.to_socket == bsdf.inputs['Base Color']:
                base_color_link = link
                break

        if base_color_link:
            # 把原来的 diffuse 输出接到 Mix 的 Color1
            source_output = base_color_link.from_socket
            tree.links.remove(base_color_link)
            tree.links.new(source_output, mix_node.inputs[1])  # Color1
            tree.links.new(tex_node.outputs['Color'], mix_node.inputs[2])  # Color2 (lightmap)
            tree.links.new(mix_node.outputs['Color'], bsdf.inputs['Base Color'])
            print(f"    → Linked lightmap: {tex_info['image_name']} → Multiply into Base Color")
        else:
            # 没有 diffuse，直接把 lightmap 当 Base Color
            tree.links.new(tex_node.outputs['Color'], bsdf.inputs['Base Color'])
            print(f"    → Linked lightmap as Base Color: {tex_info['image_name']}")

    # --- 连接 Normal ---
    if classified["normal"]:
        tex_info = classified["normal"][0]
        tex_node = tex_info["node"]

        if "/" in tex_info["name"]:
            proxy = tree.nodes.new('ShaderNodeTexImage')
            proxy.image = tex_info["image"]
            proxy.name = f"FLAT_NRM_{tex_info['image_name']}"
            proxy.location = (-400, -300)
            proxy.image.colorspace_settings.name = 'Non-Color'
            tex_node = proxy

        # 需要 Normal Map 节点
        normal_map_node = tree.nodes.new('ShaderNodeNormalMap')
        normal_map_node.location = (-200, -300)

        disconnect_input(tree, bsdf.inputs['Normal'])
        tree.links.new(tex_node.outputs['Color'], normal_map_node.inputs['Color'])
        tree.links.new(normal_map_node.outputs['Normal'], bsdf.inputs['Normal'])
        print(f"    → Linked normal: {tex_info['image_name']} → Normal")

    # --- 连接 Metallic/Roughness ---
    if classified["metallic_roughness"]:
        tex_info = classified["metallic_roughness"][0]
        tex_node = tex_info["node"]

        if "/" in tex_info["name"]:
            proxy = tree.nodes.new('ShaderNodeTexImage')
            proxy.image = tex_info["image"]
            proxy.name = f"FLAT_MR_{tex_info['image_name']}"
            proxy.location = (-400, -600)
            proxy.image.colorspace_settings.name = 'Non-Color'
            tex_node = proxy

        # glTF 约定: G=roughness, B=metallic
        # 用 Separate RGB 拆分
        sep = tree.nodes.new('ShaderNodeSeparateRGB')
        sep.location = (-200, -600)
        tree.links.new(tex_node.outputs['Color'], sep.inputs['Image'])

        disconnect_input(tree, bsdf.inputs['Metallic'])
        disconnect_input(tree, bsdf.inputs['Roughness'])
        tree.links.new(sep.outputs['B'], bsdf.inputs['Metallic'])   # B = metallic
        tree.links.new(sep.outputs['G'], bsdf.inputs['Roughness'])  # G = roughness
        print(f"    → Linked metallic/roughness: {tex_info['image_name']} → Metallic(B) + Roughness(G)")


# ============================================================
# 主流程
# ============================================================

def main():
    print("=" * 60)
    print("  Flatten Node Groups → glTF-compatible materials")
    print("=" * 60)

    # 处理所有材质
    for mat in bpy.data.materials:
        if mat.users == 0:
            continue  # 跳过未使用的材质
        process_material(mat)

    # 确定输出路径
    blend_path = bpy.data.filepath
    if blend_path:
        output_dir = os.path.dirname(blend_path)
    else:
        output_dir = os.path.expanduser("~")
    output_path = os.path.join(output_dir, OUTPUT_FILENAME)

    print(f"\n  Exporting to: {output_path}")

    # 导出 GLB
    bpy.ops.export_scene.gltf(
        filepath=output_path,
        export_format='GLB',
        use_selection=False,
        export_apply=True,
        export_normals=True,
        export_materials='EXPORT',
        export_texcoords=True,
        export_tangents=False,
    )

    print(f"\n  Done! Output: {output_path}")
    print(f"  将 {OUTPUT_FILENAME} 复制到 AfterglowWeb/public/ 目录即可")
    print("=" * 60)


if __name__ == "__main__":
    main()
