# Hi3D v1.0

**High-resolution 3D geometry refinement built upon Sparc3D.**  
**基于 Sparc3D 的高分辨率 3D 几何细化模型。**

[International Website](http://hi3d.ai/) · [国内官网](https://www.hi3d.cn/)

> **Open-source release: October 23, 2026.** This preview contains project documentation, the system overview, and selected visual comparisons. Source code is not included yet.
>
> **正式开源日期：2026 年 10 月 23 日。** 本次预告先公开项目说明、系统框架与部分效果对比，暂不发布实际代码。

## Overview / 项目介绍

Hi3D v1.0 is the official Hi3D project for high-resolution 3D geometry refinement. Built upon the Sparc3D geometry pipeline, it uses a reference image together with a coarse mesh to refine object geometry through successive sparse reconstruction levels.

Hi3D v1.0 是 Hi3D 官方推出的高分辨率 3D 几何细化项目。模型基于 Sparc3D 几何流程，以参考图和粗网格共同作为输入，通过逐级稀疏重建细化物体几何。

The project report describes training on approximately **two million curated 3D assets** and refinement up to an effective sparse extraction-grid resolution of **1536³**. Hi3D v1.0 retains the Sparc3D representation and refinement architecture, with an emphasis on scaling training data and high-resolution geometry.

项目报告采用约 **200 万个精选 3D 资产**的训练规模，并支持最高 **1536³** 有效稀疏提取网格分辨率。Hi3D v1.0 沿用 Sparc3D 的表示与细化架构，重点在于训练数据规模与高分辨率几何。

## System overview / 系统框架

[![Hi3D v1.0 system overview: reference image and coarse mesh, sparse refinement, and three mesh extraction levels](assets/framework.png)](assets/framework.png)

Image features condition sparse flow on coordinates derived from the coarse geometry. A self-pruning sparse VAE decoder expands the features around predicted surfaces, followed by mesh extraction. Between levels, the extracted mesh is revoxelized to provide the next sparse support.

图像特征在粗几何所确定的稀疏坐标上为流模型提供条件；自剪枝稀疏 VAE 解码器在预测表面附近扩展特征并提取网格。相邻层级之间，将已提取的网格重新体素化，形成下一层的稀疏支撑。

| Level / 层级 | Nominal structural grid / 标称结构网格 | Sparse extraction grid / 稀疏提取网格 |
| --- | --- | --- |
| 1 | 64³ | 512³ |
| 2 | 128³ | 1024³ |
| 3 | 192³ | 1536³ |

These labels describe nominal sparse coordinate and surface-extraction resolutions. They do not represent a dense 1536³ latent tensor or a fixed output triangle count.

以上数值表示标称稀疏坐标与表面提取分辨率，并非稠密 1536³ 潜变量张量，也不表示固定的输出三角面数。

## Visual comparisons / 效果对比

### Local geometry / 局部几何

[![Local geometry comparison across nine methods, with Hi3D v1.0 highlighted in teal](assets/local-detail-comparison.jpg)](assets/local-detail-comparison.jpg)

Red boxes mark the reference regions enlarged in each row. Shaded geometry and colored surface-normal renderings show local structures, including layered relief, architectural moldings, curved boundaries, scales, teeth, and concentric grooves. **Hi3D v1.0 is highlighted in teal in the rightmost column.** Click the figure to inspect the full-resolution image.

红框标出各行放大的参考区域。灰模与彩色表面法线渲染共同展示层叠浮雕、建筑线脚、曲线边界、鳞片、牙齿与同心凹槽等局部结构。**最右列以青绿色标出 Hi3D v1.0。** 点击图片可查看完整分辨率。

### Multi-view comparisons / 多视角对比

[![Multi-view geometry comparisons for a vehicle, armored character, armchair, and handbag](assets/global-comparison-1.jpg)](assets/global-comparison-1.jpg)

Front, back, and bottom views are shown for four objects. Results combine shaded geometry and colored surface-normal renderings; the color does not represent generated textures. Hi3D v1.0 is highlighted in teal.

四组物体分别展示正面、背面与底部视角。结果组合呈现灰模与彩色表面法线渲染，彩色部分不表示生成的纹理。Hi3D v1.0 以青绿色标出。

<details>
<summary>More multi-view examples / 更多多视角示例</summary>

[![Additional multi-view examples for a humanoid bust, mechanical robots, and a stylized character](assets/global-comparison-2.jpg)](assets/global-comparison-2.jpg)

Additional examples follow the same column order and rendering conventions as the preceding figure.

更多示例沿用上一张图的列顺序与渲染方式。

</details>

## Scope and limitations / 适用范围与局限

- **Geometry output.** Hi3D v1.0 focuses on geometry; it does not generate textures, materials, or vertex colors.  
  **几何输出。** Hi3D v1.0 聚焦几何，不生成纹理、材质或顶点颜色。
- **Coarse-mesh dependency.** An upstream coarse mesh supplies global structure. Severe structural errors in that initialization may persist after refinement.  
  **依赖粗网格。** 上游粗网格提供整体结构，其中严重的结构错误可能在细化后保留。
- **Occluded regions.** Geometry hidden from the input image is a plausible completion rather than an exact reconstruction of unseen surfaces.  
  **遮挡区域。** 输入图不可见的几何属于合理补全，不能视为真实隐藏表面的精确重建。
- **Application-specific validation.** Visual detail and nominal resolution do not by themselves guarantee mesh suitability for a particular downstream workflow.  
  **按应用验证。** 视觉细节与标称分辨率本身不能保证网格满足特定下游工作流的要求。

## Release plan / 发布计划

| Date / 日期 | Milestone / 进度 |
| --- | --- |
| October 16, 2026 / 2026 年 10 月 16 日 | Project preview: documentation, framework, and visual examples / 项目预告：说明、框架与效果展示 |
| October 23, 2026 / 2026 年 10 月 23 日 | Hi3D v1.0 open-source release / Hi3D v1.0 正式开源 |

Release contents, setup instructions, and usage terms will be documented here with the formal release. Watch this repository for updates.

正式发布时将在本仓库补充发布内容、安装说明与使用条款，欢迎关注后续更新。

## Acknowledgments / 致谢

Hi3D v1.0 builds upon [Sparc3D: Sparse Representation and Construction for High-Resolution 3D Shapes Modeling](https://arxiv.org/abs/2505.14521). We acknowledge the original Sparc3D work and the research it builds upon.

Hi3D v1.0 基于 [Sparc3D](https://arxiv.org/abs/2505.14521) 构建，感谢原始 Sparc3D 工作及其所依托的研究。
