# Hi3D v1.0

**English** · <a href="README.zh-CN.md"><kbd>简体中文</kbd></a>

**High-resolution 3D geometry refinement built upon Sparc3D.**

[International Website](http://hi3d.ai/) · [China Website](https://www.hi3d.cn/)

Planned open-source release: October 23, 2026.

This preview includes project documentation, the system overview, and selected visual comparisons. Source code will follow with the release.

## Overview

Hi3D v1.0 is the official Hi3D project for high-resolution 3D geometry refinement. Built upon the Sparc3D geometry pipeline, it uses a reference image together with a coarse mesh to refine object geometry through successive sparse reconstruction levels.

The project report describes training on approximately **two million curated 3D assets** and refinement up to an effective sparse extraction-grid resolution of **1536³**. Hi3D v1.0 retains the Sparc3D representation and refinement architecture, with an emphasis on scaling training data and high-resolution geometry.

## System overview

[![Hi3D v1.0 system overview: reference image and coarse mesh, sparse refinement, and three mesh extraction levels](assets/framework.png)](assets/framework.png)

Image features condition sparse flow on coordinates derived from the coarse geometry. A self-pruning sparse VAE decoder expands the features around predicted surfaces, followed by mesh extraction. Between levels, the extracted mesh is revoxelized to provide the next sparse support.

| Level | Nominal structural grid | Sparse extraction grid |
| --- | --- | --- |
| 1 | 64³ | 512³ |
| 2 | 128³ | 1024³ |
| 3 | 192³ | 1536³ |

These labels describe nominal sparse coordinate and surface-extraction resolutions. They do not represent a dense 1536³ latent tensor or a fixed output triangle count.

## Visual comparisons

### Local geometry

[![Local geometry comparison across nine methods, with Hi3D v1.0 highlighted in teal](assets/local-detail-comparison.jpg)](assets/local-detail-comparison.jpg)

Red boxes mark the reference regions enlarged in each row. Shaded geometry and colored surface-normal renderings show local structures, including layered relief, architectural moldings, curved boundaries, scales, teeth, and concentric grooves. **Hi3D v1.0 is highlighted in teal in the rightmost column.** Click the figure to inspect the full-resolution image.

### Multi-view comparisons

[![Multi-view geometry comparisons for a vehicle, armored character, armchair, and handbag](assets/global-comparison-1.jpg)](assets/global-comparison-1.jpg)

Front, back, and bottom views are shown for four objects. Results combine shaded geometry and colored surface-normal renderings; the color does not represent generated textures. Hi3D v1.0 is highlighted in teal.

<details>
<summary>More multi-view examples</summary>

[![Additional multi-view examples for a humanoid bust, mechanical robots, and a stylized character](assets/global-comparison-2.jpg)](assets/global-comparison-2.jpg)

Additional examples follow the same column order and rendering conventions as the preceding figure.

</details>

## Scope and limitations

- **Geometry output.** Hi3D v1.0 focuses on geometry; it does not generate textures, materials, or vertex colors.
- **Coarse-mesh dependency.** An upstream coarse mesh supplies global structure. Severe structural errors in that initialization may persist after refinement.
- **Occluded regions.** Geometry hidden from the input image is a plausible completion rather than an exact reconstruction of unseen surfaces.
- **Application-specific validation.** Visual detail and nominal resolution do not by themselves guarantee mesh suitability for a particular downstream workflow.

## Acknowledgments

Hi3D v1.0 builds upon [Sparc3D: Sparse Representation and Construction for High-Resolution 3D Shapes Modeling](https://arxiv.org/abs/2505.14521). We acknowledge the original Sparc3D work and the research it builds upon.

