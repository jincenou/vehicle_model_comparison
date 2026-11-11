# Third-party dependencies and attribution

This repository contains comparison orchestration and evaluation utilities. Download upstream code, datasets and model weights separately and follow their respective licenses, model cards and access requirements. An alternate hosting service does not waive model access or license obligations.

- [nuScenes](https://www.nuscenes.org/): source dataset; not redistributed.
- [Asset Harvester](https://github.com/NVIDIA/asset-harvester): native image-to-Gaussian model.
- [TRELLIS](https://github.com/microsoft/TRELLIS): original model; this experiment also depends on a local legacy Gaussian loader.
- [TRELLIS.2](https://github.com/microsoft/TRELLIS.2): native image-to-3D generation, rendering and export.
- [Hunyuan3D 2.1](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1): shape and texture pipelines.
- ReconViaGen v0.2, SplatAD and OmniRe: native model implementations are external dependencies; resolve their exact original provenance before reproducing a run.
- [Street Gaussians](https://github.com/zju3dv/street_gaussians): native scene reconstruction.
- NumPy, SciPy, Pillow and scikit-image: CPU numerical/image utilities.
- PyTorch, native Gaussian rasterizers, CUDA extensions, Blender, feature encoders and auxiliary networks: optional model/evaluation runtime dependencies.

`configs/trellis2_auxiliary_artifacts.json` retains pinned auxiliary repository revisions and file hashes, not payloads or access credentials. Main and auxiliary weight downloads must be checked against the recorded inventory before execution. The snapshot preparation code is evidence of the experiment's setup, not a grant to redistribute those dependencies.

Please cite the original model and dataset publications when reporting results. This repository does not supply fabricated publication metadata for any project.
