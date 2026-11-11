"""SplatAD nuScenes adapter with physically absent real heldout observations."""

from dataclasses import dataclass, field
from typing import Type

import torch
from nerfstudio.data.dataparsers.nuscenes_dataparser import NuScenes

from nuscenes_benchmark import BenchmarkNuScenes, BenchmarkNuScenesConfig


@dataclass
class TrainOnlyNuScenesConfig(BenchmarkNuScenesConfig):
    _target: Type = field(default_factory=lambda: TrainOnlyNuScenes)


class TrainOnlyNuScenes(BenchmarkNuScenes):
    def _get_train_eval_indices(self, sensors):
        roles = sensors.metadata['benchmark_train'].reshape(-1)
        assert bool(roles.all()) and len(roles) > 0
        train = torch.arange(len(roles), dtype=torch.long)
        # Nerfstudio requires an eval dataset. Its only entry is a duplicate
        # training observation, so no real heldout RGB or LiDAR is accessible.
        return train, train[:1]

    def _generate_dataparser_outputs(self, split='train'):
        out = NuScenes._generate_dataparser_outputs(self, split)
        assert all(self.benchmark_files[str(path.relative_to(self.config.data))]['split'] == 'train'
                   for path in out.image_filenames)
        assert bool(out.metadata['lidars'].metadata['benchmark_train'].all())
        out.metadata['benchmark_split_sha256'] = self.config.split_sha256
        out.metadata['benchmark_eval_is_train_shadow'] = True
        return out
