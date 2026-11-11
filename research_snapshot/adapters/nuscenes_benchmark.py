"""Native SplatAD nuScenes parser with an immutable, shared sensor-block split."""
import hashlib,json
from dataclasses import dataclass,field
from pathlib import Path
from typing import Type
import torch
from nerfstudio.data.dataparsers.nuscenes_dataparser import NuScenes,NuScenesDataParserConfig

@dataclass
class BenchmarkNuScenesConfig(NuScenesDataParserConfig):
    _target:Type=field(default_factory=lambda:BenchmarkNuScenes)
    split_manifest:Path=Path('.')
    split_sha256:str=''

class BenchmarkNuScenes(NuScenes):
    def __init__(self,config):
        super().__init__(config)
        raw=config.split_manifest.read_bytes()
        assert hashlib.sha256(raw).hexdigest()==config.split_sha256,'Split manifest changed'
        self.benchmark_manifest=json.loads(raw)
        assert self.benchmark_manifest['scene']==config.sequence
        self.benchmark_files={r['filename']:r for r in self.benchmark_manifest['files']}
    def _tag(self,sensors,files):
        keys=[str(p.relative_to(self.config.data)) for p in files]
        assert len(keys)==len(set(keys))
        sensors.metadata['benchmark_train']=torch.tensor([self.benchmark_files[k]['split']=='train' for k in keys],dtype=torch.bool).reshape(-1,1)
        return sensors,files
    def _get_cameras(self):
        return self._tag(*super()._get_cameras())
    def _get_lidars(self):
        return self._tag(*super()._get_lidars())
    def _get_train_eval_indices(self,sensors):
        mask=sensors.metadata['benchmark_train'].reshape(-1)
        train=mask.nonzero(as_tuple=True)[0]; val=(~mask).nonzero(as_tuple=True)[0]
        assert len(train)>0 and len(val)>0
        return train,val
    def _add_trajectories_velocities(self,trajectories):
        # The native parser already removes <=1-sample trajectories immediately
        # after this call. Apply that same criterion before velocity interpolation,
        # which otherwise indexes [0] on actors absent from the selected split.
        valid=[t for t in trajectories if len(t['timestamps'])>1]
        return super()._add_trajectories_velocities(valid) if valid else []
    def _generate_dataparser_outputs(self,split='train'):
        out=super()._generate_dataparser_outputs(split)
        expected='train' if split=='train' else 'eval'
        assert all(self.benchmark_files[str(p.relative_to(self.config.data))]['split']==expected for p in out.image_filenames)
        lidar_mask=out.metadata['lidars'].metadata['benchmark_train']
        assert bool((lidar_mask==(expected=='train')).all())
        # Only the selected split is returned to initialization, point painting and losses.
        out.metadata['benchmark_split_sha256']=self.config.split_sha256
        return out
