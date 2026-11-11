"""Reload and render a selected native SplatAD actor with its own RGB decoder."""

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch
torch.empty(1,device="cuda")  # Claim the assigned idle GPU before loading the dataset.


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'adapters'))
sys.path.insert(0, str(ROOT / 'scripts'))
from train_splatad_reviewed_scene import MonitoredTrainer  # Legacy YAML target alias.
from train_splatad_human_train_only_scene import MonitoredTrainer as TrainOnlyMonitoredTrainer  # Strict YAML target alias.
from nerfstudio.utils.eval_utils import eval_setup
import common_camera_contract
from common_camera_contract import load_contract, projection_audit


def save_image(array, path):
    Image.fromarray((array.clamp(0, 1).cpu().numpy() * 255).round().astype(np.uint8)).save(path)


def main():
    parser = argparse.ArgumentParser()
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--sample-id')
    group.add_argument('--human-id')
    parser.add_argument('--variant', choices=('legacy', 'train_only'), default='legacy')
    args = parser.parse_args()
    inventory=json.loads((ROOT/'configs/common_camera_vehicle432_v2/manifest.json').read_text())
    entry=next(x for x in inventory['objects']if x['sample_id']==args.sample_id)
    assert entry['methods']['splatad']['status']=='native_actor_candidate_pending_source_render'
    selected=entry['object']
    common_camera_contract.CONTRACT=Path(entry['contract'])
    assert hashlib.sha256(common_camera_contract.CONTRACT.read_bytes()).hexdigest()==entry['contract_sha256']
    scene=selected['scene'];source_image=Path(selected['anchor']['source_rgb']).resolve()
    assert hashlib.sha256(source_image.read_bytes()).hexdigest()==selected['anchor']['source_rgb_sha256']
    item={'splatad_actor_uuid':selected['native_instance_token'],'raw_timestamp_us':selected['camera']['timestamp_us']}
    job=entry['methods']['splatad']['training_job'];assert job['status']=='formal_validated'
    validation=json.loads(Path(job['validation']).read_text());assert validation['status']=='passed'
    actor=next(x for x in validation['actors'] if x['uuid']==item['splatad_actor_uuid'])
    actor_bundle=actor['bundle'];actor_gaussians=actor['gaussians']
    config_path=Path(validation['checkpoint']).parent.parent/'config.yml'
    out=ROOT/'exports/common_camera_vehicle432_v1/splatad'/args.sample_id
    assert not out.exists();reconstruction_only=True
    config, pipeline, checkpoint, step = eval_setup(config_path, test_mode='val')
    assert step == 30000
    model = pipeline.model
    manager = pipeline.datamanager
    filenames = [Path(p).resolve() for p in manager.train_dataset.image_filenames]
    matches = [i for i, path in enumerate(filenames) if path == source_image]
    assert len(matches) == 1, (source_image, len(matches))
    camera_index = matches[0]
    camera = manager.train_dataset.cameras[camera_index:camera_index + 1].to(model.device)
    contract=load_contract(args.sample_id)
    parsed=manager.train_dataparser_outputs
    assert float(parsed.dataparser_scale)==1.0
    model_from_world=np.eye(4)
    model_from_world[:3]=parsed.dataparser_transform.detach().cpu().numpy()[:3]
    cv_to_gl=np.diag([1.,-1.,-1.,1.])
    model_from_camera_cv=model_from_world @ np.asarray(contract['T_world_from_camera_opencv'])
    model_from_camera_gl=model_from_camera_cv @ cv_to_gl
    native_pose_error=float(np.max(np.abs(camera.camera_to_worlds[0].cpu().numpy()-model_from_camera_gl[:3])))
    assert native_pose_error<1e-3, native_pose_error
    native_render_wh=[int(camera.width.item()),int(camera.height.item())]
    camera.camera_to_worlds[0]=torch.tensor(model_from_camera_gl[:3],device=model.device,dtype=camera.camera_to_worlds.dtype)
    width,height=contract['render_wh']
    camera.width.fill_(width);camera.height.fill_(height)
    raw_k=np.asarray(contract['K_raw'])
    for name,value in [('fx',raw_k[0,0]),('fy',raw_k[1,1]),('cx',raw_k[0,2]),('cy',raw_k[1,2])]:
        getattr(camera,name).fill_(float(value))
    assert abs(float(camera.times.item())-(contract['timestamp_us']/1e6-float(parsed.time_offset)))<1e-4
    model.config.use_camopt_in_eval=False
    model.config.compensate_rs_camera=False
    for key in ('lin_vel_x','lin_vel_y','lin_vel_z','ang_vel_x','ang_vel_y','ang_vel_z','rs_time'):
        model.rs_editing[key]=0.
    camera_audit=projection_audit(contract,model_from_world,
        np.vstack([camera.camera_to_worlds[0].cpu().numpy(),[0,0,0,1]]) @ cv_to_gl,
        camera.get_intrinsics_matrices()[0].cpu().numpy())
    camera_audit.update(native_training_render_wh=native_render_wh,native_pose_max_error=native_pose_error,
                        timestamp_us=contract['timestamp_us'],native_time_offset_seconds=float(parsed.time_offset))
    actor_index = next(i for i, actor in enumerate(manager.train_dataparser_outputs.metadata['trajectories'])
                       if actor['uuid'] == item['splatad_actor_uuid'])
    mask = model.id.detach().reshape(-1) == actor_index
    assert int(mask.sum()) == actor_gaussians
    bundle = torch.load(actor_bundle, map_location='cpu', weights_only=False)
    assert bundle['uuid'] == item['splatad_actor_uuid'] and bundle['actor_index'] == actor_index
    for name, parameter in model.gauss_params.items():
        assert torch.equal(parameter.detach()[mask].cpu(), bundle['gauss_params'][name])
    out.mkdir(parents=True, exist_ok=True)
    pipeline.eval()
    with torch.no_grad():
        scene_result = model.get_outputs_for_camera(camera=copy.deepcopy(camera))
        original_parameters = model.gauss_params
        subset = torch.nn.ParameterDict({name: torch.nn.Parameter(parameter.detach()[mask].clone(), requires_grad=False)
                                         for name, parameter in original_parameters.items()})
        model.gauss_params = subset
        original_background = model.config.background_color
        try:
            model.config.background_color = 'white'
            actor_result = model.get_outputs_for_camera(camera=copy.deepcopy(camera))
        finally:
            model.gauss_params = original_parameters
            model.config.background_color = original_background
    rgb = actor_result['rgb'].detach()
    alpha = actor_result['accumulation'].detach()
    assert torch.isfinite(rgb).all() and torch.isfinite(alpha).all()
    assert float(alpha.max()) > 0 and float(rgb.std()) > 1e-5
    save_image(scene_result['rgb'], out / 'native_full_scene_source.png')
    save_image(rgb, out / 'native_actor_source.png')
    save_image(alpha.squeeze(-1), out / 'native_actor_alpha.png')
    result = {'status': 'passed', 'sample_id': args.sample_id, 'scene': scene,
              'source_image': str(source_image), 'source_camera_index': camera_index,
              'source_relation': 'exact frame traced from frozen retained input provenance; no pixel metric asserted',
              'actor_uuid': item['splatad_actor_uuid'], 'gaussians': int(mask.sum()),
              'checkpoint': str(checkpoint), 'step': step, 'bundle': actor_bundle,
              'variant': 'reconstruction_only' if reconstruction_only else args.variant,
              'real_heldout_eligible': not reconstruction_only,
              'frozen_raw_timestamp_us':item.get('raw_timestamp_us'),
              'renderer': 'native SplatAD rasterizer and native RGB decoder',
              'background': 'white',
              'common_camera_audit':camera_audit,
              'render_wh': [int(rgb.shape[1]), int(rgb.shape[0])],
              'source_camera_K': camera.get_intrinsics_matrices()[0].detach().cpu().tolist(),
              'source_camera_to_world': camera.camera_to_worlds[0].detach().cpu().tolist(),
              'source_camera_time': camera.times.detach().cpu().reshape(-1).tolist(),
              'full_scene_preview': str(out / 'native_full_scene_source.png'),
              'actor_preview': str(out / 'native_actor_source.png'),
              'actor_alpha': str(out / 'native_actor_alpha.png'),
              'actor_alpha_pixels_gt_0_5': int((alpha > 0.5).sum()),
              'rgb_std': float(rgb.std())}
    (out / 'validation.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
