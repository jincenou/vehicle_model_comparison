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


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'adapters'))
sys.path.insert(0, str(ROOT / 'scripts'))
from train_splatad_reviewed_scene import MonitoredTrainer  # Legacy YAML target alias.
from train_splatad_human_train_only_scene import MonitoredTrainer as TrainOnlyMonitoredTrainer  # Strict YAML target alias.
from nerfstudio.utils.eval_utils import eval_setup


def save_image(array, path):
    Image.fromarray((array.clamp(0, 1).cpu().numpy() * 255).round().astype(np.uint8)).save(path)


def main():
    parser = argparse.ArgumentParser()
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--sample-id')
    group.add_argument('--human-id')
    parser.add_argument('--variant', choices=('legacy', 'train_only'), default='legacy')
    args = parser.parse_args()
    if args.human_id:
        args.sample_id=args.human_id
        manifest=json.loads((ROOT/'inputs/benchmark_20261106/human_snapshot_manifest_v1.json').read_text())
        assert manifest['selection_sha256']==hashlib.sha256((ROOT/'inputs/benchmark_20261106/selection.json').read_bytes()).hexdigest()
        item=next(x for x in manifest['objects'] if x['id']==args.human_id)
        assert item['status']=='source_time_frozen'
        source_image=Path(item['source_image']).resolve()
        assert hashlib.sha256(source_image.read_bytes()).hexdigest()==item['source_sha256']
        assert args.variant=='train_only'
    else:
        mapping = json.loads((ROOT / 'reports/scene_actor_mapping_20261106.json').read_text())
        item = next(x for x in mapping['objects'] if x['sample_id'] == args.sample_id)
    assert item['dataset']=='nuscenes' and item.get('native_id')
    item['splatad_actor_uuid']=item['native_id'].split(':')[-1]
    if not args.human_id:
        selected = next(v for v in json.loads((ROOT / 'inputs/benchmark_20261106/selection.json').read_text())['vehicles']
                        if v['sample_id'] == args.sample_id)
        identity = json.loads(Path(item['identity_file']).read_text())['matches'][args.sample_id]
        assert (identity.get('native_id') == item['native_id'] or
                any(ref and ref.get('native_id') == item['native_id']
                    for ref in identity.get('reference_matches', {}).values()))
        if selected.get('review_id'):
            source_manifest = ROOT.parent / 'assetbench/reports/vehicle_expansion_v4/selection.json'
        else:
            source_manifest = ROOT / 'inputs/vehicle53_selection.json'
        provenance = next(row for row in json.loads(source_manifest.read_text())['samples']
                          if row['sample_id'] == args.sample_id)
        assert provenance['observation']['sha256'] == selected['retained_rgba_sha256']
        source_image = Path(provenance['observation']['source_image']).resolve()
    scene = item['scene']
    reconstruction_only = scene in ('scene-0066','scene-0067')
    if args.variant == 'train_only':
        scope = 'reconstruction_only' if reconstruction_only else ('human_reviewed95' if args.human_id else 'reviewed288_train_only')
        validation_path = ROOT / f'reports/splatad_{scope}_formal_{scene}_validation.json'
        validation = json.loads(validation_path.read_text())
        assert validation['status'] == 'passed' and validation['scene'] == scene
        if reconstruction_only:
            assert validation['real_heldout_eligible'] is False
        else:
            isolation_scope='human_reviewed95' if args.human_id else 'reviewed288'
            isolation = json.loads((ROOT / f'reports/splatad_train_only/{isolation_scope}/{scene}.json').read_text())
            assert isolation['status'] == 'verified_train_only' and isolation['heldout_rgb_lidar_present'] is False
            assert validation['split_sha256'] == isolation['manifest_sha256']
        config_path = Path(validation['checkpoint']).parent.parent/'config.yml'
        actor = next((a for a in validation['actors'] if a['uuid'] == item['splatad_actor_uuid']), None)
        assert actor is not None, f'Native actor absent after strict training: {item["splatad_actor_uuid"]}'
        actor_bundle = actor['bundle']
        actor_gaussians = actor['gaussians']
        export_tag='splatad_human95' if args.human_id else 'splatad_reviewed288_train_only'
        out = ROOT / f'exports/{export_tag}/objects/{args.sample_id}'
    else:
        config_path = ROOT / f'runs/splatad_reviewed288/formal/{scene}/splatad/reviewed288_v1/config.yml'
        actor_bundle = item['splatad_bundle']
        actor_gaussians = item['splatad_gaussians']
        out = ROOT / f'exports/splatad_reviewed288/objects/{args.sample_id}'
    config, pipeline, checkpoint, step = eval_setup(config_path, test_mode='val')
    assert step == 30000
    model = pipeline.model
    manager = pipeline.datamanager
    filenames = [Path(p).resolve() for p in manager.train_dataset.image_filenames]
    matches = [i for i, path in enumerate(filenames) if path == source_image]
    assert len(matches) == 1, (source_image, len(matches))
    camera_index = matches[0]
    camera = manager.train_dataset.cameras[camera_index:camera_index + 1].to(model.device)
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
