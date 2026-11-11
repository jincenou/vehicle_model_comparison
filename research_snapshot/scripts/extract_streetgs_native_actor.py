"""Export one selected StreetGS actor and render its exact source camera."""

import argparse
import json
from pathlib import Path
import re
import sys

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
EXPANSION = ROOT.parent / 'assetbench/reports/vehicle_expansion_v4/selection.json'
BASE = ROOT / 'inputs/vehicle53_selection.json'


def save_rgb(tensor, path):
    rgb = tensor.detach().float().clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    Image.fromarray(np.round(rgb * 255).astype(np.uint8)).save(path)


def save_alpha(tensor, path):
    alpha = tensor.detach().float().clamp(0, 1).squeeze(0).cpu().numpy()
    Image.fromarray(np.round(alpha * 255).astype(np.uint8)).save(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample-id', required=True)
    parser.add_argument('--stage', choices=['pilot', 'formal'], required=True)
    args, config_args = parser.parse_known_args()
    sys.argv = [sys.argv[0], *config_args]
    from lib.config import cfg
    from lib.datasets.dataset import Dataset
    from lib.models.scene import Scene
    from lib.models.street_gaussian_model import StreetGaussianModel
    from lib.models.street_gaussian_renderer import StreetGaussianRenderer
    assert cfg.mode == 'evaluate', 'Pass mode evaluate through StreetGS config options'
    assert list(cfg.data.cameras) == [0, 1, 2], 'Load the trained three-camera model first'
    assert cfg.data.type == 'Waymo'
    assert Path(cfg.source_path).resolve().is_relative_to((ROOT / 'inputs/streetgs_train_only').resolve())
    scene = 'waymo_val_' + Path(cfg.source_path).name
    assert args.sample_id.startswith(scene + '_')
    selected = next(x for x in json.loads((ROOT / 'inputs/benchmark_20261106/selection.json').read_text())['vehicles']
                    if x['sample_id'] == args.sample_id)
    manifest = EXPANSION if selected.get('review_id') else BASE
    provenance = next(x for x in json.loads(manifest.read_text())['samples']
                      if x['sample_id'] == args.sample_id)
    assert provenance['observation']['sha256'] == selected['retained_rgba_sha256']
    source_image = Path(provenance['observation']['source_image']).resolve()
    source_match = re.fullmatch(r'f(\d+)_c([1-5])', source_image.stem)
    assert source_match, source_image
    source_frame = int(source_match.group(1))
    source_cam = int(source_match.group(2)) - 1
    split = json.loads((ROOT / f'configs/streetgs_source_splits/{scene}.json').read_text())
    assert source_frame in split['train_frame_ids'], 'Selected input frame is not in the isolated training sequence'
    train_frame_index = split['train_frame_ids'].index(source_frame)
    native_image_name = f'{train_frame_index:06d}_{source_cam}'
    train_image = Path(cfg.source_path) / 'images' / f'{native_image_name}.png'
    assert train_image.is_file(), f'Source frame/camera absent from train-only scene: {train_image}'
    report = json.loads((ROOT / f'reports/streetgs_native_tracks/{scene}.json').read_text())
    identity = next(x for x in report['objects'] if x['sample_id'] == args.sample_id)
    assert identity['status'] == 'native_dynamic_actor_ready'
    actor_name = f"obj_{identity['streetgs_track_id']:03d}"
    expected_step = 100 if args.stage == 'pilot' else cfg.train.iterations
    checkpoint = Path(cfg.trained_model_dir) / f'iteration_{expected_step}.pth'
    assert checkpoint.is_file()
    with torch.no_grad():
        dataset = Dataset()
        model = StreetGaussianModel(dataset.scene_info.metadata)
        trained = Scene(gaussians=model, dataset=dataset)
        assert trained.loaded_iter == expected_step
        assert actor_name in model.obj_list
        cfg.data.cameras = [source_cam]
        source_dataset = Dataset()
        cameras = source_dataset.train_cameras[1.0] + source_dataset.test_cameras[1.0]
        matches = [cam for cam in cameras if cam.image_name == native_image_name]
        assert len(matches) == 1, (native_image_name, len(matches))
        camera = matches[0]
        actor = getattr(model, actor_name)
        assert camera.meta['frame'] >= actor.start_frame
        assert camera.meta['frame'] <= actor.end_frame
        assert actor.get_xyz.shape[0] > 0
        output = ROOT / f'exports/streetgs_reviewed288/objects/{args.sample_id}/{args.stage}'
        output.mkdir(parents=True, exist_ok=True)
        actor.save_ply(str(output / 'native_actor_local.ply'))
        renderer = StreetGaussianRenderer()
        full_scene = renderer.render(camera, model)
        model.set_visibility([actor_name])
        model.parse_camera(camera)
        actor_render = renderer.render_kernel(camera, model, white_background=True)
        rgb = actor_render['rgb']
        alpha = actor_render['acc']
        assert torch.isfinite(rgb).all() and torch.isfinite(alpha).all()
        assert float(alpha.max()) > 0 and float(rgb.std()) > 1e-5
        save_rgb(full_scene['rgb'], output / 'native_full_scene_source.png')
        save_rgb(rgb, output / 'native_actor_source.png')
        save_alpha(alpha, output / 'native_actor_alpha.png')
    result = {
        'status': 'passed', 'sample_id': args.sample_id, 'scene': scene,
        'stage': args.stage, 'source_image': str(source_image),
        'source_raw_frame': source_frame, 'train_frame_index': train_frame_index,
        'native_image_name': native_image_name, 'train_only_image': str(train_image),
        'source_relation': 'exact frozen source frame/camera loaded for evaluation; no pixel metric asserted',
        'native_track_id': identity['streetgs_track_id'], 'actor_name': actor_name,
        'gaussians': int(actor.get_xyz.shape[0]), 'checkpoint': str(checkpoint),
        'step': expected_step, 'renderer': 'native StreetGS Gaussian rasterizer',
        'actor_local_ply': str(output / 'native_actor_local.ply'),
        'full_scene_preview': str(output / 'native_full_scene_source.png'),
        'actor_preview': str(output / 'native_actor_source.png'),
        'actor_alpha': str(output / 'native_actor_alpha.png'),
        'actor_alpha_pixels_gt_0_5': int((alpha > 0.5).sum().item()),
        'rgb_std': float(rgb.std().item()),
    }
    (output / 'validation.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
