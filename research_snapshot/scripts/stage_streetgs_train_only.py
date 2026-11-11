"""Stage native StreetGS Waymo files with no heldout RGB or LiDAR access."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
FULL = ROOT / 'inputs/streetgs_full_conversion'
TRAIN = ROOT / 'inputs/streetgs_train_only'
REPORTS = ROOT / 'reports/streetgs_train_only'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def link_exact(source, target):
    assert source.is_file() and source.stat().st_size > 0, source
    assert not target.exists() and not target.is_symlink(), target
    target.symlink_to(source)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene', required=True)
    args = parser.parse_args()
    split_path = ROOT / 'configs/streetgs_source_splits' / (args.scene + '.json')
    split = json.loads(split_path.read_text())
    assert split['status'] == 'frozen' and split['scene'] == args.scene
    number = args.scene.rsplit('_', 1)[1]
    source = FULL / number
    assert (source / 'track/track_info.txt').is_file()
    assert (source / 'pointcloud.npz').is_file()
    destination = TRAIN / number
    destination.mkdir(parents=True, exist_ok=False)
    for name in ('images', 'ego_pose', 'dynamic_mask', 'track'):
        (destination / name).mkdir()
    for name in ('intrinsics', 'extrinsics'):
        assert (source / name).is_dir()
        (destination / name).symlink_to(source / name, target_is_directory=True)

    frame_map = {old: new for new, old in enumerate(split['train_frame_ids'])}
    heldout = set(split['heldout_frame_ids'])
    assert not heldout & set(frame_map)
    assert set(split['protected_frame_ids']) <= set(frame_map)
    for old, new in frame_map.items():
        for cam in range(5):
            link_exact(source / 'images' / f'{old:06d}_{cam}.png',
                       destination / 'images' / f'{new:06d}_{cam}.png')
            link_exact(source / 'dynamic_mask' / f'{old:06d}_{cam}.png',
                       destination / 'dynamic_mask' / f'{new:06d}_{cam}.png')
            link_exact(source / 'ego_pose' / f'{old:06d}_{cam}.txt',
                       destination / 'ego_pose' / f'{new:06d}_{cam}.txt')
        link_exact(source / 'ego_pose' / f'{old:06d}.txt',
                   destination / 'ego_pose' / f'{new:06d}.txt')

    timestamps = json.loads((source / 'timestamps.json').read_text())
    remapped = {channel: {f'{new:06d}': values[f'{old:06d}']
                          for old, new in frame_map.items()}
                for channel, values in timestamps.items()}
    (destination / 'timestamps.json').write_text(json.dumps(remapped, indent=2) + '\n')

    lines = (source / 'track/track_info.txt').read_text().splitlines()
    assert lines and lines[0].startswith('frame_id ')
    filtered = [lines[0]]
    for line in lines[1:]:
        parts = line.split()
        if int(parts[0]) in frame_map:
            parts[0] = str(frame_map[int(parts[0])])
            filtered.append(' '.join(parts))
    assert len(filtered) > 1
    (destination / 'track/track_info.txt').write_text('\n'.join(filtered) + '\n')
    visibility = json.loads((source / 'track/track_camera_vis.json').read_text())
    remapped_visibility = {track_id: {str(frame_map[int(frame)]): cameras
                                       for frame, cameras in frame_dict.items()
                                       if int(frame) in frame_map}
                           for track_id, frame_dict in visibility.items()}
    remapped_visibility = {track_id: frames for track_id, frames in remapped_visibility.items()
                           if frames}
    (destination / 'track/track_camera_vis.json').write_text(
        json.dumps(remapped_visibility, indent=1) + '\n')
    shutil.copyfile(source / 'track/track_ids.json',
                    destination / 'track/track_ids.json')

    with np.load(source / 'pointcloud.npz', allow_pickle=True) as full:
        points = full['pointcloud'].item()
        projections = full['camera_projection'].item()
    assert len(points) == len(projections) == split['raw_frame_count']
    kept_points = {new: points[old] for old, new in frame_map.items()}
    kept_projections = {new: projections[old] for old, new in frame_map.items()}
    np.savez_compressed(destination / 'pointcloud.npz',
                        pointcloud=kept_points, camera_projection=kept_projections)

    actual = {p.name for p in (destination / 'images').iterdir()}
    expected = {f'{new:06d}_{cam}.png' for new in frame_map.values() for cam in range(5)}
    assert actual == expected
    assert {p.name for p in (destination / 'dynamic_mask').iterdir()} == expected
    assert len(kept_points) == len(frame_map)
    for folder in ('images', 'dynamic_mask'):
        assert not any(int(p.resolve().name.split('_')[0]) in heldout
                       for p in (destination / folder).iterdir())
    REPORTS.mkdir(parents=True, exist_ok=True)
    audit = {'status': 'physically_isolated_train_only', 'scene': args.scene,
             'source_full_conversion': str(source), 'train_root': str(destination),
             'source_split_sha256': sha(split_path), 'raw_frames': split['raw_frame_count'],
             'train_frames': len(frame_map), 'heldout_frames': len(heldout),
             'train_images': len(actual), 'train_dynamic_masks': len(expected),
             'train_lidar_frames': len(kept_points),
             'train_track_rows': len(filtered) - 1,
             'heldout_rgb_lidar_present_in_train_root': False,
             'pointcloud_sha256': sha(destination / 'pointcloud.npz'),
             'time': time.time()}
    (REPORTS / (args.scene + '.json')).write_text(json.dumps(audit, indent=2) + '\n')
    print(json.dumps(audit), flush=True)


if __name__ == '__main__':
    main()
