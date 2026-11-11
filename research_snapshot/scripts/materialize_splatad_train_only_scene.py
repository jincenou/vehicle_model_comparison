"""Physically isolate nuScenes training observations for a native SplatAD scene."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parents[1]
CHANNELS = {'CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK',
            'CAM_BACK_LEFT', 'CAM_BACK_RIGHT', 'LIDAR_TOP'}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene', required=True)
    parser.add_argument('--scope', choices=('vehicle', 'human', 'reconstruction_only'), required=True)
    args = parser.parse_args()
    scene = args.scene
    tag = {'vehicle': 'reviewed288', 'human': 'human_reviewed95',
           'reconstruction_only': 'reconstruction_only'}[args.scope]
    split_dir = {'vehicle': 'nuscenes_splits_reviewed288',
                 'human': 'nuscenes_human_splits_reviewed95',
                 'reconstruction_only': 'nuscenes_reconstruction_only'}[args.scope]
    split_path = ROOT / 'configs' / split_dir / f'{scene}.json'
    split = json.loads(split_path.read_text())
    assert split['scene'] == scene
    if args.scope == 'reconstruction_only':
        assert split['variant'] == 'reconstruction_only_no_real_heldout'
        assert not split['heldout_sample_tokens']
        assert set(split['protected_sample_tokens']) == {
            row['sample_token'] for row in split['files']}
    else:
        assert split['heldout_sample_tokens']
    source = Path(split['data_root'])
    target = ROOT / 'inputs/splatad_train_only' / tag / scene
    manifest_path = ROOT / 'configs/splatad_train_only' / tag / f'{scene}.json'
    report_path = ROOT / 'reports/splatad_train_only' / tag / f'{scene}.json'
    assert not target.exists() and not manifest_path.exists() and not report_path.exists()
    version = source / 'v1.0-trainval'
    files = [row for row in split['files'] if row['split'] == 'train' and row['channel'] in CHANNELS]
    heldout = set(split['heldout_sample_tokens'])
    assert files and all(row['sample_token'] not in heldout for row in files)
    train_tokens = {row['sample_token'] for row in files}
    samples = json.loads((version / 'sample.json').read_text())
    train_samples = sorted((row for row in samples if row['token'] in train_tokens),
                           key=lambda row: row['timestamp'])
    assert len(train_samples) == len(train_tokens)
    for i, row in enumerate(train_samples):
        row['prev'] = train_samples[i - 1]['token'] if i else ''
        row['next'] = train_samples[i + 1]['token'] if i + 1 < len(train_samples) else ''
    data_rows = json.loads((version / 'sample_data.json').read_text())
    keep = {row['sample_data_token'] for row in files}
    filtered = [row for row in data_rows if row['token'] in keep]
    assert len(filtered) == len(keep) == len(files)
    by_calib = {}
    for row in filtered:
        by_calib.setdefault(row['calibrated_sensor_token'], []).append(row)
    for rows in by_calib.values():
        rows.sort(key=lambda row: row['timestamp'])
        for i, row in enumerate(rows):
            row['prev'] = rows[i - 1]['token'] if i else ''
            row['next'] = rows[i + 1]['token'] if i + 1 < len(rows) else ''
    target.mkdir(parents=True)
    for sub in ('samples', 'sweeps'):
        (target / sub).mkdir()
    (target / 'maps').symlink_to(source / 'maps', target_is_directory=True)
    metadata = target / 'v1.0-trainval'
    metadata.mkdir()
    for path in version.glob('*.json'):
        if path.name == 'sample.json':
            value = samples
        elif path.name == 'sample_data.json':
            value = filtered
        elif path.name == 'scene.json':
            value = json.loads(path.read_text())
            assert len(value) == 1 and value[0]['name'] == scene
            value[0].update(nbr_samples=len(train_samples),
                            first_sample_token=train_samples[0]['token'],
                            last_sample_token=train_samples[-1]['token'])
        else:
            shutil.copyfile(path, metadata / path.name)
            continue
        (metadata / path.name).write_text(json.dumps(value))
    for row in files:
        relative = Path(row['filename'])
        original = source / relative
        assert original.is_file() and original.stat().st_size > 0
        dest = target / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.symlink_to(original)
    actual = {str(path.relative_to(target)) for folder in ('samples', 'sweeps')
              for path in (target / folder).rglob('*') if path.is_symlink()}
    expected = {row['filename'] for row in files}
    assert actual == expected
    manifest = {'scene': scene, 'data_root': str(target),
                'source_split': str(split_path), 'source_split_sha256': digest(split_path),
                'heldout_sample_tokens': sorted(heldout),
                'files': [{**row, 'split': 'train'} for row in files],
                'eval_policy': 'Internal validation reuses a train observation; real heldout files are absent.'
                               if args.scope != 'reconstruction_only'
                               else 'No independent real heldout exists; internal validation is a train shadow and heldout metrics are ineligible.'}
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    report = {'status': 'verified_reconstruction_only' if args.scope == 'reconstruction_only' else 'verified_train_only',
              'scene': scene, 'scope': args.scope,
              'data_root': str(target), 'manifest': str(manifest_path),
              'manifest_sha256': digest(manifest_path),
              'source_split_sha256': digest(split_path),
              'train_files': len(files), 'heldout_blocks': len(heldout),
              'heldout_rgb_lidar_present': False}
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
