"""Freeze real source-protected Waymo time-block splits before StreetGS staging."""

from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[1]
REPO = Path(('/opt/rigid-benchmark/project/baselines/street_gaussians'.replace('/opt/rigid-benchmark/project', __import__("os").environ.get('RIGID_PROJECT_ROOT', '/opt/rigid-benchmark/project'))))
sys.path.insert(0, str(REPO / 'submodules/simple-waymo-open-dataset-reader'))
from simple_waymo_open_dataset_reader import WaymoDataFileReader


def strings(value):
    if isinstance(value, dict):
        for item in value.values():
            yield from strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from strings(item)
    elif isinstance(value, str):
        yield value


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def main():
    frozen_path = ROOT / 'inputs/benchmark_20261106/selection.json'
    frozen = json.loads(frozen_path.read_text())
    approved = {r['uid'] for r in frozen['vehicles']}
    old_path = ROOT / 'inputs/vehicle53_selection.json'
    new_path = Path(('/opt/rigid-benchmark/assetbench/reports/vehicle_expansion_v4_selected/selection.json'.replace('/opt/rigid-benchmark/assetbench', __import__("os").environ.get('RIGID_ASSETBENCH', '/opt/rigid-benchmark/assetbench')).replace('/opt/rigid-benchmark/project', __import__("os").environ.get('RIGID_PROJECT_ROOT', '/opt/rigid-benchmark/project'))))
    original = json.loads(old_path.read_text())['samples']
    expansion = json.loads(new_path.read_text())['samples']
    rows = [r for r in original + expansion if r['uid'] in approved and r['dataset'] == 'waymo']
    assert len(rows) == 89 and len({r['uid'] for r in rows}) == 89
    sources = json.loads((ROOT / 'inputs/streetgs_scene_source/manifest.json').read_text())
    output = ROOT / 'configs/streetgs_source_splits'
    output.mkdir(parents=True, exist_ok=True)
    result = {'status': 'source_protected_splits_frozen',
              'rule': 'Original StreetGS validation stride 4 on raw chronological frames; candidate heldout frame i%4==3 is moved to train if any retained production provenance uses that frame. All cameras and LiDAR share frame roles. Model initialization later receives only physically staged train observations.',
              'frozen_selection_sha256': sha(frozen_path),
              'old_full_provenance_sha256': sha(old_path),
              'new_full_provenance_sha256': sha(new_path),
              'scenes': []}
    for source in sources['scenes']:
        scene = source['scene']
        selected = [r for r in rows if r['scene_id'] == scene]
        assert len(selected) == source['selected_vehicle_count']
        protected = set()
        protected_paths = set()
        for row in selected:
            for value in strings(row):
                match = re.search(r'/f(\d{4})_c[1-5]\.jpg$', value)
                if match:
                    protected.add(int(match.group(1)))
                    protected_paths.add(value)
        assert protected, scene
        reader = WaymoDataFileReader(source['tfrecord'])
        try:
            total = len(reader.get_record_table())
        finally:
            reader.file.close()
        assert total >= 150 and max(protected) < total, (scene, total, max(protected))
        heldout = [i for i in range(total) if i % 4 == 3 and i not in protected]
        train = [i for i in range(total) if i not in heldout]
        assert heldout and train and protected <= set(train)
        split = {'status': 'frozen', 'scene': scene, 'raw_tfrecord': source['tfrecord'],
                 'raw_tfrecord_bytes': source['tfrecord_bytes'],
                 'raw_frame_count': total, 'train_frame_ids': train,
                 'heldout_frame_ids': heldout, 'protected_frame_ids': sorted(protected),
                 'protected_source_paths': sorted(protected_paths),
                 'target_sample_ids': [r['sample_id'] for r in selected],
                 'target_uids': [r['uid'] for r in selected],
                 'target_categories': dict(Counter(r['category'] for r in selected)),
                 'rule': result['rule']}
        target = output / (scene + '.json')
        if target.exists():
            assert json.loads(target.read_text()) == split
        else:
            target.write_text(json.dumps(split, ensure_ascii=False, indent=2) + '\n')
        result['scenes'].append({'scene': scene, 'split': str(target),
                                 'split_sha256': sha(target), 'train_frames': len(train),
                                 'heldout_frames': len(heldout),
                                 'protected_frames': len(protected),
                                 'targets': len(selected)})
    report = ROOT / 'reports/streetgs_source_split_preflight.json'
    if report.exists():
        assert json.loads(report.read_text()) == result
    else:
        report.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'status': result['status'], 'scenes': result['scenes']}))


if __name__ == '__main__':
    main()
