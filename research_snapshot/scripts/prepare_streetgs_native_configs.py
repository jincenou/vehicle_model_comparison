"""Bind the official StreetGS Waymo example to source-isolated benchmark scenes."""

import hashlib
import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
OFFICIAL = Path(('/opt/rigid-benchmark/project/baselines/street_gaussians/configs/example/waymo_train_031.yaml'.replace('/opt/rigid-benchmark/project', __import__("os").environ.get('RIGID_PROJECT_ROOT', '/opt/rigid-benchmark/project'))))
SCENES = ['waymo_val_030', 'waymo_val_031', 'waymo_val_032', 'waymo_val_033',
          'waymo_val_034', 'waymo_val_035', 'waymo_val_037', 'waymo_val_038']


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    original = yaml.safe_load(OFFICIAL.read_text())
    assert original['data']['type'] == 'Waymo'
    assert original['data']['use_colmap'] is True
    assert original['model']['nsg']['include_obj'] is True
    assert original['model']['nsg']['include_sky'] is True
    assert original['train']['iterations'] == 50000
    assert original['data']['split_train'] == 1 and original['data']['split_test'] == -1
    out = ROOT / 'configs/streetgs_reviewed288'
    out.mkdir(parents=True, exist_ok=True)
    entries = []
    for gpu, scene in enumerate(SCENES):
        audit_path = ROOT / f'reports/streetgs_train_only/{scene}.json'
        audit = json.loads(audit_path.read_text())
        assert audit['status'] == 'physically_isolated_train_only'
        assert audit['heldout_rgb_lidar_present_in_train_root'] is False
        for stage in ('pilot', 'formal'):
            if stage == 'pilot' and scene != 'waymo_val_031':
                continue
            config = yaml.safe_load(OFFICIAL.read_text())
            model_dir = ROOT / f'runs/streetgs_reviewed288/{stage}/{scene}'
            config['source_path'] = audit['train_root']
            config['model_path'] = str(model_dir)
            config['record_dir'] = str(ROOT / f'logs/streetgs_reviewed288/{stage}/{scene}')
            config['exp_name'] = f'reviewed288_{scene}_{stage}'
            config['gpus'] = [0 if stage == 'pilot' else gpu]
            config['data']['selected_frames'] = [0, audit['train_frames'] - 1]
            if stage == 'pilot':
                config['train']['iterations'] = 100
                config['train']['test_iterations'] = []
                config['train']['save_iterations'] = [100]
                config['train']['checkpoint_iterations'] = [100]
            path = out / f'{scene}_{stage}.yaml'
            path.write_text(yaml.safe_dump(config, sort_keys=False))
            entries.append({'scene': scene, 'stage': stage, 'gpu': config['gpus'][0],
                            'config': str(path), 'config_sha256': sha(path),
                            'train_root': audit['train_root'],
                            'split_sha256': audit['source_split_sha256'],
                            'train_frames': audit['train_frames'],
                            'heldout_frames': audit['heldout_frames'],
                            'iterations': config['train']['iterations']})
    report = {'status': 'configs_prepared', 'official_config': str(OFFICIAL),
              'official_config_sha256': sha(OFFICIAL),
              'adaptations': ['absolute source and output paths', 'reindexed selected frame endpoint',
                              'per-job GPU index', 'pilot-only 100-step save and checkpoint schedule'],
              'entries': entries}
    target = ROOT / 'reports/streetgs_reviewed288_config_audit.json'
    target.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'status': report['status'], 'configs': len(entries),
                      'formal_scenes': len([e for e in entries if e['stage'] == 'formal'])}))


if __name__ == '__main__':
    main()
