"""Validate release metadata and scope without loading models or private data."""
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
CATEGORIES = dict(car=317, truck=78, van=31, bus=6, barrier=12, pole=12,
                  bollard=11, road_debris=11, cone=9, trash_can=7, hydrant=6)
METHODS = {'ours', 'ah', 'trellis_original', 'trellis2', 'hunyuan21',
           'recon02', 'splatad', 'omnire', 'streetgs'}


def read(path):
    return json.loads(path.read_text())


def require(ok, message):
    if not ok:
        raise ValueError(message)


def validate_membership(cohort, scenes):
    rows = cohort['objects']
    ids = [x['scene'] for x in scenes['scenes']]
    require(len(ids) == len(set(ids)) == 100, 'Expected 100 unique frozen scenes')
    require(len(rows) == 500, 'Expected 500 frozen assets')
    require(len({x['sample_id'] for x in rows}) == 500, 'Duplicate sample IDs')
    require(len({x['uid'] for x in rows}) == 500, 'Duplicate asset UIDs')
    require(all(x['scene'] in ids for x in rows), 'Asset outside frozen scenes')
    require(dict(Counter(x['category'] for x in rows)) == CATEGORIES, 'Category counts changed')
    require(Counter(x['cohort_origin'] for x in rows) ==
            Counter(previous_frozen432=432, user_approved_nonvehicle111=68), 'Cohort origins changed')
    require(set(cohort['methods']) == METHODS and len(cohort['methods']) == 9, 'Methods changed')
    require(Counter(x['input_mode'] for x in rows) == Counter(flux_plus_real=364, flux_only=136),
            'Historical pipeline input disclosure changed')
    require(cohort['counts']['records'] == 500 and cohort['counts']['previous'] == 432
            and cohort['counts']['added'] == 68, 'Declared counts disagree')
    require(cohort['category_counts'] == CATEGORIES, 'Declared categories disagree')
    for row in rows:
        require(all(re.fullmatch('[0-9a-f]{64}', v) for k, v in row.items() if k.endswith('sha256')),
                'Malformed content hash')


def main():
    validate_membership(read(ROOT/'configs/cohort_rigid500.json'), read(ROOT/'configs/scenes_nuscenes100.json'))
    require(all(x['enabled'] is False for x in read(ROOT/'configs/metric_plan.json')['metrics']),
            'Formal metric gate enabled')
    provenance = read(ROOT/'provenance/source_files.json')
    for row in provenance['files']:
        path = ROOT/row['published']
        require(path.resolve().is_relative_to(ROOT), 'Provenance escapes repository')
        require(hashlib.sha256(path.read_bytes()).hexdigest() == row['published_sha256'],
                'Published source hash mismatch: '+row['published'])
    files = [p for p in ROOT.rglob('*') if p.is_file() and not any(x in p.parts for x in ['.git','__pycache__','.venv'])]
    for p in files:
        rel = p.relative_to(ROOT)
        require(not p.is_symlink(), 'Release contains live symlink')
        require(p.suffix not in {'.pt','.pth','.ckpt','.safetensors','.ply','.glb','.png','.jpg','.jpeg','.mp4'},
                'Payload file: '+str(rel))
        require(p.stat().st_size < 2*1024*1024, 'Unexpectedly large file: '+str(rel))
        if p.suffix == '.py':
            ast.parse(p.read_text(), filename=str(rel))
        if p.suffix in {'.py','.json','.md','.txt','.yml'}:
            text = p.read_text()
            require(not re.search(r'(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}|-----BEGIN (?:OPENSSH |RSA |EC )?PRIVATE KEY-----)', text), 'Credential-shaped content: '+str(rel))
            if rel.parts[0] in {'research_snapshot','configs','evaluation'}:
                require(not re.search(r'/(?:mnt|home)/', text), 'Private deployment path: '+str(rel))
    print(json.dumps(dict(status='passed', assets=500, frozen_scenes=100, methods=9,
                          published_source_files=len(provenance['files']), files=len(files),
                          formal_metrics_enabled=False), indent=2))


if __name__ == '__main__':
    main()
