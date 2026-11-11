"""Publish a hash-bound snapshot of accepted orientation for evaluation consumers."""
import argparse, hashlib, json, sys
from collections import Counter
from pathlib import Path

R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R / 'evaluation_nuscenes_v1/scripts'))
from asset_orientation_contract import make_contract

p = argparse.ArgumentParser()
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=False)
inv = json.loads((R / 'configs/asset_orientation_v1/inventory.json').read_text())
lookup = {(r['sample_id'], r['method']): r for r in inv['rows']}
for f in (R / 'configs/asset_orientation_v1/future_inventories').glob('*.json'):
    for r in json.loads(f.read_text())['rows']:
        lookup[r['sample_id'], r['method']] = r
approved = R / 'configs/asset_orientation_v1/approved'
records = []
for key, row in lookup.items():
    decision_path = approved / ('__'.join(key) + '.json')
    record = dict(sample_id=key[0], method=key[1],
                  status='heading_not_accepted', physical_registration_qualified=False)
    if decision_path.exists():
        decision = json.loads(decision_path.read_text())
        contract = make_contract(row, decision)
        contract['decision_path'] = str(decision_path)
        contract['decision_sha256'] = hashlib.sha256(decision_path.read_bytes()).hexdigest()
        file = a.output / decision_path.name
        file.write_text(json.dumps(contract, indent=2, allow_nan=False) + '\n')
        record.update(status='semantic_heading_qualified', contract=str(file),
                      contract_sha256=hashlib.sha256(file.read_bytes()).hexdigest())
    elif row['status'] != 'ready_for_orientation_audit':
        record['status'] = row['status']
    records.append(record)
manifest = dict(schema='metric_orientation_inventory_v1', records=records,
                counts=dict(Counter(r['status'] for r in records)),
                scene_methods=dict(methods=['splatad', 'omnire', 'streetgs'],
                    status='native_track_frame_contract_required',
                    rule='Use verified instance pose at frozen source time; do not apply generated-asset PCA to world coordinates.'),
                scope='Semantic coordinate contract only; no metric is enabled by this manifest alone.')
(a.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
print(json.dumps(manifest['counts']))
