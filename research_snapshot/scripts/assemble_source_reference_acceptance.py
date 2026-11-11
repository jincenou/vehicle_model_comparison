"""Assemble reviewed reference eligibility without dropping frozen objects."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

R = Path(__file__).resolve().parents[1]

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def assemble(manifest, reviews):
    manifest = Path(manifest).resolve()
    inventory = json.loads(manifest.read_text())
    objects = inventory['objects']
    index = {o['sample_id']: o for o in objects}
    assert len(index) == len(objects) == 432
    audited = {}
    evidence = []
    for path in reviews:
        path = Path(path).resolve()
        review = json.loads(path.read_text())
        assert review['schema'] == 'source_reference_visual_review_v1'
        assert Path(review['reference_manifest']).resolve() == manifest
        assert review['reference_manifest_sha256'] == sha(manifest)
        assert sha(review['decisions_source']) == review['decisions_sha256']
        for decision in review['records']:
            sid = decision['sample_id']
            assert sid in index and sid not in audited, 'Unknown or duplicate review: ' + sid
            contract = Path(index[sid]['contract'])
            assert Path(decision['reference_contract']) == contract
            assert decision['reference_contract_sha256'] == index[sid]['contract_sha256'] == sha(contract)
            assert sha(decision['review_image']) == decision['review_image_sha256']
            approved = decision['decision'] == 'accepted_visible_reference'
            assert decision['reference_visual_qualified'] == approved
            if approved:
                c = json.loads(contract.read_text())
                assert c['identity_status'] == 'recorded_identity_supported_at_source'
                assert c['native_reference_pixels_verified'] and c['foreground_rgb_max_error'] == 0
                assert c['raw_frame_edge_contact'] is False
                assert decision['external_occlusion'] == decision['truncation'] == 'none_visible'
                assert decision['mask_alignment'] == 'acceptable_visible_target'
            audited[sid] = (decision, str(path))
        evidence.append(dict(path=str(path), sha256=sha(path)))
    rows = []
    for item in objects:
        decision, path = audited.get(item['sample_id'], ({}, None))
        rows.append(dict(sample_id=item['sample_id'], scene=item['scene'], category=item['category'],
                         reference_contract=item['contract'], reference_contract_sha256=item['contract_sha256'],
                         reference_visual_qualified=decision.get('reference_visual_qualified', False),
                         reference_review_decision=decision.get('decision', 'not_reviewed'),
                         reference_review=path, predicted_asset_registration_qualified=False,
                         formal_metrics_qualified=False))
    return dict(created_at_utc=datetime.now(timezone.utc).isoformat(), schema='source_reference_acceptance_v1',
                reference_manifest=str(manifest), reference_manifest_sha256=sha(manifest), reviews=evidence,
                builder_sha256=sha(__file__), reviewed_count=len(audited), planned_objects=432,
                counts=dict(Counter(r['reference_review_decision'] for r in rows)),
                reference_visual_qualified_count=sum(r['reference_visual_qualified'] for r in rows), objects=rows,
                scope='Reference-only Codex visual audit; not user annotations or asset registration approval')

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', type=Path, default=R/'configs/source_reference_vehicle432_v1/manifest.json')
    p.add_argument('--reviews', type=Path, nargs='+', required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    data = assemble(a.manifest, a.reviews)
    with a.out.open('x') as f:
        json.dump(data, f, indent=2)
        f.write('\n')
    print(json.dumps(data['counts']))

if __name__ == '__main__':
    main()
