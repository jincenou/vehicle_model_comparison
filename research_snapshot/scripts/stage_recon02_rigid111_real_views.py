"""Use frozen real anchors plus technically audited same-tracklet real train views."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

R=Path(__file__).resolve().parents[1]
O=R/'inputs/recon02_rigid111_real_multiview_v1'
AUDIT=R/'inputs/rigid111_anchor_audit_v1/recon_extra_view_audit.json'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    assert not O.exists()
    single=R/'inputs/rigid111_image_inputs_v1/manifest.json'
    inputs=json.loads(single.read_text())['objects']
    audited={r['uid']:r for r in json.loads(AUDIT.read_text())['records']}
    selection=json.loads((R/'inputs/benchmark_nuscenes100_v1/selection.json').read_text())
    splits={};split_records={}
    for s in selection['scenes']:
        if s['scene'] not in {x['scene'] for x in inputs}:continue
        p=Path(s['split']);assert sha(p)==s['split_sha256']
        splits[s['scene']]={x['filename']:x for x in json.loads(p.read_text())['files']}
        split_records[s['scene']]={'path':str(p),'sha256':sha(p)}
    records=[]
    for row in inputs:
        views=[];excluded=[];seen=set()
        if row['scene_training_split']=='train':
            views.append(dict(image=row['image_rgba'],sha256=row['image_rgba_sha256'],size=[512,512],
                raw_image=row['raw_source'],raw_sha256=row['raw_source_sha256'],
                sample_data_token=row['sample_data_token'],timestamp_us=row['timestamp_us'],camera=row['camera'],
                split='train',crop_box=row['raw_crop_box'],H_canvas_from_raw_pixel_centres=row['H_canvas_from_raw_pixel_centres'],
                identity_evidence='Exact source anchor of the frozen published asset recovered from frontend export; no native instance claim'))
            seen.add(row['raw_source'])
        else:
            excluded.append(dict(path=row['raw_source'],reason='source_anchor_in_existing_eval_split; not used by Recon; not independent heldout because pipeline already used it'))
        for view in audited[row['uid']]['accepted_train_views']:
            v=dict(view)
            raw=v.get('raw_image',v.get('source_image'));image=v.get('image',v.get('rgba_path'))
            if raw in seen:continue
            relative=str(Path(raw).relative_to(('/data/nuScenes'.replace('/data', __import__("os").environ.get('RIGID_DATA_ROOT', '/data')))))
            split=splits[row['scene']][relative];assert split['split']=='train'
            expected=v.get('sha256',v.get('rgba_sha256'));assert sha(image)==expected
            assert sha(raw)==v['raw_sha256']
            v.update(image=image,sha256=expected,raw_image=raw,split='train',
                sample_data_token=split.get('sample_data_token',split.get('token')),
                timestamp_us=split.get('timestamp_us',split.get('timestamp')),camera=split['channel'])
            assert v.get('identity_evidence'), 'Audit must disclose identity evidence'
            views.append(v);seen.add(raw)
        assert len(views)==len({v['sample_data_token'] for v in views})
        status='ready_multi_view' if len(views)>1 else 'single_view_requires_native_pilot' if views else 'missing_eligible_real_view'
        records.append(dict(uid=row['uid'],sample_id=row['sample_id'],scene=row['scene'],category=row['category'],
            status=status,views=views,real_input_count=len(views),generated_input_count=0,excluded_views=excluded,
            original_input_mode='flux_only',original_real_count=0,native_instance_identity_verified=False,
            native_identity={},identity_scope='Original frontend object/tracklet and raw-photo technical provenance; not nuScenes native instance verification',
            source_image_model_anchor=row['image_rgba'],real_heldout_eligible=False))
    assert len(records)==111
    result=dict(schema=1,status='frozen_real_view_inventory_pending_runtime',created_at=datetime.now(timezone.utc).isoformat(),
        selection_sha256=sha(R/'inputs/benchmark_nuscenes100_v1/selection.json'),
        cohort_manifest_sha256=sha(R/'inputs/rigid543_nusc100_v1/manifest.json'),image_manifest_sha256=sha(single),
        extra_view_audit=str(AUDIT),extra_view_audit_sha256=sha(AUDIT),script_sha256=sha(__file__),
        objects=records,scene_splits=split_records,counts=dict(Counter(r['status'] for r in records)),
        view_count_histogram=dict(Counter(r['real_input_count'] for r in records)),
        policy='Only real train views tied to the original frontend asset and tracklet. The B0 route did not submit auxiliary views to vehicle VLM review; absence of that review is not an explicit rejection. No FLUX, duplicates, heldout views or appearance-driven selection. Preserve all111 including unsupported inputs.',
        real_heldout_eligible=False,common_camera_qualified=False)
    O.mkdir();(O/'manifest.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(counts=result['counts'],view_count_histogram=result['view_count_histogram'])))

if __name__=='__main__':main()
