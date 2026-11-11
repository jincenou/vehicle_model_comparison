"""Incremental CPU-only preparation for the frozen 500-object benchmark.

References existing artifacts, verifies content and exports accepted coordinate
contracts. Does not generate assets, fit poses, score metrics, or edit job states.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
MODULES = ROOT/'evaluation_nuscenes_v1/scripts'
if not MODULES.exists(): MODULES = ROOT.parent/'evaluation'  # Public release layout.
sys.path.insert(0, str(MODULES))
from preparation_cache import HashCache, Deferred
from asset_orientation_contract import make_contract
from source_render_gate import validate_native_source
from PIL import Image

METHODS = ['ours','ah','trellis_original','trellis2','hunyuan21','recon02','splatad','omnire','streetgs']
SCENE = METHODS[-3:]


def read(path):
    return json.loads(Path(path).read_text())


def save(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+'\n'
    if path.exists() and path.read_text() == content: return
    temp = path.with_name(path.name+'.tmp'); temp.write_text(content); temp.replace(path)


def load_snapshot(path, evidence):
    raw = Path(path).read_bytes()
    evidence.append(dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest()))
    return json.loads(raw)


def declared_artifact(cohort, method, jobs, old_ah):
    sid = cohort['sample_id']
    if method == 'ours':
        return dict(generation_status='existing_frozen_asset', asset=cohort['asset'],
                    asset_sha256=cohort['asset_expected_sha256'], representation='gaussian_ply',
                    input_regime=cohort['input_mode']), None
    if method == 'ah' and cohort['cohort_origin'] == 'previous_frozen432':
        old = old_ah[sid]
        path = Path(old['asset'])
        state = read(path.parents[2]/'job.json')
        return dict(generation_status='existing_frozen_asset', asset=str(path), asset_sha256=old['asset_sha256'],
                    representation='gaussian_ply', input_regime='real_source_image_and_mask'), state
    job = jobs[method][sid]
    row = dict(generation_status=job['status'], generation_reason=job.get('reason'),
               representation='gaussian_ply' if method in ['ah','trellis_original','recon02'] else 'glb_mesh')
    if job['status'] != 'completed': return row, None
    folder = Path(job.get('run_dir', jobs['_prep']/'runs/trellis2_vehicle432_v1'/sid))
    state = read(folder/('job.json' if method == 'ah' else 'state.json'))
    if method == 'ah':
        path, expected = job['output'], job['output_sha256']
    elif row['representation'] == 'gaussian_ply':
        path = str(folder/'native_gaussians.ply'); expected = state['output_hashes']['native_gaussians.ply']
    else:
        path, expected = state['output'], state['output_sha256']
    row.update(asset=path, asset_sha256=expected, generation_evidence=str(folder/('job.json' if method=='ah' else 'state.json')),
               generation_evidence_sha256=hashlib.sha256((folder/('job.json' if method=='ah' else 'state.json')).read_bytes()).hexdigest(),
               input_regime='real_multiview' if method=='recon02' else 'real_source_image_and_mask')
    return row, state


def check_actual_inputs(method, state, frozen_input, multiview, cache):
    if method == 'ours': return dict(status='historical_pipeline_regime_disclosed_not_equal_real_input_ablation')
    if method == 'ah':
        source = read(Path(state['input_root'])/'source_manifest.json')
        if source['sample_id'] != state['sample_id']: raise ValueError('AH source identity mismatch')
        if state.get('synthetic_input_ids'): raise ValueError('Unexpected synthetic AH inputs')
        cache.check(source['image'], source['source_image_sha256'])
        mask_hash = cache.check(source['mask'], source.get('source_mask_sha256'))
        if state.get('real_input_ids') != [source['source_image_sha256']]: raise ValueError('AH input hash list mismatch')
        return dict(status='recorded_real_image_and_mask_verified' if source.get('source_mask_sha256')
                    else 'real_image_verified_historical_mask_hash_missing', real_views=1,
                    mask_current_sha256=mask_hash,
                    mask_hash_origin='generation_record' if source.get('source_mask_sha256') else 'first_preparation_intake',
                    original_mask_hash_verified=bool(source.get('source_mask_sha256')))
    if method == 'recon02':
        views = multiview['views']; actual = state['inputs']
        if not views or len(views) != len(actual): raise ValueError('Real-view count mismatch')
        if state.get('generated_input_count') != 0: raise ValueError('Synthetic Recon input')
        for v, used in zip(views, actual):
            if v['split'] != 'train' or used['path'] != v['image'] or used['sha256'] != v['sha256']:
                raise ValueError('Recon views differ from frozen real training inputs')
            cache.check(used['path'], used['sha256'])
        return dict(status='frozen_real_multiview_verified', real_views=len(views))
    if state['input'] != frozen_input['image_rgba'] or state['input_sha256'] != frozen_input['image_rgba_sha256']:
        raise ValueError('Generation input differs from frozen real RGBA')
    if frozen_input['generated_input_count'] != 0 or frozen_input['real_input_count'] != 1:
        raise ValueError('Frozen single-image input regime differs')
    cache.check(state['input'], state['input_sha256'])
    return dict(status='frozen_real_rgba_verified', real_views=1)


def scene_intake(row, entry, job, cache):
    if entry is None or not entry['reference_visual_qualified']:
        row['preparation_status'] = 'waiting_reference_qualification'; return
    if job.get('status') != 'completed':
        row.update(preparation_status='waiting_native_scene_output', reason=job.get('reason',job.get('inventory_status',job.get('status','not_available')))); return
    refpath = Path(entry['reference_contract']); cache.check(refpath,entry['reference_contract_sha256']); ref=read(refpath)
    cache.check(ref['raw_source'],ref['raw_source_sha256'])
    for f in ref['files'].values(): cache.check(f['path'],f['sha256'])
    vp=Path(job['validation']); cache.check(vp,job['validation_sha256']); validation=read(vp)
    cp=Path(ref['camera_contract']); cache.check(cp,ref['camera_contract_sha256']); camera=read(cp)
    gate=validate_native_source(validation,camera,ref['camera_contract_sha256'],ref,row['method'])
    rendered={}
    for key in ['actor_preview','actor_alpha']:
        file=Path(validation[key]); digest=cache.check(file)
        with Image.open(file) as image:
            if list(image.size)!=camera['render_wh']: raise ValueError('Native render dimensions differ from camera')
            image.verify()
        rendered[key]=dict(path=str(file),sha256=digest,hash_origin='first_preparation_intake_not_original_batch_binding')
    row.update(preparation_status='source_camera_intake_ready',source_camera_intake=gate,native_source_camera_qualified=True,
               source_render_files=rendered,reference_contract=str(refpath),reference_contract_sha256=entry['reference_contract_sha256'],
               validation=str(vp),validation_sha256=job['validation_sha256'],
               eligible_next_action='Independent scoring-stage verification of fixed source-image protocol; no heldout claim')


def run_once(root, out, byte_budget):
    prep=root/'model_preparation/20261109_2100_CST'; evidence=[]
    cohort=load_snapshot(root/'inputs/rigid500_nusc100_v1/manifest.json',evidence)
    objects=cohort['objects']; assert len(objects)==len({x['sample_id'] for x in objects})==500
    cachefile=out/'hash_cache.json'; cache=HashCache(read(cachefile).get('entries',{}) if cachefile.exists() else {},byte_budget)
    accepted=load_snapshot(root/'configs/source_reference_vehicle432_v1/acceptance_all432_20261111.json',evidence)
    from assemble_source_reference_acceptance import assemble
    revalidated=assemble(accepted['reference_manifest'],[r['path'] for r in accepted['reviews']])
    if revalidated['objects']!=accepted['objects']: raise ValueError('Reference acceptance changed')
    refs={x['sample_id']:x for x in accepted['objects']}
    inputs={}; multi={}
    for rel in ['inputs/vehicle432_image_inputs_v1/manifest.json','inputs/rigid111_image_inputs_v1/manifest.json']:
        inputs.update({x['sample_id']:x for x in load_snapshot(root/rel,evidence)['objects']})
    for rel in ['inputs/recon02_vehicle432_real_multiview_v2/manifest.json','inputs/recon02_rigid111_real_multiview_v1/manifest.json']:
        multi.update({x['sample_id']:x for x in load_snapshot(root/rel,evidence)['objects']})
    inventory=load_snapshot(root/'configs/asset_orientation_v1/inventory.json',evidence)
    old_ah={x['sample_id']:x for x in inventory['rows'] if x['method']=='ah'}
    jobs={'_prep':prep}
    for method in METHODS[1:6]:
        jobs[method]={}
        if method!='ah':
            path=prep/('runs/trellis2_vehicle432_parallel_v1/effective_batch_state.json' if method=='trellis2' else f'runs/{method}_vehicle432_v1/batch_state.json')
            jobs[method].update({x['sample_id']:x for x in load_snapshot(path,evidence)['jobs']})
        jobs[method].update({x['sample_id']:x for x in load_snapshot(prep/f'runs/{method}_rigid68_v1/batch_state.json',evidence)['jobs']})
    scene_jobs={}
    for name in ['scene_camera_vehicle_batch_v1','legacy_camera_batch_v1']:
        for j in load_snapshot(root/'state'/f'{name}.json',evidence)['jobs']:
            scene_jobs[j['sample_id'],j['method']]=j
    recovery=load_snapshot(root/'reports/splatad_degenerate_recovery_result.json',evidence)
    key=(recovery['sample_id'],recovery['method'])
    if recovery['technical_finite_render'] is True and scene_jobs[key]['status']=='failed':
        cache.check(recovery['validation'],recovery['validation_sha256'])
        scene_jobs[key]=dict(scene_jobs[key],status='completed',validation=recovery['validation'],validation_sha256=recovery['validation_sha256'],retained_degenerate=True)
    records=[]
    for obj in objects:
        sid=obj['sample_id']; entry=refs.get(sid)
        for method in METHODS:
            row=dict(sample_id=sid,uid=obj['uid'],scene=obj['scene'],category=obj['category'],method=method,
                     reference_status=entry['reference_review_decision'] if entry else 'nonvehicle_reference_review_pending',
                     reference_only_qualified=bool(entry and entry['reference_visual_qualified']),
                     formal_metrics_enabled=False,physical_registration_qualified=False,
                     heldout_qualified=False,surface_reference_qualified=False)
            try:
                if method in SCENE:
                    scene_intake(row,entry,scene_jobs.get((sid,method),{}),cache)
                else:
                    artifact,state=declared_artifact(obj,method,jobs,old_ah); row.update(artifact)
                    if 'asset' not in row:
                        row['preparation_status']='waiting_generation' if row['generation_status'] in ['running','queued'] else 'native_output_unavailable'
                    else:
                        row['actual_input_check']=check_actual_inputs(method,state,inputs[sid],multi.get(sid),cache)
                        cache.check(row['asset'],row['asset_sha256']); row['artifact_integrity_verified']=True
                        approval=root/'configs/asset_orientation_v1/approved'/f'{sid}__{method}.json'
                        if approval.exists():
                            decision=read(approval); contract=make_contract(row,decision,verify_asset=False)
                            contract.update(decision_path=str(approval),decision_sha256=cache.check(approval))
                            path=out/'orientation_contracts'/approval.name;save(path,contract)
                            row.update(semantic_heading_qualified=True,orientation_contract=str(path),
                                       orientation_contract_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                                       preparation_status='waiting_physical_registration')
                        else:
                            row.update(semantic_heading_qualified=False,preparation_status='waiting_semantic_heading' if obj['cohort_origin']=='previous_frozen432' else 'waiting_nonvehicle_axes_and_symmetry')
            except Deferred as error:
                row.update(preparation_status='queued_integrity_check',reason=str(error))
            except Exception as error:
                row.update(preparation_status='evidence_error',reason=repr(error))
            records.append(row)
    assert len(records)==4500 and len({(x['sample_id'],x['method']) for x in records})==4500
    now=datetime.now(timezone.utc).isoformat()
    report=dict(schema='rigid500_incremental_metric_preparation_v1',updated_at_utc=now,pid=os.getpid(),
                membership=500,methods=METHODS,reference_only_qualified_count=accepted['reference_visual_qualified_count'],
                nonvehicle_reference_reviewed_count=0,formal_metrics_enabled=False,assets_copied=0,
                hash_bytes_read=cache.bytes_read,hash_cache_hits=cache.hits,evidence=evidence,
                counts=dict(Counter(x['preparation_status'] for x in records)),
                by_method={m:dict(Counter(x['preparation_status'] for x in records if x['method']==m)) for m in METHODS},
                records=records)
    report['code_sha256']={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
        [Path(__file__),MODULES/'preparation_cache.py',MODULES/'asset_orientation_contract.py',
         MODULES/'source_render_gate.py',ROOT/'scripts/assemble_source_reference_acceptance.py']}
    save(cachefile,dict(entries=cache.entries))
    save(out/'manifest.json',report)
    summary={k:v for k,v in report.items() if k not in ['records','evidence']}
    save(out/'summary.json',summary)
    save(out/'worker_status.json',dict(status='pass_complete',pid=os.getpid(),updated_at_utc=now,
         manifest=str(out/'manifest.json'),hash_bytes_read=cache.bytes_read,counts=report['counts']))
    print(json.dumps(summary,ensure_ascii=False),flush=True)
    return summary


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--workspace',type=Path,default=ROOT)
    ap.add_argument('--output',type=Path);ap.add_argument('--watch',action='store_true')
    ap.add_argument('--interval',type=int,default=120);ap.add_argument('--hash-budget-mib',type=int,default=1024)
    args=ap.parse_args(); assert args.interval>=30 and args.hash_budget_mib>0
    out=args.output or args.workspace/'evaluation_nuscenes_v1/preparation_rigid500_v1'
    out.mkdir(parents=True,exist_ok=True)
    with (out/'worker.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while True:
            save(out/'worker_status.json',dict(status='preparing',pid=os.getpid(),started_at_utc=datetime.now(timezone.utc).isoformat()))
            try:run_once(args.workspace,out,args.hash_budget_mib*1024**2)
            except Exception as e:
                save(out/'worker_status.json',dict(status='pass_failed',pid=os.getpid(),reason=repr(e),updated_at_utc=datetime.now(timezone.utc).isoformat()))
                if not args.watch:raise
                print(repr(e),flush=True)
            if not args.watch:break
            time.sleep(args.interval)


if __name__=='__main__':main()
