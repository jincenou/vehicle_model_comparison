"""Freeze all generated vehicle assets for semantic-heading audit, never guess front."""
from pathlib import Path
from collections import Counter
import json,hashlib,datetime
R=Path(__file__).resolve().parents[1];P=R/'model_preparation/20261109_2100_CST';O=R/'configs/asset_orientation_v1'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
selection=read(R/'inputs/benchmark_nuscenes100_v1/selection.json');inputs=read(R/'inputs/vehicle432_image_inputs_v1/manifest.json')['objects']
libpath=Path(('/opt/rigid-benchmark/project/workplace1/asset_library/manifests/asset_library_rigid.json'.replace('/opt/rigid-benchmark/project', __import__("os").environ.get('RIGID_PROJECT_ROOT', '/opt/rigid-benchmark/project'))));lib=read(libpath)['assets'];libroot=libpath.parents[1]
bycanonical={str((libroot/x['canonical']['canonical_path']).resolve()):x for x in lib if x.get('canonical',{}).get('canonical_path')}
old={x['sample_id']:x for n in [53,235]for x in read(R/f'configs/metric{n}_registry.json')['samples']if x['dataset']=='nuscenes'}
new={x['sample_id']:x for x in selection['new_vehicles']}
states={m:read(P/f'runs/{m}_vehicle432_v1/batch_state.json')for m in ['trellis_original','trellis2','hunyuan21','recon02']}
jobs={m:{j['sample_id']:j for j in s['jobs']}for m,s in states.items()}
I=[[1,0,0],[0,1,0],[0,0,1]];ahup=[[0,0,-1],[-1,0,0],[0,1,0]];trellis_ply_up=[[1,0,0],[0,0,1],[0,-1,0]]
rows=[]
for x in inputs:
 sid=x['sample_id']
 for m in ['ours','ah','trellis_original','trellis2','hunyuan21','recon02']:
  row=dict(sample_id=sid,method=m,category=x['category'],source_rgba=x['image_rgba'],source_rgba_sha256=x['image_rgba_sha256'],status='pending_source_output',heading_status='unverified',up_rotation=I,representation='gaussian_ply'if m in ['ours','ah','trellis_original','recon02']else'glb_mesh',physical_camera_qualified=False)
  if m=='ours':
   asset=old[sid]['assets']['ours']if sid in old else {'ply':new[sid]['asset']['path'],'sha256':new[sid]['asset']['sha256']}
   path=Path(asset['ply']);metadata=bycanonical.get(str(path.resolve()));assert metadata
   row.update(asset=str(path),asset_sha256=asset['sha256'],status='ready_for_orientation_audit',canonical_metadata=metadata['canonical'],heading_evidence=str(libpath))
   c=metadata['canonical']
   if c.get('heading_verified')is True and c.get('heading')=='+X'and c.get('up')=='+Z'and c.get('sha256')==asset['sha256']:
    row.update(heading_status='published_heading_verified',native_to_semantic=I)
  elif m=='ah':
   row['up_rotation']=ahup
   if sid in old:asset=old[sid]['assets']['ah'];row.update(asset=asset['ply'],asset_sha256=asset['sha256'],status='ready_for_orientation_audit')
   else:
    tag='nuscenes100_new233_pilot_20261110'if sid=='scene-0002_t000013'else'nuscenes100_new233_batch_20261110'
    jp=R/f'runs/ah/{tag}_{sid}/job.json';j=read(jp);v=read(j['reload_validation']);assert v['status']=='passed'
    row.update(asset=j['output_ply'],asset_sha256=v['ply_sha256'],status='ready_for_orientation_audit',reload_evidence=j['reload_validation'])
  else:
   j=jobs[m][sid];row['generation_status']=j['status']
   if j['status']=='completed':
    folder=Path(j['run_dir']);s=read(folder/'state.json')
    path=folder/'native_gaussians.ply'if row['representation']=='gaussian_ply'else Path(s['output'])
    expected=s['output_hashes'][path.name]if row['representation']=='gaussian_ply'else s['output_sha256']
    row.update(asset=str(path),asset_sha256=expected,status='ready_for_orientation_audit',generation_evidence=str(folder/'state.json'))
   elif j['status']=='failed':row.update(status='generation_failed',reason=j.get('reason'))
   if m=='trellis_original':row['up_rotation']=trellis_ply_up
  rows.append(row)
assert len(rows)==432*6
result=dict(created_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),selection_sha256=sha(R/'inputs/benchmark_nuscenes100_v1/selection.json'),scope='all 432 vehicles x six asset-generation methods; scene-method track-frame registration is separate',target_axes={'front':'+X','left':'+Y','up':'+Z'},raw_assets_immutable=True,physical_pose_registration=False,metric_score_optimization=False,display=dict(resolution=512,vertical_fov_degrees=40,elevation_degrees=12,distance_over_length=1.8,white_background=True,semantic_azimuths=[45,90,135],heading_review_cardinals=[0,90,180,270]),rows=rows,counts=dict(Counter(x['status']for x in rows)),heading_counts=dict(Counter(x['heading_status']for x in rows)),scene_methods=['splatad','omnire','streetgs'],scene_method_policy='Use verified native instance pose at frozen time to recover front/up; never PCA world-positioned actors as substitutes for pose.',evidence=dict(library=str(libpath),library_sha256=sha(libpath),original_ply_axis_code=str(P/'sources/trellis_original_legacy_20261110/trellis/representations/gaussian/gaussian_model.py'),recon_ply_axis_code=str(P/'sources/recon02-f6720923260927333e22761fa722144d214733d9/trellis/representations/gaussian/gaussian_model.py'),legacy_review_code=('/opt/rigid-benchmark/project/workplace1/reconviagen_demo/scripts/finalize_heading.py'.replace('/opt/rigid-benchmark/project', __import__("os").environ.get('RIGID_PROJECT_ROOT', '/opt/rigid-benchmark/project')))))
path=O/'inventory.json';assert not path.exists();path.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(result['counts'],result['heading_counts'])
