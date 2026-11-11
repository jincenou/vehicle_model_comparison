"""Freeze available already-used real views within the existing scene training split."""
from collections import Counter
import datetime
import hashlib
import json
from pathlib import Path
from PIL import Image

R=Path(__file__).resolve().parents[1]
S=R/'inputs/benchmark_nuscenes100_v1/selection.json'
L=Path(('/opt/rigid-benchmark/project/workplace1/asset_library/manifests/asset_library_rigid.json'.replace('/opt/rigid-benchmark/project', __import__("os").environ.get('RIGID_PROJECT_ROOT', '/opt/rigid-benchmark/project'))))
O=R/'inputs/recon02_vehicle432_real_multiview_v2'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

selection=json.loads(S.read_text());library=json.loads(L.read_text())['assets']
single=json.loads((R/'inputs/vehicle432_image_inputs_v1/manifest.json').read_text())['objects']
byid={}
oldids={x['sample_id']for x in selection['existing_vehicles']}
for x in library:
    sid=x.get('provenance',{}).get('source_obj_id')
    if x.get('dataset')=='nuscenes'and sid in oldids:
        assert sid not in byid,'Ambiguous published source object'
        byid[sid]=x
for x in selection['new_vehicles']:byid[x['sample_id']]=x
assert len(byid)==len(single)==432
split_cache={};split_records={}
for scene in selection['scenes']:
    path=Path(scene['split']);assert sha(path)==scene['split_sha256']
    split=json.loads(path.read_text())
    split_cache[scene['scene']]={x['filename']:x for x in split['files']}
    split_records[scene['scene']]={'path':str(path),'sha256':scene['split_sha256']}
objects=[]
for row in single:
    sid=row['sample_id'];published=byid[sid];prov=published['provenance']
    identity=prov.get('production_identity',{})
    # Legacy manually accepted objects can lack a native nuScenes instance mapping.
    # Preserve that limitation; published real-view provenance is not native-ID proof.
    assert prov.get('source_obj_id',sid)==sid,sid
    views=[];excluded=[];seen=set()
    for v in prov.get('real_images',[]):
        raw=Path(v['source_image']);relative=str(raw.relative_to(('/data/nuScenes'.replace('/data', __import__("os").environ.get('RIGID_DATA_ROOT', '/data')))))
        split=split_cache[row['scene']].get(relative)
        if not split or split['split']!='train':
            excluded.append({'path':str(raw),'reason':'not_in_existing_training_split'});continue
        if relative in seen:
            excluded.append({'path':str(raw),'reason':'duplicate_raw_view'});continue
        image=Path(v['rgba_path']);expected=v.get('rgba_sha256',v.get('sha256'))
        try:
            assert sha(image)==expected,'RGBA hash mismatch'
            expected_raw=v.get('quality_review',{}).get('source_sha256')
            actual=sha(raw)
            if expected_raw:assert actual==expected_raw,'Raw image hash mismatch'
            with Image.open(image)as im:
                assert im.mode=='RGBA'and im.getchannel('A').getbbox(),'Invalid alpha'
                size=im.size
        except (AssertionError,OSError)as error:
            excluded.append({'path':str(image),'reason':str(error)});continue
        seen.add(relative)
        views.append({'image':str(image),'sha256':expected,'size':size,'raw_image':str(raw),
                      'raw_sha256':actual,'sample_data_token':split.get('sample_data_token',split.get('token')),
                      'timestamp_us':split.get('timestamp',split.get('timestamp_us')),'camera':split['channel'],'split':'train',
                      'crop_box':v.get('crop_box'),'original_real_view_eligible':v.get('real_view_eligible'),
                      'identity_evidence':'Already-accepted real_images of the frozen published asset; native-ID mapping status recorded separately'})
    if not views and row.get('sample_data_token'):
        # A FLUX-only production route may still retain the real source crop used
        # by all single-image baselines. Reuse that independently frozen input.
        matches=[v for v in split_cache[row['scene']].values()
                 if v.get('sample_data_token',v.get('token'))==row['sample_data_token']]
        assert len(matches)==1,sid
        split=matches[0]
        if split['split']=='train':
            image=Path(row['image_rgba']);assert sha(image)==row['image_rgba_sha256']
            raw=Path(('/data/nuScenes'.replace('/data', __import__("os").environ.get('RIGID_DATA_ROOT', '/data'))))/split['filename']
            with Image.open(image)as im:
                assert im.mode=='RGBA'and im.getchannel('A').getbbox()
                size=im.size
            views.append({'image':str(image),'sha256':row['image_rgba_sha256'],'size':size,
                'raw_image':str(raw),'raw_sha256':sha(raw),'sample_data_token':row['sample_data_token'],
                'timestamp_us':split.get('timestamp',split.get('timestamp_us')),'camera':split['channel'],
                'split':'train','identity_evidence':'Frozen single-image input with native instance and sample_data token',
                'fallback_reason':'Production route has no accepted real_images; use existing frozen real source, without padding',
                'H_canvas_from_raw_pixel_centres':row.get('H_canvas_from_raw_pixel_centres')})
    assert all(isinstance(v['timestamp_us'],int) for v in views),sid
    status='ready_multi_view'if len(views)>=2 else'single_view_requires_native_pilot'if views else'missing_eligible_real_view'
    objects.append({'sample_id':sid,'uid':row['uid'],'scene':row['scene'],'category':row['category'],
                    'native_identity':identity,'native_instance_identity_verified':identity.get('status')=='matched',
                    'status':status,'views':views,'real_input_count':len(views),
                    'generated_input_count':0,'excluded_views':excluded,'original_input_mode':prov.get('input_mode'),
                    'original_real_count':len(prov.get('real_images',[])),'source_asset_raw_sha256':prov['raw_sha256'],
                    'source_image_model_anchor':row['image_rgba'],'real_heldout_eligible':False})
out={'schema':1,'status':'frozen_real_view_inventory_pending_runtime','created_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),
     'selection_sha256':sha(S),'library_sha256':sha(L),'script_sha256':sha(__file__),'objects':objects,'scene_splits':split_records,
     'counts':dict(Counter(x['status']for x in objects)),'view_count_histogram':dict(Counter(x['real_input_count']for x in objects)),
     'policy':'Use original accepted real views in original order, filtered to existing training split; deduplicate raw frames. When none exist, use the already-frozen real single-image input only if its sample_data token is in the train split. No FLUX views, no padding, no result-driven selection. Preserve all 432 objects.',
     'extra_supervision':'ReconViaGen receives true multiview RGB/alpha; poses and LiDAR are not passed to image model; per-view identity is inherited from published asset provenance.'}
O.mkdir(exist_ok=False)
(O/'manifest.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'objects':len(objects),'counts':out['counts'],'views':out['view_count_histogram']}))
