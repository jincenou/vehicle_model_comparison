"""Audit candidate camera metadata and annotation identity without fitting assets."""
import argparse
from collections import Counter,defaultdict
import hashlib,itertools,json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from scipy.spatial.transform import Rotation,Slerp

RAW=Path(('/data/nuScenes'.replace('/data', __import__("os").environ.get('RIGID_DATA_ROOT', '/data'))))
LIB=Path(('/opt/rigid-benchmark/project/workplace1/asset_library'.replace('/opt/rigid-benchmark/project', __import__("os").environ.get('RIGID_PROJECT_ROOT', '/opt/rigid-benchmark/project'))))

def transform(q,t):
    T=np.eye(4);T[:3,:3]=Rotation.from_quat(np.asarray(q)[[1,2,3,0]]).as_matrix();T[:3,3]=t
    return T

def projection(T,size,K,width,height):
    signs=np.array(list(itertools.product([-1,1],repeat=3)))
    points=(signs*np.asarray(size)/2)@T[:3,:3].T+T[:3,3]
    visible=[p for p in points if p[2]>=.1]
    for i,j in itertools.combinations(range(8),2):
        if np.count_nonzero(signs[i]!=signs[j])!=1:continue
        a,b=points[i],points[j]
        if (a[2]<.1)!=(b[2]<.1):visible.append(a+(.1-a[2])/(b[2]-a[2])*(b-a))
    if not visible:return None
    p=np.asarray(visible)@K.T;uv=p[:,:2]/p[:,2:3]
    lo=np.maximum(uv.min(0),0);hi=np.minimum(uv.max(0),[width,height])
    return [*lo,*hi] if (hi>lo).all() else None

def iou(a,b):
    lo=np.maximum(a[:2],b[:2]);hi=np.minimum(a[2:],b[2:]);inter=float(np.maximum(hi-lo,0).prod())
    area=lambda x:max(0,x[2]-x[0])*max(0,x[3]-x[1])
    return inter/(area(a)+area(b)-inter) if area(a)+area(b)>inter else 0

def pose_at(track,t):
    times=np.array([x[0] for x in track],dtype=np.int64);j=int(np.searchsorted(times,t))
    if j==0 or j==len(track):
        # nuScenes get_boxes uses the endpoint annotation outside the bracketing interval.
        tm,a=track[0 if j==0 else -1]
        return transform(a['rotation'],a['translation']),a['size'],int(t-tm),'endpoint'
    ta,a=track[j-1];tb,b=track[j];f=(t-ta)/(tb-ta)
    qs=np.array([a['rotation'],b['rotation']])[:,[1,2,3,0]]
    T=np.eye(4);T[:3,:3]=Slerp([0,1],Rotation.from_quat(qs))([f]).as_matrix()[0]
    T[:3,3]=(1-f)*np.array(a['translation'])+f*np.array(b['translation'])
    return T,b['size'],0,'linear_translation_slerp'

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--audit',required=True,type=Path);args=ap.parse_args()
    out=args.audit/'camera_identity';out.mkdir(exist_ok=False)
    assets=json.loads((args.audit/'candidate_assets.json').read_text())
    anchors={x['uid']:x for x in json.loads((args.audit/'anchor_recovery/anchors.json').read_text())}
    library={x['uid']:x for x in json.loads((LIB/'manifests/asset_library_rigid.json').read_text())['assets']}
    inst={x['token']:x['category_token'] for x in json.loads((RAW/'v1.0-trainval/instance.json').read_text())}
    cats={x['token']:x['name'] for x in json.loads((RAW/'v1.0-trainval/category.json').read_text())}
    groups=defaultdict(list)
    for a in assets:groups[a['scene']].append(a)
    result=[]
    for scene,group in sorted(groups.items()):
        root=args.audit/'scenes'/scene
        load=lambda n:json.loads((root/(n+'.json')).read_text())
        sd={str(RAW/x['filename']):x for x in load('sample_data')}
        poses={x['token']:x for x in load('ego_pose')};cal={x['token']:x for x in load('calibrated_sensor')}
        samples={x['token']:x for x in load('sample')};tracks=defaultdict(list)
        for a in load('sample_annotation'):
            if cats[inst[a['instance_token']]].startswith('vehicle.'):
                tracks[a['instance_token']].append((samples[a['sample_token']]['timestamp'],a))
        for t in tracks.values():t.sort(key=lambda x:x[0])
        for asset in group:
            entry=library[asset['uid']];s=entry['provenance']['source'];anchor=anchors[asset['uid']]
            d=sd[s['source_image']];p=poses[d['ego_pose_token']];c=cal[d['calibrated_sensor_token']]
            Twc=transform(p['rotation'],p['translation'])@transform(c['rotation'],c['translation'])
            K=np.asarray(c['camera_intrinsic']);H=np.array(anchor['H_stored_from_raw'])
            # PIL maps integer pixel centres as (x+0.5)*scale-0.5 after cropping.
            H[0,2]+=(H[0,0]-1)/2;H[1,2]+=(H[1,1]-1)/2
            row={'uid':asset['uid'],'scene':scene,'recorded_instance':asset['instance_token'],
                 'sample_data_token':d['token'],'timestamp_us':d['timestamp'],
                 'source_K_max_error':float(np.abs(K-np.array(s['camera_intrinsic'])).max()),
                 'source_T_world_from_camera_max_error':float(np.abs(Twc-np.array(s['camera_to_world'])).max()),
                 'K_raw':K.tolist(),'T_world_from_camera':Twc.tolist(),
                 'H_stored_from_raw_pixel_centres':H.tolist(),'K_stored':(H@K).tolist(),
                 'mask_resolution':anchor['mask_resolution'],'projection_candidates':[]}
            bbox=s['bbox_xyxy'];rank=[]
            for token,track in tracks.items():
                # Do not project objects outside their annotated lifetime by more than one sensor offset.
                if d['timestamp']<track[0][0]-100000 or d['timestamp']>track[-1][0]+100000:continue
                Two,size,dt,interp=pose_at(track,d['timestamp'])
                b=projection(np.linalg.inv(Twc)@Two,[size[1],size[0],size[2]],K,d['width'],d['height'])
                if b:
                    rank.append({'instance_token':token,'iou':iou(bbox,b),'projected_bbox':b,
                                 'nuscenes_category':cats[inst[token]],'pose_rule':interp,'endpoint_offset_us':dt})
            rank.sort(key=lambda x:x['iou'],reverse=True);row['projection_candidates']=rank[:3]
            best=rank[0] if rank else None;margin=best['iou']-(rank[1]['iou'] if len(rank)>1 else 0) if best else 0
            row['top_margin']=margin
            confident=bool(best and best['iou']>=.55 and margin>=.15)
            row['source_projection_confident']=confident
            if not confident:row['identity_check']='source_projection_ambiguous'
            elif not asset['instance_token']:row['identity_check']='new_source_match_requires_multiview_or_visual_review'
            elif best['instance_token']==asset['instance_token']:row['identity_check']='recorded_identity_supported_at_source'
            else:row['identity_check']='recorded_identity_conflicts_with_source_projection'
            # Pure coordinate round-trip check; does not validate an asset registration.
            points=np.array([[0,0,1],[d['width']-1,d['height']-1,1],[K[0,2],K[1,2],1]])
            roundtrip=(points@H.T)@np.linalg.inv(H).T
            row['pixel_transform_roundtrip_error']=float(np.max(np.abs(roundtrip-points)))
            result.append(row)
        print(scene,'camera/identity checked',len(group),flush=True)
    (out/'objects.json').write_text(json.dumps(result,indent=2))
    summary={'status':'completed_not_identity_freeze','objects':len(result),
             'identity_checks':dict(Counter(x['identity_check'] for x in result)),
             'max_intrinsic_error':max(x['source_K_max_error'] for x in result),
             'max_camera_pose_error':max(x['source_T_world_from_camera_max_error'] for x in result),
             'max_pixel_roundtrip_error':max(x['pixel_transform_roundtrip_error'] for x in result),
             'thresholds':{'box_iou':.55,'runner_up_margin':.15},
             'threshold_origin':'existing production identity rule; no model-output scores used',
             'asset_alignment_verified':False,'resized_alpha_semantics_verified':False}
    (out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary))
    # Diagnostic sheets show every candidate; native identity disagreements remain visible.
    for start in range(0,len(result),48):
        batch=result[start:start+48];sheet=Image.new('RGB',(1600,8*160),'white');draw=ImageDraw.Draw(sheet)
        for i,row in enumerate(batch):
            a=anchors[row['uid']];x=(i%6)*266;y=(i//6)*160
            with Image.open(a['anchor_rgba']) as im:
                white=Image.new('RGBA',im.size,'white');white.alpha_composite(im.convert('RGBA'))
                panel=white.convert('RGB');panel.thumbnail((252,112));sheet.paste(panel,(x+(252-panel.width)//2,y+22))
            label=row['uid'].split(':')[-1];draw.text((x+3,y+2),label,fill='black')
            best=row['projection_candidates'][0] if row['projection_candidates'] else None
            tag=row['identity_check'].replace('recorded_identity_','').replace('source_projection_','')
            draw.text((x+3,y+137),tag[:36],fill='red' if 'conflict' in tag else 'black')
        sheet.save(out/f'anchors_{start//48:02d}.jpg',quality=92)

if __name__=='__main__':main()
