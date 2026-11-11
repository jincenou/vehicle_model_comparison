"""Build hash-bound object frames/common orbit rigs without reading model outputs."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R/'evaluation_nuscenes_v1/scripts'))
from reference_camera_rig import make_rig, check_pose, corners, project
from validate_nuscenes_candidate_cameras import pose_at, transform, projection

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def save(p, d):
    p.write_text(json.dumps(d, indent=2, allow_nan=False)+'\n')

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, required=True)
    args=ap.parse_args()
    args.output=args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'objects').mkdir()
    manifest=R/'configs/common_camera_vehicle432_v3/manifest.json'
    source=json.loads(manifest.read_text()); cache={}; results=[]
    for row in source['objects']:
        sid, scene = row['sample_id'], row['scene']
        camera_evidence=row['object']['camera']
        identity=row.get('identity_status', camera_evidence.get('identity_check'))
        result=dict(sample_id=sid, scene=scene, category=row['category'],
                    native_instance_token=row['native_instance_token'], identity_status=identity,
                    status='deferred_identity', formal_metrics_qualified=False)
        if identity!='recorded_identity_supported_at_source':
            results.append(result); continue
        try:
            cp=Path(row['contract']); assert sha(cp)==row['contract_sha256'], 'camera contract changed'
            c=json.loads(cp.read_text()); token=row['native_instance_token']
            assert c['native_instance_token']==token
            if scene not in cache:
                roots=[R/'inputs/nuscenes_reviewed288'/scene/'v1.0-trainval',
                       R/'inputs/nuscenes'/scene/'v1.0-trainval',
                       R/'inputs/nuscenes100_candidates_20261109_2055/scenes'/scene,
                       R/'inputs/benchmark_nuscenes100_v1/data'/scene/'v1.0-trainval']
                root=next(p for p in roots if (p/'sample_annotation.json').exists())
                names=['sample','sample_annotation','sample_data','ego_pose','calibrated_sensor']
                tables={n:json.loads((root/(n+'.json')).read_text()) for n in names}
                indexes={n:{v['token']:v for v in tables[n]} for n in names}
                tracks=defaultdict(list)
                for a in tables['sample_annotation']:
                    tracks[a['instance_token']].append((indexes['sample'][a['sample_token']]['timestamp'],a))
                for track in tracks.values(): track.sort(key=lambda x:x[0])
                cache[scene]=(indexes,tracks,dict(root=str(root),table_sha256={n:sha(root/(n+'.json')) for n in names}))
            tables,tracks,evidence=cache[scene]
            sd=tables['sample_data'][c['sample_data_token']]
            assert sd['timestamp']==c['timestamp_us'], 'timestamp mismatch'
            assert Path(c['source_rgb'])==Path(('/data/nuScenes'.replace('/data', __import__("os").environ.get('RIGID_DATA_ROOT', '/data'))))/sd['filename'], 'source mismatch'
            assert sha(c['source_rgb'])==c['source_rgb_sha256'], 'source RGB changed'
            ep=tables['ego_pose'][sd['ego_pose_token']]; cs=tables['calibrated_sensor'][sd['calibrated_sensor_token']]
            Twc=transform(ep['rotation'],ep['translation'])@transform(cs['rotation'],cs['translation'])
            K=np.array(cs['camera_intrinsic']); assert np.allclose(K,c['K_raw'],rtol=0,atol=1e-8)
            assert np.allclose(Twc,c['T_world_from_camera_opencv'],rtol=0,atol=1e-8)
            track=tracks[token]; t=sd['timestamp']
            assert track[0][0]-100000<=t<=track[-1][0]+100000, 'source outside annotated lifetime'
            Two,wlh,dt,rule=pose_at(track,t); check_pose(Two)
            lwh=[wlh[1],wlh[0],wlh[2]]; Tco=np.linalg.inv(Twc)@Two
            bbox=projection(Tco,lwh,K,*c['render_wh']); assert bbox is not None, 'reference box not visible'
            previous=next(v for v in camera_evidence['projection_candidates'] if v['instance_token']==token)
            delta=float(np.max(np.abs(np.array(bbox)-np.array(previous['projected_bbox']))))
            assert delta<.05, 'reference pose disagrees with frozen identity projection'
            j=int(np.searchsorted([v[0] for v in track],t))
            used=[track[0 if j==0 else -1]] if j in [0,len(track)] else track[j-1:j+1]
            bracket=[dict(timestamp_us=tm,annotation_token=a['token'],sample_token=a['sample_token'],
                          translation=a['translation'],rotation_wxyz=a['rotation'],size_wlh=a['size']) for tm,a in used]
            pts=corners(lwh); uv,z=project(pts,Tco,K)
            world_pts=pts@Two[:3,:3].T+Two[:3,3]
            uv2,z2=project(world_pts,np.linalg.inv(Twc),K)
            pixel_error=float(np.max(np.abs(uv-uv2)))
            assert pixel_error<1e-6 and np.max(np.abs(z-z2))<1e-8
            rig=make_rig(lwh,Two)
            frame=dict(schema='nuscenes_reference_object_frame_v1', sample_id=sid,
                       scene=scene,native_instance_token=token,identity_status=identity,
                       camera_contract=str(cp),camera_contract_sha256=sha(cp),
                       metadata=evidence,annotation_bracket=bracket,timestamp_us=t,
                       pose_rule=rule,endpoint_offset_us=dt,
                       timestamp_rule_note='Interpolate annotation track at exact sensor time (linear translation/SLERP); not SDK keyframe shortcut. No motion GT claim.',
                       coordinates=dict(x='front / annotation length',y='left / annotation width',z='up / annotation height',units='metres'),
                       size_length_width_height_m=lwh,T_world_from_object=Two.tolist(),
                       T_source_camera_from_object=Tco.tolist(),source_K=K.tolist(),
                       source_render_wh=c['render_wh'],source_rgb=c['source_rgb'],
                       source_rgb_sha256=c['source_rgb_sha256'],
                       source_projection_bbox_clipped=bbox,
                       source_box_corners_cross_image_boundary=bool((uv[:,0]<-.5).any()or(uv[:,0]>=c['render_wh'][0]-.5).any()or(uv[:,1]<-.5).any()or(uv[:,1]>=c['render_wh'][1]-.5).any()or(z<=0).any()),
                       audit=dict(frozen_projection_max_error_px=delta,coordinate_projection_max_error_px=pixel_error),
                       orbit=rig,formal_metrics_qualified=False,asset_registration_certified=False,
                       reference_visibility_qualification='pending; box boundary flag is not a visible-mask occlusion label',
                       limitations=['Annotation boxes are not complete surface GT.',
                                    'All predicted assets still require independently validated registration.',
                                    'Reference box size/pose is additional conditioning, not recovered absolute scale.',
                                    'Scene models retain their estimated actor transforms; do not snap predictions to annotations.'])
            out=args.output/'objects'/f'{sid}.json';save(out,frame)
            result.update(status='reference_frame_prepared',contract=str(out),contract_sha256=sha(out),
                          source_box_corners_cross_image_boundary=frame['source_box_corners_cross_image_boundary'])
        except Exception as e:
            result.update(status='reference_frame_failed',reason=repr(e))
        results.append(result)
    output=dict(created_at_utc=datetime.now(timezone.utc).isoformat(),schema='reference_object_frame_inventory_v1',
                camera_manifest=str(manifest),camera_manifest_sha256=sha(manifest),
                builder_sha256=sha(__file__),rig_implementation_sha256=sha(R/'evaluation_nuscenes_v1/scripts/reference_camera_rig.py'),
                pose_implementation_sha256=sha(R/'scripts/validate_nuscenes_candidate_cameras.py'),
                counts=dict(Counter(x['status'] for x in results)),objects=results,
                scope='Reference geometry and common cameras only; all 432 planned objects retained; no formal scoring enabled')
    save(args.output/'manifest.json',output); print(json.dumps(output['counts']))

if __name__=='__main__':main()
