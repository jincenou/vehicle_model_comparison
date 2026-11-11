"""Recompute native scene unit/camera checks; not common asset registration."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

R=Path(__file__).resolve().parents[1]
MODULES=R/'evaluation_nuscenes_v1/scripts'
if not MODULES.exists():MODULES=R.parent/'evaluation'
sys.path.insert(0,str(MODULES))
from asset_orientation_contract import checked_rigid


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    manifest=R/'evaluation_nuscenes_v1/preparation_rigid500_v1/manifest.json'
    raw=manifest.read_bytes();data=json.loads(raw);rows=[]
    for row in data['records']:
        if row['preparation_status']!='source_camera_intake_ready':continue
        result=dict(sample_id=row['sample_id'],method=row['method'],formal_metrics_qualified=False)
        try:
            vp=Path(row['validation']);assert sha(vp)==row['validation_sha256']
            v=json.loads(vp.read_text());audit=v['common_camera_audit'];cp=Path(audit['contract'])
            assert sha(cp)==audit['contract_sha256'];camera=json.loads(cp.read_text())
            assert camera['sample_id']==row['sample_id'] and camera['world_units']=='metres'
            transform=checked_rigid(audit['T_model_from_world'])
            source=checked_rigid(camera['T_world_from_camera_opencv'])
            actual=np.asarray(v['source_camera_to_world'] if row['method']=='splatad' else v['camera_to_world'])
            if row['method']=='splatad':
                # SplatAD stores the Nerfstudio 3x4 OpenGL camera; the native
                # adapter explicitly converts it to OpenCV for projection.
                assert actual.shape==(3,4)
                actual=np.vstack([actual,[0,0,0,1]])@np.diag([1,-1,-1,1])
            assert actual.shape==(4,4)
            intrinsics=np.asarray(v['source_camera_K'] if row['method']=='splatad' else v['camera_K'])
            k=np.asarray(camera['K_raw']);width,height=camera['render_wh']
            pose_error=float(np.abs(actual-transform@source).max())
            pixels=np.array([[x,y,1.] for x in [0.,width/2,width-1.] for y in [0.,height/2,height-1.]])
            points=np.concatenate([(np.linalg.inv(k)@pixels.T).T*z for z in [2.,10.,50.]])
            world=np.c_[points,np.ones(len(points))]@source.T
            modeled=world@transform.T
            projected=(modeled@np.linalg.inv(actual).T)[:,:3]@intrinsics.T
            error=float(np.abs(projected[:,:2]/projected[:,2:]-np.tile(pixels[:,:2],(3,1))).max())
            assert np.isfinite([pose_error,error]).all() and pose_error<.001 and error<.05
            result.update(status='native_units_and_continuous_projection_verified',
                          validation_sha256=sha(vp),camera_contract_sha256=sha(cp),
                          world_to_model_is_unit_preserving_rigid=True,
                          pose_error=pose_error,continuous_projection_error_px=error,
                          pixel_sampling_origin_qualified=False,
                          canonical_asset_extent_and_centre_qualified=False)
        except Exception as error:
            result.update(status='audit_failed',reason=repr(error))
        rows.append(result)
    report=dict(schema='prepared_scene_spatial_audit_v1',preparation_snapshot_sha256=hashlib.sha256(raw).hexdigest(),
                snapshot_time=data['updated_at_utc'],audited_records=len(rows),
                counts=dict(Counter(x['status'] for x in rows)),by_method=dict(Counter(x['method'] for x in rows)),
                records=rows,formal_metrics_enabled=False,
                limitation='Continuous camera projection and metre units do not certify pixel sampling or common asset extent/centre conditioning. Native scene reconstruction and length-conditioned canonical asset comparisons require explicit separate labels.')
    out=R/'reports/prepared_scene_spatial_audit_20261111.json'
    out.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:report[k] for k in ['audited_records','counts','by_method','formal_metrics_enabled']}))


if __name__=='__main__':main()
