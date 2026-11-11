"""Audit shared spatial-calibration math against existing six-method pilot artifacts.

No renderer execution, asset edits, score fitting or formal qualification.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
MODULES=ROOT/'evaluation_nuscenes_v1/scripts'
if not MODULES.exists():MODULES=ROOT.parent/'evaluation'
sys.path.insert(0,str(MODULES))
from spatial_calibration import length_conditioned,native_camera


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--sample-id',default='scene-0064_t000637')
    parser.add_argument('--workspace',type=Path,default=ROOT);args=parser.parse_args();root=args.workspace;sid=args.sample_id
    rp=root/'configs/reference_object_frames_v1/objects'/f'{sid}.json';ref=json.loads(rp.read_text());rows=[]
    for method in ['ours','ah','trellis_original','trellis2','hunyuan21','recon02']:
        dp=root/'exports/asset_orientation_final_v1'/method/sid/'orientation_render.json';display=json.loads(dp.read_text())
        vp=root/'exports/asset_registration_pilot_v1'/sid/method/'validation.json';validation=json.loads(vp.read_text())
        assert sha(display['asset'])==display['asset_sha256']==validation['asset_sha256']
        assert sha(rp)==validation['reference_contract_sha256'] and sha(dp)==validation['display_contract_sha256']
        decision=json.loads(Path(display['heading_decision']).read_text())
        assert sha(display['heading_decision'])==display['heading_decision_sha256']
        assert decision['asset_sha256']==display['asset_sha256']
        rotation=np.asarray(display['candidate_native_to_axis']);assert np.allclose(rotation,decision['native_to_semantic'])
        center_input=rotation.T@np.asarray(display['center'])
        contract=length_conditioned(rotation,center_input,display['display_length'],ref['size_length_width_height_m'][0])
        transform=np.asarray(contract['T_reference_from_input'])
        error=float(np.abs(transform-np.asarray(validation['T_reference_object_from_input'])).max())
        assert error<1e-9
        # Image-space comparison must use one reference camera, not auto-fit a camera per asset.
        camera_errors=[]
        for view in validation['views']:
            if view['name']=='source':
                expected_camera=ref['T_source_camera_from_object'];expected_k=ref['source_K'];expected_wh=ref['source_render_wh']
            else:
                i=int(view['name'].split('_')[1]);orbit=next(x for x in ref['orbit']['views'] if x['index']==i)
                expected_camera=orbit['T_camera_from_object'];expected_k=ref['orbit']['K'];expected_wh=ref['orbit']['render_wh']
            assert view['wh']==expected_wh and np.allclose(view['K'],expected_k,rtol=0,atol=1e-10)
            assert np.allclose(view['T_camera_from_object'],expected_camera,rtol=0,atol=1e-10)
            if 'render_camera_in_native_units' in view:
                camera=native_camera(expected_camera,transform,view['near'],view['far'])
                camera_errors.append(float(np.abs(np.asarray(camera['T_camera_from_input'])-view['render_camera_in_native_units']).max()))
                assert camera_errors[-1]<1e-9
        rows.append(dict(method=method,asset=display['asset'],asset_sha256=display['asset_sha256'],
            original_display_length=display['display_length'],isotropic_scale=contract['scale_reference_units_per_input_unit'],
            conditioned_length_m=display['display_length']*contract['scale_reference_units_per_input_unit'],
            similarity_matrix_difference=error,native_camera_matrix_difference=max(camera_errors,default=None),
            shared_reference_cameras_checked=len(validation['views']),calibration_contract=contract,
            measurement_policy='legacy Gaussian-centre/mesh-vertex quantiles; representation-dependent, not qualified',
            source_reference_eligible=validation['source_reference_eligible'],formal_metrics_qualified=False))
    report=dict(schema='spatial_calibration_numerical_audit_v1',sample_id=sid,reference=str(rp),reference_sha256=sha(rp),
        target_length_m=ref['size_length_width_height_m'][0],methods=rows,raw_assets_modified=False,
        shared_similarity_math_validated=True,measurement_policy_validated=False,formal_metrics_enabled=False,
        next_required_gate='Validate a shared extent/centre measurement rule across native representations before formal batch registration')
    out=root/'reports'/f'spatial_calibration_pilot_{sid}_20261111.json';out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(report=str(out),target_length_m=report['target_length_m'],methods=len(rows),
                         original_lengths={r['method']:r['original_display_length'] for r in rows},formal_metrics_enabled=False)))


if __name__=='__main__':main()
