"""Strict intake checks for native scene-actor source scores, not direct-asset registration."""
import math
import numpy as np


def validate_native_source(validation, camera, camera_hash, reference, method):
    if method not in ('splatad','omnire','streetgs'):
        raise ValueError('native scene adapter only')
    status=validation.get('status')
    if status not in ('passed','rendered_degenerate_output'):
        raise ValueError('no valid native render')
    if status=='rendered_degenerate_output'and validation.get('technical_finite_render')is not True:
        raise ValueError('degenerate render needs finite-render evidence')
    if validation.get('method',method)!=method:
        raise ValueError('method mismatch')
    if not validation['sample_id']==camera['sample_id']==reference['sample_id']:
        raise ValueError('sample mismatch')
    if validation['actor_uuid']!=reference['native_instance_token']or camera['native_instance_token']!=reference['native_instance_token']:
        raise ValueError('actor identity mismatch')
    if validation['frozen_raw_timestamp_us']!=camera['timestamp_us']or reference['timestamp_us']!=camera['timestamp_us']:
        raise ValueError('source timestamp mismatch')
    if validation['source_image']!=camera['source_rgb']or reference['raw_source']!=camera['source_rgb']:
        raise ValueError('source image mismatch')
    audit=validation['common_camera_audit']
    if audit['contract_sha256']!=camera_hash:
        raise ValueError('camera hash mismatch')
    if audit.get('camera_pose_refinement')is not False or audit.get('render_time_shift_seconds')!=0 or audit.get('rolling_shutter_seconds')!=0:
        raise ValueError('private camera or time correction enabled or unspecified')
    for key,limit in [('projection_max_error_px',.05),('pose_matrix_max_error',.001)]:
        x=audit[key]
        if not isinstance(x,(int,float))or not math.isfinite(x)or not 0<=x<limit:
            raise ValueError('camera numerical audit failed: '+key)
    if audit['render_wh']!=camera['render_wh']or validation['render_wh']!=camera['render_wh']:
        raise ValueError('render canvas mismatch')
    if not np.allclose(audit['K_raw'],camera['K_raw'],rtol=0,atol=1e-8)or not np.allclose(audit['T_world_from_camera_opencv'],camera['T_world_from_camera_opencv'],rtol=0,atol=1e-8):
        raise ValueError('camera audit references different K or pose')
    if reference['identity_status']!='recorded_identity_supported_at_source'or not reference['native_reference_pixels_verified']:
        raise ValueError('reference identity/native pixels not qualified')
    if method!='splatad'and not validation.get('identity_validation'):
        raise ValueError('native instance insertion-order evidence missing')
    return dict(native_source_identity_camera_checked=True,
                gaussian_count=validation['gaussians'],degenerate_render=status=='rendered_degenerate_output',
                scope='Frozen observed source state, native learned actor pose and appearance; no new image alignment or test-pose fitting')
