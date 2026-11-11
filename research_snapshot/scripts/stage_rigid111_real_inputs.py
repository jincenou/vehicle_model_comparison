"""Prepare real source adapters for the explicitly frozen nonvehicle extension."""
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import numpy as np
from PIL import Image
from stage_vehicle432_image_inputs import canvas, sha

R=Path(__file__).resolve().parents[1]
O=R/'inputs/rigid111_image_inputs_v1'

def main():
    assert not O.exists()
    cohort_path=R/'inputs/rigid543_nusc100_v1/manifest.json'
    cohort=json.loads(cohort_path.read_text())
    assert sha(cohort_path)==json.loads(cohort_path.with_name('freeze.json').read_text())['manifest_sha256']
    members=[r for r in cohort['objects'] if r['cohort_origin']=='user_approved_nonvehicle111']
    anchor_path=R/'inputs/rigid111_anchor_audit_v1/anchor_recovery/anchors.json'
    anchors={r['uid']:r for r in json.loads(anchor_path.read_text())}
    selection=json.loads((R/'inputs/benchmark_nuscenes100_v1/selection.json').read_text())
    scenes={r['scene']:r for r in selection['scenes']};splits={}
    for scene in {r['scene'] for r in members}:
        p=Path(scenes[scene]['split']);assert sha(p)==scenes[scene]['split_sha256']
        splits[scene]={v['filename']:v for v in json.loads(p.read_text())['files']}
    O.mkdir();rows=[]
    for member in members:
        a=anchors[member['uid']];assert not a['reasons']
        source=Path(a['anchor_rgba']);assert sha(source)==a['anchor_rgba_sha256']
        raw=Path(a['source_rgb']);assert sha(raw)==a['source_rgb_sha256']
        record=splits[member['scene']][str(raw.relative_to(('/data/nuScenes'.replace('/data', __import__("os").environ.get('RIGID_DATA_ROOT', '/data')))))]
        image,A,H,size=canvas(source);alpha=image.getchannel('A');assert alpha.getbbox()
        folder=O/'objects'/member['sample_id'];folder.mkdir(parents=True)
        rgba=folder/'image_rgba.png';image.save(rgba)
        rgb=folder/'frame.jpeg';mask=folder/'mask.png'
        white=Image.new('RGB',image.size,'white');white.paste(image,mask=alpha)
        white.save(rgb,quality=95,subsampling=0);alpha.save(mask)
        box=a['crop_box'];sx=size[0]/(box[2]-box[0]);sy=size[1]/(box[3]-box[1])
        Hstored=np.array([[sx,0,-sx*box[0]+(sx-1)/2],[0,sy,-sy*box[1]+(sy-1)/2],[0,0,1]])
        Hraw=H@Hstored
        points=np.array([[box[0],box[1],1],[box[2]-1,box[3]-1,1]],float)
        error=float(np.abs((points@Hraw.T)@np.linalg.inv(Hraw).T-points).max());assert error<1e-9
        rows.append(dict(uid=member['uid'],sample_id=member['sample_id'],review_id=member['review_id'],
            scene=member['scene'],category=member['category'],dataset='nuscenes',existing_reviewed_object=True,
            source_rgb=str(rgb),source_rgb_sha256=sha(rgb),source_mask=str(mask),source_mask_sha256=sha(mask),
            image_rgba=str(rgba),image_rgba_sha256=sha(rgba),retained_rgba=str(source),retained_rgba_sha256=sha(source),
            raw_source=str(raw),raw_source_sha256=sha(raw),raw_crop_box=box,retained_size=list(size),adapter_size=[512,512],
            pixel_transform_A_edge_coordinates=A.tolist(),H_canvas_from_retained_pixel_centres=H.tolist(),
            H_stored_from_raw_pixel_centres=Hstored.tolist(),H_canvas_from_raw_pixel_centres=Hraw.tolist(),
            pixel_transform_roundtrip_error=error,mask_origin=a['mask_origin'],anchor_recovery_status=a['status'],
            native_pixel_equality_verified=a.get('foreground_rgb_max_error')==0,
            real_input_count=1,generated_input_count=0,original_ours_input_mode=member['input_mode'],
            camera_given_to_image_model=False,real_heldout_eligible=False,asset_alignment_verified=False,
            source_view_camera_registration='pending',model_specific_preprocessing_validation='pending',
            scene_training_split=record['split'],sample_data_token=record.get('sample_data_token',record.get('token')),
            timestamp_us=record.get('timestamp_us',record.get('timestamp')),camera=record['channel'],
            input_scope='Recovered real anchor used by the pipeline frontend, not generated FLUX imagery. Scene split kept unchanged; source is never independent heldout.'))
    assert len(rows)==111 and len({r['uid'] for r in rows})==111
    manifest=dict(schema=1,status='frozen_source_adapter',created_at=datetime.now(timezone.utc).isoformat(),
        cohort_manifest=str(cohort_path),cohort_manifest_sha256=sha(cohort_path),anchor_audit_sha256=sha(anchor_path),
        script_sha256=sha(__file__),objects=rows,counts=dict(objects=111,real=111,generated=0),
        source_scene_split_counts=dict(Counter(r['scene_training_split'] for r in rows)),
        real_heldout_eligible=False,all_model_inputs_runtime_validated=False,
        policy='Freeze real frontend anchors and SAM alpha for same-object comparisons. Preserve scene training splits; any source in an old eval split is disclosed and is not a heldout reference. No regenerated asset or synthetic view is used as a baseline input.')
    (O/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(counts=manifest['counts'],split_counts=manifest['source_scene_split_counts'])))

if __name__=='__main__':main()
