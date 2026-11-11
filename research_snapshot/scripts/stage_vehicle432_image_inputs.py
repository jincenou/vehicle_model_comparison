"""Stage frozen same-object images, preserving all 199 existing JPEG/mask bytes.

New 233 objects use the established 512-pixel aspect-preserving adapter. Keep
both edge-coordinate legacy matrices and correct pixel-centre homographies.
These are source-view inputs, never independent real heldout references.
"""
import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
from PIL import Image

R=Path(__file__).resolve().parents[1]
B=R/'inputs/benchmark_nuscenes100_v1'
OUT=R/'inputs/vehicle432_image_inputs_v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def canvas(source):
    with Image.open(source) as original:
        assert original.mode=='RGBA'
        w,h=original.size;scale=512/max(w,h);nw,nh=round(w*scale),round(h*scale)
        assert min(nw,nh)>0
        left,top=(512-nw)//2,(512-nh)//2
        image=Image.new('RGBA',(512,512),(255,255,255,0))
        image.paste(original.resize((nw,nh),Image.Resampling.LANCZOS),(left,top))
    A=np.array([[nw/w,0,left],[0,nh/h,top],[0,0,1]],dtype=float)
    H=A.copy();H[0,2]+=(H[0,0]-1)/2;H[1,2]+=(H[1,1]-1)/2
    return image,A,H,(w,h)
def main():
    assert not OUT.exists(),'Immutable derived input set already exists'
    frozen=read(B/'freeze.json');assert sha(B/'selection.json')==frozen['selection_sha256']
    selection=read(B/'selection.json');items=[]
    for row in selection['existing_vehicles']:
        assert sha(row['retained_rgba'])==row['retained_rgba_sha256']
        assert sha(row['source_rgb'])==row['source_rgb_sha256']
        assert sha(row['source_mask'])==row['source_mask_sha256']
        image,A,H,size=canvas(row['retained_rgba'])
        assert np.allclose(A,row['pixel_transform_A'],atol=1e-12,rtol=0),row['uid']
        with Image.open(row['source_mask']) as mask:
            assert np.array_equal(np.asarray(image.getchannel('A')),np.asarray(mask.convert('L'))),row['uid']
        items.append((row,True))
    for row in selection['new_vehicles']:
        a=row['anchor'];assert sha(a['anchor_rgba'])==a['anchor_rgba_sha256']
        items.append((row,False))
    assert len(items)==len({x['sample_id'] for x,_ in items})==432
    OUT.mkdir();records=[]
    for row,existing in items:
        source=row['retained_rgba'] if existing else row['anchor']['anchor_rgba']
        source_sha=row['retained_rgba_sha256'] if existing else row['anchor']['anchor_rgba_sha256']
        image,A,H,size=canvas(source);folder=OUT/'objects'/row['sample_id'];folder.mkdir(parents=True)
        rgba=folder/'image_rgba.png';image.save(rgba)
        if existing:
            rgb=Path(row['source_rgb']);mask=Path(row['source_mask'])
        else:
            rgb=folder/'frame.jpeg';mask=folder/'mask.png'
            alpha=image.getchannel('A');assert alpha.getbbox() is not None
            white=Image.new('RGB',(512,512),'white');white.paste(image,mask=alpha)
            white.save(rgb,quality=95,subsampling=0);alpha.save(mask)
        record={'uid':row['uid'],'sample_id':row['sample_id'],'scene':row['sample_id'].split('_')[0],
            'category':row['category'],'dataset':'nuscenes','existing_reviewed_object':existing,
            'source_rgb':str(rgb),'source_rgb_sha256':sha(rgb),'source_mask':str(mask),'source_mask_sha256':sha(mask),
            'image_rgba':str(rgba),'image_rgba_sha256':sha(rgba),'retained_rgba':source,'retained_rgba_sha256':source_sha,
            'retained_size':list(size),'adapter_size':[512,512],'resize_filter':'PIL LANCZOS',
            'pixel_transform_A_edge_coordinates':A.tolist(),'H_canvas_from_retained_pixel_centres':H.tolist(),
            'real_input_count':1,'generated_input_count':0,'camera_given_to_image_model':False,
            'real_heldout_eligible':False,'model_specific_preprocessing_validation':'pending',
            'source_view_camera_registration':'pending'}
        if not existing:
            Hraw=H@np.asarray(row['camera']['H_stored_from_raw_pixel_centres'])
            record.update(H_canvas_from_raw_pixel_centres=Hraw.tolist(),
                K_canvas=(Hraw@np.asarray(row['camera']['K_raw'])).tolist(),
                T_world_from_camera=row['camera']['T_world_from_camera'],timestamp_us=row['camera']['timestamp_us'],
                sample_data_token=row['camera']['sample_data_token'],native_instance_token=row['native_instance_token'])
        # This identity checks coordinates only; never certify asset registration here.
        points=np.array([[0,0,1],[size[0]-1,size[1]-1,1],[size[0]/2,size[1]/2,1]])
        record['pixel_transform_roundtrip_error']=float(np.max(np.abs((points@H.T)@np.linalg.inv(H).T-points)))
        assert record['pixel_transform_roundtrip_error']<1e-9
        records.append(record)
    manifest={'schema':1,'status':'source_input_adapters_staged','created_at':datetime.now(timezone.utc).isoformat(),
        'selection':str(B/'selection.json'),'selection_sha256':frozen['selection_sha256'],'script_sha256':sha(__file__),
        'objects':records,'counts':{'objects':432,'existing_inputs_preserved':199,'new_inputs':233},
        'policy':'Same source identity and alpha; preserve old JPEG/mask bytes. Single-view interface is not an assertion that native model information budgets or preprocessing are identical.',
        'rgba_note':'RGBA has soft alpha and original foreground colours; JPEG uses white background. Native models may require a documented additional adapter.',
        'pixel_coordinates':'Integer pixel centres. H uses x_out=(x_in+0.5)*scale-0.5+padding; legacy A is retained separately as edge coordinates.',
        'all_model_inputs_runtime_validated':False,'asset_alignment_verified':False,'real_heldout_eligible':False}
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(manifest['counts']))
if __name__=='__main__':main()
