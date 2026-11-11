"""Score traced source views with fixed provenance crop, never reference-mask predictions."""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from skimage.metrics import structural_similarity
import lpips

ROOT=Path(__file__).resolve().parents[1]

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):
    p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp.json')
    tmp.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n');tmp.replace(p)

def canvas(im,mode):
    w,h=im.size;ratio=min(512/w,512/h);size=(max(1,round(w*ratio)),max(1,round(h*ratio)))
    bg=Image.new(mode,(512,512),255 if mode=='RGB' else 0)
    # PIL RGB scalar 255 is red; explicitly use white.
    if mode=='RGB':bg=Image.new('RGB',(512,512),(255,255,255))
    bg.paste(im.resize(size,Image.Resampling.BILINEAR),((512-size[0])//2,(512-size[1])//2))
    return bg

def measures(x,y,am,bm,net):
    a=np.asarray(x,dtype=np.float64)/255;b=np.asarray(y,dtype=np.float64)/255
    ma=np.asarray(am)>127;mb=np.asarray(bm)>127
    union=ma|mb;inter=ma&mb;mse=float(np.mean((a-b)**2))
    ss=float(structural_similarity(a,b,channel_axis=2,data_range=1,gaussian_weights=True,
          sigma=1.5,use_sample_covariance=False,win_size=11))
    t=lambda im:torch.from_numpy(np.array(im,dtype=np.float32)/127.5-1).permute(2,0,1)[None].cuda()
    with torch.no_grad():lp=float(net(t(x),t(y)).item())
    return {'PSNR_source_dB':-10*np.log10(mse) if mse>0 else None,'perfect_match':mse==0,
            'SSIM_source':ss,'LPIPS_source':lp,'IoU_source':float(inter.sum()/union.sum()) if union.any() else None}

def main():
    p=argparse.ArgumentParser();p.add_argument('--method',choices=['splatad','omnire','streetgs'],required=True);p.add_argument('--sample-id',required=True);a=p.parse_args()
    torch.set_num_threads(4)
    torch.hub.download_url_to_file=lambda *args,**kwargs: (_ for _ in ()).throw(RuntimeError('Unexpected metric weight download'))
    tag={'splatad':'splatad_reviewed288_train_only','omnire':'omnire_reviewed288','streetgs':'streetgs_reviewed288'}[a.method]
    source=ROOT/'exports'/tag/'objects'/a.sample_id
    if a.method=='streetgs':source=source/'formal'
    validation=json.loads((source/'validation.json').read_text())
    out=ROOT/'metrics/native_source_v1'/a.method/a.sample_id
    base={'method':a.method,'sample_id':a.sample_id,'qualification':'native_source_reconstruction_diagnostic',
          'common_asset_alignment_qualified':False,'geometry':None,'heldout':None,
          'missing':{'geometry':'no_independent_reference_surface','heldout':'not_evaluated_in_source_pass',
                     'common_asset_alignment':'Ours/AH source pose alignment and cross-method rig remain to validate'}}
    if validation['status']!='passed':
        write(out/'metrics.json',dict(base,status='unavailable',reason=validation.get('reason','native_validation_failed')));return
    if a.method=='splatad':assert validation.get('background')=='white','Re-render source on white before scoring'
    cohort=json.loads((ROOT/'inputs/benchmark_20261106/selection.json').read_text())
    selected=next(x for x in cohort['vehicles'] if x['sample_id']==a.sample_id)
    manifest=ROOT.parent/'assetbench/reports/vehicle_expansion_v4/selection.json' if selected.get('review_id') else ROOT/'inputs/vehicle53_selection.json'
    prov=next(x for x in json.loads(manifest.read_text())['samples'] if x['sample_id']==a.sample_id)
    obs=prov['observation'];assert Path(obs['source_image']).resolve()==Path(validation['source_image']).resolve()
    assert sha(selected['retained_rgba'])==selected['retained_rgba_sha256']==obs['sha256']
    raw=Image.open(obs['source_image']).convert('RGB');rgba=Image.open(selected['retained_rgba']).convert('RGBA')
    x0,y0,x1,y1=obs['crop_box'];assert 0<=x0<x1<=raw.width and 0<=y0<y1<=raw.height
    alpha=rgba.getchannel('A');gt=Image.new('RGB',rgba.size,(255,255,255));gt.paste(rgba,mask=alpha)
    pred=Image.open(validation['actor_preview']).convert('RGB');pa=Image.open(validation['actor_alpha']).convert('L')
    assert pred.size==pa.size
    # Exact full-view scaling only; a cropped native FOV cannot be stretched to full-image FOV.
    ratio_error=abs(pred.width/raw.width-pred.height/raw.height)
    if ratio_error>1/raw.height+1/raw.width:
        write(out/'metrics.json',dict(base,status='unavailable',reason='native_render_fov_aspect_mismatch',raw_wh=list(raw.size),render_wh=list(pred.size)));return
    box=(x0*pred.width/raw.width,y0*pred.height/raw.height,x1*pred.width/raw.width,y1*pred.height/raw.height)
    pred=pred.resize(rgba.size,Image.Resampling.BILINEAR,box=box)
    pa=pa.resize(rgba.size,Image.Resampling.BILINEAR,box=box)
    gt,alpha,pred,pa=canvas(gt,'RGB'),canvas(alpha,'L'),canvas(pred,'RGB'),canvas(pa,'L')
    net=lpips.LPIPS(net='alex',version='0.1',verbose=False).cuda().eval()
    identity=measures(gt,gt,alpha,alpha,net)
    assert identity['perfect_match'] and abs(identity['SSIM_source']-1)<1e-9
    assert abs(identity['LPIPS_source'])<1e-6 and identity['IoU_source']==1
    scores=measures(gt,pred,alpha,pa,net)
    out.mkdir(parents=True,exist_ok=True)
    for name,im in [('reference',gt),('reference_alpha',alpha),('prediction',pred),('prediction_alpha',pa)]:im.save(out/(name+'.png'))
    write(out/'metrics.json',dict(base,status='completed',category=selected['category'],scores=scores,
          identity_test=identity,validation_sha256=sha(source/'validation.json'),source_rgba_sha256=selected['retained_rgba_sha256'],
          crop_box=obs['crop_box'],canvas=[512,512],background='white',alpha_threshold=0.5,
          preprocessing='fixed provenance crop; aspect-preserving resize and padding; no per-method foreground recrop',
          source_camera_policy='native source camera including method-native trained pose correction',
          prediction_mask_policy='own alpha only; no reference mask applied to prediction'))
    print(json.dumps({'method':a.method,'sample_id':a.sample_id,'scores':scores}),flush=True)

if __name__=='__main__':main()
