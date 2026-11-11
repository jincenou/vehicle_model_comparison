"""Reference mask placement in raw camera pixels, without model-dependent alignment."""
import numpy as np
from PIL import Image
from source_metric_core import reference_roi, roi_canvas


def place_reference(raw_rgb, retained_rgba, crop_box, stored_from_raw):
    raw=np.asarray(raw_rgb);rgba=np.asarray(retained_rgba)
    if raw.dtype!=np.uint8 or raw.ndim!=3 or raw.shape[2]!=3:
        raise ValueError('raw uint8 RGB required')
    if rgba.dtype!=np.uint8 or rgba.ndim!=3 or rgba.shape[2]!=4:
        raise ValueError('retained uint8 RGBA required')
    if len(crop_box)!=4 or any(type(x)is not int for x in crop_box):
        raise ValueError('integer pixel-edge crop box required')
    left,top,right,bottom=crop_box;cw,ch=right-left,bottom-top
    if cw<=0 or ch<=0:raise ValueError('empty crop')
    h,w=raw.shape[:2];sh,sw=rgba.shape[:2];sx,sy=sw/cw,sh/ch
    expected=np.array([[sx,0,-left*sx+(sx-1)/2],[0,sy,-top*sy+(sy-1)/2],[0,0,1.]])
    H=np.asarray(stored_from_raw)
    if H.shape!=(3,3)or not np.allclose(H,expected,rtol=0,atol=1e-8):
        raise ValueError('crop/resize pixel-centre transform disagrees with provenance')
    if not(rgba[:,:,3]>127).any():raise ValueError('empty reference alpha')
    native=(sw,sh)==(cw,ch)
    mask_image=Image.fromarray(rgba[:,:,3]).resize((cw,ch),Image.Resampling.NEAREST)
    canvas=Image.new('L',(w,h),0);canvas.paste(mask_image,(left,top));mask=np.asarray(canvas)>127
    if not mask.any():raise ValueError('reference alpha lies outside raw image')
    # Exact native RGB check only where the crop actually intersects the raw camera.
    rgb_error=None
    if native:
        x0,y0,x1,y1=max(0,left),max(0,top),min(w,right),min(h,bottom)
        retained=rgba[y0-top:y1-top,x0-left:x1-left];m=retained[:,:,3]>127
        rgb_error=int(np.abs(raw[y0:y1,x0:x1].astype(np.int16)-retained[:,:,:3].astype(np.int16))[m].max())
        if rgb_error!=0:raise ValueError('native reference foreground differs from raw RGB')
    white=np.where(mask[:,:,None],raw,255).astype(np.uint8)
    roi=reference_roi(mask);s=512/(roi[2]-roi[0])
    Hroi=np.array([[s,0,-s*roi[0]+(s-1)/2],[0,s,-s*roi[1]+(s-1)/2],[0,0,1.]])
    border=dict(left=int(mask[:,0].sum()),right=int(mask[:,-1].sum()),top=int(mask[0].sum()),bottom=int(mask[-1].sum()))
    a=rgba[:,:,3]>127
    crop_border=dict(left=int(a[:,0].sum()),right=int(a[:,-1].sum()),top=int(a[0].sum()),bottom=int(a[-1].sum()))
    meta=dict(mask_mapping='exact_native_crop'if native else'nearest_upsampled_mask_diagnostic_only',
              native_reference_pixels_verified=native,foreground_rgb_max_error=rgb_error,
              raw_mask_pixels=int(mask.sum()),raw_frame_edge_pixels=border,
              raw_frame_edge_contact=any(border.values()),stored_crop_edge_pixels=crop_border,
              stored_crop_edge_contact=any(crop_border.values()),reference_roi_raw=roi,
              H_roi512_from_raw_pixel_centres=Hroi.tolist(),
              reference_mask_bbox_height_px=int(np.where(mask)[0].max()-np.where(mask)[0].min()+1),
              reference_qualified=False)
    return mask,roi_canvas(white,roi),roi_canvas(mask,roi,True),meta
