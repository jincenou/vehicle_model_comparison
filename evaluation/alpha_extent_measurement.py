"""Candidate representation-neutral extent measurement from native alpha images.

Pixel-edge coordinates: pixel i occupies [i,i+1], centre i+.5; K principal
point is resolution/2. This is a measurement experiment, not an approved metric.
"""
import numpy as np
from asset_orientation_contract import checked_rigid
from spatial_calibration import vector3, positive


def rig(center, span, resolution):
    center=vector3(center,'center');span=positive(span,'span')
    if type(resolution) is not int or resolution<32:raise ValueError('Invalid resolution')
    views=[]
    for axis in range(3):
        for sign in [1,-1]:
            direction=np.eye(3)[axis]*sign;forward=-direction
            up=np.array([0.,1,0]) if axis==2 else np.array([0.,0,1])
            right=np.cross(forward,up);down=np.cross(forward,right)
            rotation=np.stack([right,down,forward]);eye=center+3*span*direction
            view=np.eye(4);view[:3,:3]=rotation;view[:3,3]=-rotation@eye;checked_rigid(view)
            k=np.array([[resolution/span,0,resolution/2],[0,resolution/span,resolution/2],[0,0,1.]])
            views.append(dict(name=f'{"xyz"[axis]}_{"pos" if sign>0 else "neg"}',
                              T_camera_from_semantic=view.tolist(),K_pixel_edges=k.tolist(),
                              resolution=resolution,span=span,center=center.tolist(),near=span,far=5*span))
    return views


def measure_view(alpha, view):
    alpha=np.asarray(alpha);n=view['resolution']
    if alpha.shape!=(n,n) or alpha.dtype!=np.uint8:raise ValueError('Expected uint8 NxN native alpha')
    y,x=np.where(alpha>127)
    if len(x)==0:raise ValueError('No alpha support above common threshold')
    if x.min()<2 or y.min()<2 or x.max()>=n-2 or y.max()>=n-2:
        raise ValueError('Alpha support touches framing margin; measurement is clipped')
    rotation=np.asarray(view['T_camera_from_semantic'])[:3,:3];checked_rigid(view['T_camera_from_semantic'])
    center=np.asarray(view['center']);k=np.asarray(view['K_pixel_edges']);bounds={}
    for image_axis,indices in enumerate([x,y]):
        basis=rotation[image_axis];axis=int(np.argmax(np.abs(basis)));sign=basis[axis]
        if not np.allclose(np.abs(basis),np.eye(3)[axis],atol=1e-12):raise ValueError('Non-cardinal view')
        edges=np.array([indices.min(),indices.max()+1.],dtype=float)
        values=center[axis]+sign*(edges-k[image_axis,2])/k[image_axis,image_axis]
        bounds[str(axis)]=[float(values.min()),float(values.max())]
    return dict(name=view['name'],bounds=bounds,foreground_pixels=len(x),pixel_size=view['span']/n)


def combine(views):
    expected={f'{a}_{s}' for a in 'xyz' for s in ['pos','neg']}
    if len(views)!=6 or {v['name'] for v in views}!=expected:raise ValueError('All six unique views required')
    low=[];high=[];spread=[]
    for axis in range(3):
        values=np.asarray([v['bounds'][str(axis)] for v in views if str(axis) in v['bounds']])
        if values.shape!=(4,2):raise ValueError('Each axis requires four measurements')
        low.append(float(values[:,0].min()));high.append(float(values[:,1].max()))
        spread.append(float(np.ptp(values,axis=0).max()))
    lo=np.asarray(low);hi=np.asarray(high)
    if not np.isfinite([lo,hi]).all() or np.any(hi<=lo):raise ValueError('Invalid measured extent')
    return dict(lower=low,upper=high,center=((lo+hi)/2).tolist(),extent=(hi-lo).tolist(),
                across_view_boundary_spread=spread,rule='union_of_six_native_alpha_supports_byte_gt127',
                formal_metrics_qualified=False)


def convergence(coarse,fine,span,coarse_resolution):
    error=float(np.max(np.abs(np.asarray([coarse['lower'],coarse['upper']])-np.asarray([fine['lower'],fine['upper']]))))
    tolerance=4*positive(span,'span')/coarse_resolution
    return dict(boundary_max_difference=error,predeclared_tolerance=tolerance,
                passed=error<=tolerance,formal_metrics_qualified=False)
