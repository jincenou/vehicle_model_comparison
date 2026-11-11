"""Float64 feature arithmetic, independent of encoder and data qualification.

Frechet statistics follow the mean and n-1 covariance convention in
https://github.com/mseitzer/pytorch-fid/blob/master/src/pytorch_fid/fid_score.py
The covariance cross term is evaluated as a nuclear norm of centered factors;
this avoids a nonsymmetric matrix square root for rank-deficient small cohorts.
Neither these functions nor successful numerical tests authorize formal scores.
"""
import numpy as np

def features(value, minimum_rows=1):
    x=np.asarray(value)
    if x.ndim!=2 or x.shape[0]<minimum_rows or x.shape[1]<1:
        raise ValueError('Expected nonempty N x D feature matrix with sufficient rows')
    if x.dtype.kind!='f' or not np.isfinite(x).all():
        raise ValueError('Features must be finite real floats')
    return x.astype(np.float64)

def normalized(value):
    x=features(value)
    # Scale first to avoid overflow/underflow in the norm.
    scale=np.max(np.abs(x),axis=1,keepdims=True)
    if np.any(scale==0):raise ValueError('Zero embedding has no cosine direction')
    x=x/scale
    return x/np.linalg.norm(x,axis=1,keepdims=True)

def orbit_i16(source, renders):
    a=normalized(source);b=normalized(renders)
    if len(a)!=1 or len(b)!=16 or a.shape[1]!=b.shape[1]:
        raise ValueError('One source and all 16 fixed-view embeddings are required')
    scores=np.clip(b@a[0],-1,1)
    return dict(value=float(scores.mean()),per_view=scores.tolist(),views=16,best_view_selection=False)

def foreground_cosine_distance(reference, prediction):
    """Arithmetic for ED-R after externally validated foreground patch pooling."""
    a=normalized(reference);b=normalized(prediction)
    if a.shape!=b.shape or len(a)!=1:raise ValueError('Expected one matching pooled feature pair')
    return float(1-np.clip(a[0]@b[0],-1,1))

def frechet_features(reference, prediction):
    """Group Frechet distance; no implicit feature normalization or resizing.

For centered factors X/sqrt(n-1), Y/sqrt(m-1), covariance traces
are squared Frobenius norms and the covariance cross term is ||XY^T||_*.
QR removes only orthonormal left factors and preserves that nuclear norm.
The encoder/normalization/cohort/view contracts must be checked by the caller.
"""
    a=features(reference,2);b=features(prediction,2)
    if a.shape[1]!=b.shape[1]:raise ValueError('Feature dimensions differ')
    with np.errstate(over='raise',invalid='raise',divide='raise'):
        delta=a.mean(axis=0)-b.mean(axis=0)
        x=(a-a.mean(axis=0))/np.sqrt(len(a)-1)
        y=(b-b.mean(axis=0))/np.sqrt(len(b)-1)
        mean_term=float(delta@delta)
        trace_a=float(np.sum(x*x));trace_b=float(np.sum(y*y))
        # R is sufficient: no large Q matrix is needed.
        rx=np.linalg.qr(x,mode='r') if x.shape[0]>x.shape[1] else x
        ry=np.linalg.qr(y,mode='r') if y.shape[0]>y.shape[1] else y
        cross=float(np.linalg.svd(rx@ry.T,compute_uv=False).sum())
        raw=mean_term+trace_a+trace_b-2*cross
    if not np.isfinite([mean_term,trace_a,trace_b,cross,raw]).all():
        raise ValueError('Nonfinite Frechet arithmetic')
    tolerance=64*np.finfo(np.float64).eps*max(1,min(a.shape[0],b.shape[0],a.shape[1]))*max(1.,mean_term+trace_a+trace_b+2*cross)
    if raw < -tolerance:raise ArithmeticError('Negative Frechet distance beyond roundoff tolerance')
    return dict(value=max(0.,raw),raw_value=raw,roundoff_clamped=raw<0,
        roundoff_tolerance=tolerance,reference_count=len(a),prediction_count=len(b),
        feature_dimension=a.shape[1],covariance_ddof=1,mean_term=mean_term,
        reference_covariance_trace=trace_a,prediction_covariance_trace=trace_b,
        covariance_cross_trace=cross,solver='centered_factor_nuclear_norm_qr',
        reference_rank_upper_bound=min(len(a)-1,a.shape[1]),
        prediction_rank_upper_bound=min(len(b)-1,b.shape[1]),
        small_sample_reliability_certified=False)
