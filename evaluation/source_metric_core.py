"""Strict source-image numerical primitives; these do not certify data eligibility."""
import hashlib
import json
import math

import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity


def rgb01(image):
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[2] != 3 or not image.size:
        raise ValueError('Expected nonempty HWC RGB')
    if image.dtype == np.uint8:
        return image.astype(np.float64) / 255
    if image.dtype.kind != 'f' or not np.isfinite(image).all():
        raise ValueError('Expected uint8 or finite float RGB [0,1]')
    if image.min() < 0 or image.max() > 1:
        raise ValueError('Float RGB outside [0,1]; no implicit normalization')
    return image.astype(np.float64)


def pair_rgb(reference, prediction):
    a, b = rgb01(reference), rgb01(prediction)
    if a.shape != b.shape:
        raise ValueError('Image shapes must match without broadcasting')
    return a, b


def scalar(value):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError('Nonfinite ordinary score')
    return {'value': value, 'status': 'ok'}


def psnr(reference, prediction):
    a, b = pair_rgb(reference, prediction)
    mse = float(np.mean((a - b) ** 2, dtype=np.float64))
    if mse == 0:
        return dict(value=None, status='positive_infinity', perfect_match=True, mse=0.0)
    return dict(scalar(10 * math.log10(1 / mse)), perfect_match=False, mse=mse)


def ssim(reference, prediction):
    a, b = pair_rgb(reference, prediction)
    if min(a.shape[:2]) < 11:
        raise ValueError('SSIM canvas is smaller than fixed 11px window')
    return scalar(structural_similarity(
        a, b, channel_axis=-1, data_range=1, gaussian_weights=True,
        sigma=1.5, win_size=11, use_sample_covariance=False, K1=.01, K2=.03))


def binary_mask(mask):
    mask = np.asarray(mask)
    if mask.ndim != 2 or not mask.size:
        raise ValueError('Expected nonempty 2D mask')
    if mask.dtype == np.bool_:
        return mask
    if mask.dtype != np.uint8:
        raise ValueError('Mask must be bool or uint8; alpha threshold is byte >127')
    return mask > 127


def iou(reference, prediction):
    a, b = binary_mask(reference), binary_mask(prediction)
    if a.shape != b.shape:
        raise ValueError('Mask shapes must match')
    if not a.any():
        return dict(value=None, status='unavailable', reason='empty_reference_mask')
    inter, union = int((a & b).sum()), int((a | b).sum())
    return dict(scalar(inter / union), intersection=inter, union=union,
                reference_pixels=int(a.sum()), prediction_pixels=int(b.sum()))


def reference_roi(mask):
    """Reference alone defines the crop, including outside-source white padding."""
    mask = binary_mask(mask)
    y, x = np.where(mask)
    if not len(x):
        raise ValueError('Empty reference cannot define ROI')
    x0, x1, y0, y1 = int(x.min()), int(x.max())+1, int(y.min()), int(y.max())+1
    # Exact ceil(1.10 * longest side), avoiding floating-point integer-boundary drift.
    side = (11 * max(x1-x0, y1-y0) + 9) // 10
    left, top = (x0+x1-side)//2, (y0+y1-side)//2
    return [left, top, left+side, top+side]


def roi_canvas(image, box, mask=False):
    """Same explicit box for reference and prediction; no prediction-dependent fitting."""
    array = np.asarray(image)
    if len(box) != 4 or any(type(v) is not int for v in box):
        raise ValueError('ROI must contain four integers')
    left, top, right, bottom = box
    if right-left != bottom-top or right <= left:
        raise ValueError('ROI must be a positive square')
    if mask:
        array = binary_mask(array).astype(np.uint8)*255
        mode, background, resize = 'L', 0, Image.Resampling.NEAREST
    else:
        if array.dtype != np.uint8:
            raise ValueError('Canonical canvas expects saved uint8 RGB')
        rgb01(array)
        mode, background, resize = 'RGB', (255,255,255), Image.Resampling.BICUBIC
    canvas = Image.new(mode, (right-left, bottom-top), background)
    canvas.paste(Image.fromarray(array), (-left, -top))
    return np.asarray(canvas.resize((512,512), resize))


def validate_camera(validation, contract, contract_hash):
    """Verify recorded native-render camera evidence, not asset registration or occlusion."""
    if validation['status'] != 'passed' or validation['sample_id'] != contract['sample_id']:
        raise ValueError('Invalid render or wrong object')
    audit = validation['common_camera_audit']
    if audit['contract_sha256'] != contract_hash:
        raise ValueError('Camera contract hash mismatch')
    error = audit['projection_max_error_px']
    if not math.isfinite(error) or not 0 <= error < .05:
        raise ValueError('Projection probe failed')
    if audit['render_wh'] != contract['render_wh']:
        raise ValueError('Canvas mismatch')
    if audit['camera_pose_refinement'] is not False or audit['render_time_shift_seconds'] != 0:
        raise ValueError('Private camera/time correction active')
    if validation['frozen_raw_timestamp_us'] != contract['timestamp_us']:
        raise ValueError('Timestamp mismatch')
    return True


def paired_summary(rows, methods, cohort, metric, protocol_id):
    """One record per method/object; never independently drop missing values per method."""
    if not methods or len(set(methods)) != len(methods) or len(set(cohort)) != len(cohort):
        raise ValueError('Methods/cohort must be unique and methods nonempty')
    table = {}
    for row in rows:
        key = row['method'], row['sample_id']
        if key[0] not in methods or key[1] not in cohort:
            raise ValueError('Row outside frozen scope')
        if key in table:
            raise ValueError('Duplicate method/object row')
        if row['protocol_id'] != protocol_id:
            raise ValueError('Mixed metric protocols')
        table[key] = row.get('scores', {}).get(metric)
    def value(record):
        if record is None or record.get('status') == 'unavailable':
            return None
        if record.get('status') == 'positive_infinity':
            if metric != 'PSNR' or record.get('perfect_match') is not True or record.get('mse') != 0 or record.get('value') is not None:
                raise ValueError('Invalid perfect PSNR record')
            return math.inf
        if record.get('status') != 'ok' or record.get('value') is None:
            raise ValueError('Unrecognized scalar status')
        result = float(record['value'])
        if not math.isfinite(result):
            raise ValueError('Nonfinite score without explicit infinity status')
        return result
    values = {key:value(v) for key,v in table.items()}
    common = sorted(s for s in cohort if all(values.get((m,s)) is not None for m in methods))
    def stats(vals):
        if not vals:
            return dict(n=0, mean={'value':None,'status':'unavailable','reason':'empty_subset'})
        infs = sum(math.isinf(v) for v in vals)
        avg = dict(value=None,status='positive_infinity') if infs else scalar(np.mean(vals))
        return dict(n=len(vals), positive_infinity_count=infs, mean=avg)
    return dict(metric=metric, protocol_id=protocol_id, requested=len(cohort),
        paired_ids=common, paired_ids_sha256=hashlib.sha256(json.dumps(common,separators=(',',':')).encode()).hexdigest(),
        paired={m:stats([values[m,s] for s in common]) for m in methods},
        own_valid={m:stats([values[m,s] for s in cohort if values.get((m,s)) is not None]) for m in methods})
