"""Preserve native opacity logits rather than sigmoid/logit round-trip saturation."""
import numpy as np
from plyfile import PlyData


def save_native_gaussian(gs, path):
    gs.save_ply(str(path))
    payload = PlyData.read(path, mmap=False)
    # The native saver computes inverse_sigmoid(sigmoid(logits)), which
    # can produce infinity for finite high logits in float32.
    payload['vertex']['opacity'] = (gs._opacity + gs.opacity_bias).detach().cpu().numpy().ravel()
    assert all(np.isfinite(payload['vertex'][name]).all()
               for name in payload['vertex'].data.dtype.names)
    payload.write(path)
