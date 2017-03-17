import os
import numpy as np

from camera_params import *
from undistort_func import undistort, undistort_depth

def project_depth_map(imgDepth, rgb):

    H, W = imgDepth.shape

    kc_d = [k1_d, k2_d, p1_d, p2_d, k3_d];
    fc_d = [fx_d,fy_d];
    cc_d = [cx_d,cy_d]; 

    fc_rgb = [fx_rgb,fy_rgb];
    cc_rgb = [cx_rgb,cy_rgb]; 
    kc_rgb = [k1_rgb,k2_rgb,p1_rgb,p2_rgb,k3_rgb];

    rgbUndistorted = np.zeros(rgb.shape);

    for ii in range(rgb.shape[2]):
	rgbUndistorted[:,:,ii] = undistort(rgb[:,:,ii].astype(np.float),
	    fc_rgb, cc_rgb, kc_rgb, 0)

    rgbUndistorted = rgbUndistorted.astype(np.uint8)

    noiseMask = 255 * (imgDepth == max(imgDepth.reshape(imgDepth.size))).astype(np.float)
    noiseMask = undistort(noiseMask, fc_d, cc_d, kc_d, 0)
    noiseMask = noiseMask > 0

    imgDepth = undistort_depth(imgDepth.astype(np.float),fc_d,cc_d,kc_d,0, noiseMask);
