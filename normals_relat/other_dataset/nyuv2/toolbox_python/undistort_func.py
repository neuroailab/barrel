import os
import numpy as np
from apply_distortion import *

def rect(I, R = np.eye(3), f = [1, 1], c = [0, 0], k = [0,0,0,0,0], alpha = None, KK_new = None, noiseMask = None):
    if KK_new is None:
        if not alpha is None:
            KK_new = alpha
        else:
            KK_new = [[f[0],0,c[0]], [0,f[1],c[1]], [0,0,1]];
        alpha = 0
    
    nr, nc = I.shape
    Irec = np.ones([nr, nc])*255
    mx, my = np.meshgrid(range(nc), range(nr))

    #py = mx.transpose().reshape([nc*nr, 1])
    #px = my.transpose().reshape([nc*nr, 1])
    px = mx.reshape([nc*nr, 1])
    py = my.reshape([nc*nr, 1])

    #print(KK_new)
    #print(py[:100])

    rays = np.dot(np.linalg.inv(KK_new), np.concatenate([px.transpose(), py.transpose(), np.ones([1, px.size])]))
    rays2 = np.dot(R.transpose(), rays);

    #print(R)
    #print(rays[0, 99])
    #print(rays2[0, 99])

    x = np.asarray([np.divide(rays2[0,:], rays2[2,:]), np.divide(rays2[1,:], rays2[2,:])]);

    xd, not_used = apply_distortion(x, k)

    #print(sum(sum(xd)))
    #print(xd[:, 479:500])

    px2 = f[0]*(xd[0,:]+alpha*xd[1,:])+c[0]
    py2 = f[1]*xd[1,:]+c[1]

    #print(sum(px2[:1001]))
    #print(sum(px2[399:500]))
    #print(px2[479:500])

    px_0 = np.floor(px2)

    py_0 = np.floor(py2)
    py_1 = py_0 + 1

    #print(px_0[:20])
    #print(py_0[:20])
    #print(sum(px_0))
    #print(sum(abs(px_0)))

    good_points = (px_0 >= 0) & (px_0 <= (nc-2)) & (py_0 >= 0) & (py_0 <= (nr-2))

    px2 = px2[good_points]
    py2 = py2[good_points]
    px_0 = px_0[good_points]
    py_0 = py_0[good_points]

    #print(good_points.nonzero()[:20])
    #print(py_0.shape)

    alpha_x = px2 - px_0
    alpha_y = py2 - py_0

    a1 = (1 - alpha_y)*(1 - alpha_x)
    a2 = (1 - alpha_y)*alpha_x
    a3 = alpha_y * (1 - alpha_x)
    a4 = alpha_y * alpha_x

    ind_lu = px_0 * nr + py_0
    ind_ru = (px_0 + 1) * nr + py_0
    ind_ld = px_0 * nr + (py_0 + 1)
    ind_rd = (px_0 + 1) * nr + (py_0 + 1)

    ind_new = px[good_points]*nr + py[good_points]

    #tmp_test = good_points.nonzero()
    #print(tmp_test[0][:20])
    #print(sum(ind_new))
    
    px_0 = px_0.astype(np.int)
    py_0 = py_0.astype(np.int)
    py_indx = py[good_points].astype(np.int)
    px_indx = px[good_points].astype(np.int)

    if noiseMask is None:
        test_now = a1*I[py_0, px_0] + a2*I[py_0, px_0 + 1] + a3 * I[py_0 + 1, px_0] + a4 * I[py_0 + 1, px_0 + 1]
        Irec[py_indx, px_indx] = test_now[:, np.newaxis]
    else:
        noiseMask = np.logical_not(noiseMask)
	a1 = a1 * noiseMask[py_0, px_0];
	a2 = a2 * noiseMask[py_0, px_0 + 1];
	a3 = a3 * noiseMask[py_0 + 1, px_0];
	a4 = a4 * noiseMask[py_0 + 1, px_0 + 1];

	s = a1 + a2 + a3 + a4;

	badPix = s == 0;

	a1 = a1 / s;
	a2 = a2 / s;
	a3 = a3 / s;
	a4 = a4 / s;

	a1[badPix] = 0;
	a2[badPix] = 0;
	a3[badPix] = 0;
	a4[badPix] = 0;
        Irec = np.zeros([nr, nc])

        test_now = a1*I[py_0, px_0] + a2*I[py_0, px_0 + 1] + a3 * I[py_0 + 1, px_0] + a4 * I[py_0 + 1, px_0 + 1]
        Irec[py_indx, px_indx] = test_now[:, np.newaxis]

        #print(sum(sum(Irec)))
    return Irec
    

def undistort(I, fc, cc, kc, alpha_c):
    KK_new = [[fc[0], alpha_c*fc[0], cc[0]], [0, fc[1], cc[1]],[0, 0, 1]]

    I2 = rect(I,np.eye(3),fc,cc,kc,KK_new);

    return I2

def undistort_depth(I, fc, cc, kc, alpha_c, noiseMask):
    KK_new = [[fc[0], alpha_c*fc[0], cc[0]], [0, fc[1], cc[1]],[0, 0, 1]]

    I2 = rect(I,np.eye(3),fc,cc,kc,KK_new, noiseMask = noiseMask);

    return I2
