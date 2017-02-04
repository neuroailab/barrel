'''
Copyright (C) 2014 New York University

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
'''
import os
import time
import numpy as np

from common import imgutil, logutil

xx = np.newaxis

def image_montage(imgs, min, max):
    imgs = imgutil.bxyc_from_bcxy(imgs)
    return imgutil.montage(
        imgutil.scale_values(imgs, min=min, max=max),
        border=1)

def depth_montage(depths):
    import matplotlib.pyplot as plt
    if depths.ndim == 4:
        assert depths.shape[1] == 1
        depths = depths[:,0,:,:]
    #depths = imgutil.scale_values(depths, min=-2.5, max=2.5)
    #depths = map(imgutil.scale_values, depths)
    masks = []
    for i in xrange(len(depths)):
        x = depths[i]
        mask = x != x.min() 
        masks.append(mask)
        x = x[mask]
        if len(x) == 0:
            d = np.zeros_like(depths[i])
        else:
            d = imgutil.scale_values(depths[i], min=x.min(), max=x.max())
        depths[i] = d
    depths = plt.cm.jet(depths)[...,:3]
    for i in xrange(len(depths)):
        for c in xrange(3):
            depths[i, :, :, c][masks[i] == 0] = 0.2
    return imgutil.montage(depths, border=1)

def normals_montage(normals):
    '''
    Montage of normal maps. Vectors are unit length and backfaces thresholded.
    '''
    normals = normals.copy()
    x = normals[:,0,:,:] # horizontal; pos right
    y = normals[:,1,:,:] # depth; pos far
    z = normals[:,2,:,:] # vertical; pos up
    backfacing = (y > 0)
    #y[backfacing] = 0
    norm = np.sqrt(np.sum(normals**2, axis=1))
    zero = (norm < 1e-5)
    normals /= np.maximum(1e-5, norm[:,xx,:,:])
    x += 1; x *= 0.5
    y *= -1
    z += 1; z *= 0.5
    x[backfacing] = 0.4
    y[backfacing] = 0.4
    z[backfacing] = 0.4
    x[zero] = 0.0
    y[zero] = 0.0
    z[zero] = 0.0
    return imgutil.montage(normals.transpose((0,2,3,1)), border=1)

def multichannel_montage(x):
    import matplotlib.pyplot as plt
    (nimgs, nchan, nh, nw) = x.shape
    colors = plt.cm.hsv(np.linspace(0, 1, nchan+1))[:-1, :3]
    imgs = np.zeros((nimgs, nh, nw, 3))
    for i in xrange(nimgs):
        imgs[i] = imgutil.scale_values(np.dot(x[i].transpose((1,2,0)), colors))
    return imgutil.montage(imgs, border=1)

def normals_weights_montage(normals):
    '''
    Montage for normal vector output weights.
    Values can be backfacing and nonunit length.
    '''
    return imgutil.montage(imgutil.scale_values(normals.transpose((0,2,3,1))),
                           border=1)

def zero_pad_batch(batch, bsize):
    assert len(batch) <= bsize
    if len(batch) == bsize:
        return batch
    n = batch.shape[0]
    shp = batch.shape[1:]
    return np.concatenate((batch, np.zeros((bsize - n,) + shp,
                                           dtype=batch.dtype)))

