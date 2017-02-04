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

import numpy as np
from net import MachinePart

class NYUDepthModelDefs(object):
    def define_meta(self):
        '''
        precomputed means and stdev
        '''
        # just hardcoding for this release, was in meta.mat file
        orig_input_size = (240, 320) # before data transforms
        input_size = (228, 304) # after data transforms

        vgg_image_mean = np.array((123.68, 116.779, 103.939), dtype=np.float32)
        images_mean = 109.31410628
        images_std = 76.18328376
        images_istd = 1.0 / images_std
        depths_mean = 2.53434899
        depths_std = 1.22576694
        depths_istd = 1.0 / depths_std
        logdepths_mean = 0.82473954
        logdepths_std = 0.45723134
        logdepths_istd = 1.0 / logdepths_std

        self.orig_input_size = orig_input_size
        self.input_size = input_size
        self.meta = MachinePart(locals())

