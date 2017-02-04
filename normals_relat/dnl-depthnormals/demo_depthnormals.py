import os
import sys
import numpy as np

from PIL import Image

import net
from utils import depth_montage, normals_montage

def main():
    # location of depth module, config and parameters
        
    #model_name = 'depthnormals_nyud_alexnet'
    model_name = 'depthnormals_nyud_vgg'

    module_fn = 'models/iccv15/%s.py' % model_name
    config_fn = 'models/iccv15/%s.conf' % model_name
    params_dir = 'weights/%s' % model_name

    # load depth network
    machine = net.create_machine(module_fn, config_fn, params_dir)

    # demo image
    rgb = Image.open('demo_nyud_rgb.jpg')
    rgb = rgb.resize((320, 240), Image.BICUBIC)

    # build depth inference function and run
    rgb_imgs = np.asarray(rgb).reshape((1, 240, 320, 3))
    (pred_depths, pred_normals) = machine.infer_depth_and_normals(rgb_imgs)

    # save prediction
    depth_img_np = depth_montage(pred_depths)
    #import pdb
    #pdb.set_trace()
    depth_img = Image.fromarray((255*depth_img_np).astype(np.uint8))
    depth_img.save('demo_depth_prediction.png')

    normals_img_np = normals_montage(pred_normals)
    normals_img = Image.fromarray((255*normals_img_np).astype(np.uint8))
    normals_img.save('demo_normals_prediction.png')


if __name__ == '__main__':
    main()

