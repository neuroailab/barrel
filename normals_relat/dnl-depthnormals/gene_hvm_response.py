import os
import sys
import numpy as np
from scipy import misc
import dldata.stimulus_sets.hvm as nd; reload(nd)

from PIL import Image

import argparse
import net
from utils import depth_montage, normals_montage
import pdb

import h5py

def get_network(model_name):
    module_fn = 'models/iccv15/%s.py' % model_name
    config_fn = 'models/iccv15/%s.conf' % model_name
    #params_dir = 'weights/%s' % model_name
    params_dir = 'weights/iccv15/%s' % model_name

    # load depth network
    machine = net.create_machine(module_fn, config_fn, params_dir)
    return machine

def main():
    parser = argparse.ArgumentParser(description='The script to generate responses of hvmdataset')
    parser.add_argument('--network', default = 0, type = int, action = 'store', help = '0 is alexnet, 1 is vgg')
    parser.add_argument('--layer', default = "1.1", type = str, action = 'store', help = 'which layer to extract')
    parser.add_argument('--savedir', default = "/mnt/data/chengxuz/barrel/hvm_responses", type = str, action = 'store', help = 'where to store the file')
    parser.add_argument('--saveprefix', default = "hvm_layer_", type = str, action = 'store', help = 'Prefix of saving file')
    parser.add_argument('--testmode', default = 0, type = int, action = 'store', help = 'default is 0, 1 means generating images for index 0')
    parser.add_argument('--resizemethod', default = 0, type = int, action = 'store', help = 'default is 0, doing the resize, otherwise, just pad with zeros')

    args    = parser.parse_args()

    if args.network==0:
        model_name = 'depthnormals_nyud_alexnet'
    else:
        model_name = 'depthnormals_nyud_vgg'

    machine = get_network(model_name)

    dataset = nd.HvMWithDiscfade()
    meta = dataset.meta
    imgs = dataset.get_images(preproc={'resize_to': (256,256), 'dtype': 'float32', 'mode':'L', 'normalize': False})

    num_imgs, _, _ = imgs.shape

    want_out_w = 240
    want_out_h = 320
    num_channel = 3

    if args.testmode==1:
        #num_imgs = 1
        #num_imgs = 50
        num_imgs = 5760
    big_img_array = np.zeros([num_imgs, want_out_w, want_out_h, num_channel])

    for indx_img in xrange(num_imgs):
        if (indx_img % 100 ==0):
            print("Now img:%i" % indx_img)

        now_img = np.asarray(imgs[indx_img])

        if args.resizemethod==0:
            #print(now_img.shape)
            now_img = now_img * 255
            now_img = now_img.astype(np.uint8)
            now_img = misc.imresize(now_img, [want_out_w, want_out_h])
            big_img_array[indx_img, :, :, 0] = now_img
            big_img_array[indx_img, :, :, 1] = now_img
            big_img_array[indx_img, :, :, 2] = now_img
        else:
            print("Not implemented yet!")
            exit()
            pass

    #big_img_array   = 255 - big_img_array # needed to make the images right
    big_img_array   = big_img_array.astype(np.float32)

    if args.testmode==0:
        pred_depths, pred_normals = machine.infer_some_layer_depth_and_normals(big_img_array, args.layer)
        #pdb.set_trace()
        if pred_normals is None:
            save_filename = os.path.join(args.savedir, "%s%s.hdf5" % (args.saveprefix, args.layer))
            fin = h5py.File(save_filename, 'a')
            if 'data' in fin:
                del fin['data']
            dset = fin.create_dataset('data', pred_depths.shape, dtype=pred_depths.dtype)
            dset[...] = pred_depths
            fin.close()
        else:
            save_filename = os.path.join(args.savedir, "%s%s_depths.hdf5" % (args.saveprefix, args.layer))
            fin = h5py.File(save_filename, 'a')
            if 'data' in fin:
                del fin['data']
            dset = fin.create_dataset('data', pred_depths.shape, dtype=pred_depths.dtype)
            dset[...] = pred_depths
            fin.close()

            save_filename = os.path.join(args.savedir, "%s%s_normals.hdf5" % (args.saveprefix, args.layer))
            fin = h5py.File(save_filename, 'a')
            if 'data' in fin:
                del fin['data']
            dset = fin.create_dataset('data', pred_normals.shape, dtype=pred_normals.dtype)
            dset[...] = pred_normals
            fin.close()
    else:
        (pred_depths, pred_normals) = machine.infer_depth_and_normals(big_img_array)
        for img_indx in xrange(num_imgs):
            depth_tmp = pred_depths[img_indx, :, :].reshape([1, 109, 147])
            normal_tmp = pred_normals[img_indx, :, :, :].reshape([1,3, 109, 147])
            depth_img_np = depth_montage(depth_tmp)
            normals_img_np = normals_montage(normal_tmp)

            #print(depth_img_np.shape)
            # save prediction
            depth_img = Image.fromarray((255*depth_img_np).astype(np.uint8))
            depth_img.save('hvm_response_depth/depth_prediction_hvm_%i.png' % img_indx)

            normals_img = Image.fromarray((255*normals_img_np).astype(np.uint8))
            normals_img.save('hvm_response_normals/normals_prediction_hvm_%i.png' % img_indx)

            original_img = Image.fromarray(big_img_array[img_indx].astype(np.uint8))
            original_img.save('hvm_orig/hvm_test_image_%i.png' % img_indx)

    pass

if __name__=="__main__":
    main()

