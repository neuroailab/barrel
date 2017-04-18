import tensorflow as tf
import numpy as np
from PIL import Image
import os

#tfrecords_filename = '/home/chengxuz/barrel/barrel_github/dataset/scenenet/tfrecords_depth/data_0.tfrecords'
#tfrecords_filename = '/home/chengxuz/barrel/barrel_github/dataset/scenenet/tfrecords_ins/data_0.tfrecords'
#tfrecords_filename = '/home/chengxuz/barrel/barrel_github/dataset/scenenet/tfrecords/data_0.tfrecords'
#tfrecords_filename = '/om/user/chengxuz/Data/one_world_dataset/split_scenenet/train_tfrecords/0/normal/data_0.tfrecords'
#tfrecords_filename = '/mnt/fs1/Dataset/scenenet_combine/normal/data_100_20.tfrecords'
#tfrecords_filename = '/mnt/fs1/Dataset/scenenet_combine/photo/data_100_20.tfrecords'
#tfrecords_filename = '/mnt/fs1/Dataset/scenenet_combine_val/normal/data_100_20.tfrecords'
#tfrecords_filename = '/mnt/fs1/Dataset/scenenet_combine_val/photo/data_100_20.tfrecords'
#tfrecords_filename = '/mnt/fs1/Dataset/scenenet/normal/data_10012.tfrecords'
#tfrecords_filename = '/mnt/fs1/Dataset/scenenet/photo/data_10012.tfrecords'
#tfrecords_filename = '/mnt/fs1/Dataset/scenenet/normal/data_112.tfrecords'
#tfrecords_filename = '/mnt/fs1/Dataset/scenenet/photo/data_112.tfrecords'
#tfrecords_filename = '/mnt/fs1/Dataset/scenenet/depth/data_112.tfrecords'
#tfrecords_filename = '/mnt/fs1/Dataset/scenenet_combine/normal_new/data_0_20.tfrecords'
tfrecords_filename = '/mnt/fs1/Dataset/scenenet_combine/photo/data_0_20.tfrecords'
reconstructed_images = []

record_iterator = tf.python_io.tf_record_iterator(path=tfrecords_filename)
get_num = 0

for string_record in record_iterator:
    
    example = tf.train.Example()
    example.ParseFromString(string_record)
    
    img_string = (example.features.feature['image_raw']
                                  .bytes_list
                                  .value[0])
    
    #img_1d = np.fromstring(img_string, dtype=np.uint16)
    img_1d = np.fromstring(img_string, dtype=np.uint8)
    reconstructed_img = img_1d.reshape((240, 320, -1))
    
    reconstructed_images.append(reconstructed_img)
    get_num = get_num + 1
    if get_num==2900:
        break
    #break

print(get_num)
img_save_prefix = '/mnt/fs1/chengxuz/scenenet_testout'
#img_inter = reconstructed_images[0]
#img_PIL = Image.fromarray(img_inter)
#img_PIL.save('normal_test_0.jpg')
for img_out_indx in xrange(10):
    img_inter = reconstructed_images[2800 + img_out_indx]
    #print(img_inter.shape)
    #img_inter = img_inter.astype(np.float)
    #img_inter = img_inter/np.max(img_inter)
    #img_inter = (img_inter*255).astype(np.uint8)
    #img_inter = img_inter.reshape([240, 320])
    img_PIL = Image.fromarray(img_inter)
    #img_PIL.save(os.path.join(img_save_prefix, 'normal_test_%i.jpg' % img_out_indx))
    img_PIL.save(os.path.join(img_save_prefix, 'photo_test_%i.jpg' % img_out_indx))
    #img_PIL.save(os.path.join(img_save_prefix, 'depth_test_%i.png' % img_out_indx))
    
#print(np.max(img_inter))
#print(np.sum(img_inter))
#print(np.min(img_inter))
#print(np.min(img_inter[100][104]))

#img_list = [0, 25, 50, 75, 100]
#img_filenames = ["/mnt/data2/chengxuz/train/0/0/depth/%i.png" % x for x in img_list]
#img_filenames = ["/mnt/data2/chengxuz/train/0/0/instance/%i.png" % x for x in img_list]
#img_filenames = ["/mnt/data2/chengxuz/train/0/0/photo/%i.jpg" % x for x in img_list]

#for img_indx, img_filename in enumerate(img_filenames):
#    recon_img = reconstructed_images[img_indx]

#    print(np.allclose(recon_img, np.array(Image.open(img_filename))))
