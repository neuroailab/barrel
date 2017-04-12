import tensorflow as tf
import numpy as np
from PIL import Image
import os

#tfrecords_filename = '/home/chengxuz/barrel/barrel_github/dataset/scenenet/tfrecords_depth/data_0.tfrecords'
#tfrecords_filename = '/home/chengxuz/barrel/barrel_github/dataset/scenenet/tfrecords_ins/data_0.tfrecords'
#tfrecords_filename = '/home/chengxuz/barrel/barrel_github/dataset/scenenet/tfrecords/data_0.tfrecords'
tfrecords_filename = '/om/user/chengxuz/Data/one_world_dataset/split_scenenet/train_tfrecords/0/normal/data_0.tfrecords'
reconstructed_images = []

record_iterator = tf.python_io.tf_record_iterator(path=tfrecords_filename)

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
    #break

img_save_prefix = '/om/user/chengxuz/Data/one_world_dataset/split_scenenet/normal_outs/'
#img_inter = reconstructed_images[0]
#img_PIL = Image.fromarray(img_inter)
#img_PIL.save('normal_test_0.jpg')
for img_out_indx in xrange(10):
    img_inter = reconstructed_images[img_out_indx]
    img_PIL = Image.fromarray(img_inter)
    img_PIL.save(os.path.join(img_save_prefix, 'normal_test_%i.jpg' % img_out_indx))
    
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
