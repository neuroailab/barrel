from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import sys

sys.path.append('../normal_pred/')
import normal_encoder_asymmetric_with_bypass

def combine_normal_tfutils(inputs, **kwargs):
    #print(inputs.keys())
    #print(inputs['image_t'].get_shape().as_list(), inputs['image_t'].dtype)
    inputs_t = tf.cast(inputs['image_t'], tf.float32)
    with tf.variable_scope("create"):
        m_t = normal_encoder_asymmetric_with_bypass.normal_vgg16(inputs_t, **kwargs)

    inputs_s = tf.cast(inputs['image_s'], tf.float32)
    with tf.variable_scope("create", reuse=True):
        m_s = normal_encoder_asymmetric_with_bypass.normal_vgg16(inputs_s, **kwargs)

    return [m_t.output, m_s.output], m_s.params
