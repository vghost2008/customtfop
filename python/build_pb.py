#!/usr/bin/env python
#WorldSmallestNet

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from tensorflow.python.framework import graph_util
import tensorflow as tf
import numpy as np
import random, os, shutil
module_path = os.path.realpath(__file__)
module_dir = os.path.dirname(os.path.dirname(module_path))
lib_path = os.path.join(module_dir, "customop",'libcustomop.so')
print(lib_path)
mymodule = tf.load_op_library(lib_path)


#Define Graph
with tf.Graph().as_default():
    #Placeholder
    x = tf.placeholder(tf.float32,[None,1],name="input")
    v = x*2+10
    v = mymodule.my_set_value(v,[7],tf.constant([0]))
    output = tf.identity(v,"output");
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    print("Input:   | Output: ",sess.run(output,feed_dict={x:[[1],[2]]}))
    constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def,['output'])
    with tf.gfile.FastGFile("../output/frozen_graph.pb", mode='wb') as f:
        f.write(constant_graph.SerializeToString())
