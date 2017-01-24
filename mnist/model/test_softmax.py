from __future__ import print_function

import os
import shutil
import softmax as mnist
import tensorflow as tf
import time
import unittest

class TestSoftmax(unittest.TestCase):

  def test_can_train_model(self):
    s = tf.Session()
    g = mnist.build_graph()
    g = mnist.train(s, g)

    self.assertTrue(mnist.accuracy(s, g) > 0.8)
    s.close()

  def test_can_load_model(self):
    s = tf.Session()
    g = mnist.load(s)

    self.assertTrue(mnist.accuracy(s, g) > 0.8)
    s.close()

  def test_can_save_model(self):
    s = tf.Session()
    g = mnist.build_graph()
    g = mnist.train(s, g)
    
    path = '/tmp/test_softmax/%f' % time.time()
    model_name = 'test'
    full_path = path + '/' + model_name
    os.makedirs(full_path)

    mnist.save(s, path=path, model_name=model_name)

    self.assertTrue(os.path.exists(full_path + '/model.meta'))
    self.assertTrue(os.path.exists(full_path + '/checkpoint'))
    shutil.rmtree(path)
