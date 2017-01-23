from __future__ import print_function

import os, sys, shutil
import tensorflow as tf
import mnist_modeling.softmax as mnist

model_name = 'default'

model_path = 'model/' + model_name

if os.path.exists(model_path):
  shutil.rmtree(model_path)

os.makedirs(model_path)

s = tf.Session()
g = mnist.build_graph()
g = mnist.train(s, g)

print('Accuracy: %f' % mnist.accuracy(s, g))

mnist.save(s)
print('Saved model to model/')

mnist.visualize_model(s, g, save=True, path= model_path + '/model.png')
print('Model visualization saved to model/model.png')

s.close()
