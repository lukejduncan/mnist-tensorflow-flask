from __future__ import print_function

import argparse

parser = argparse.ArgumentParser(description='Train an mnist model')
parser.add_argument('--name', default='default')
args = parser.parse_args()

import os, sys, shutil
import tensorflow as tf
import mnist.model.softmax as mnist

model_name = args.name

model_path = 'model/' + model_name

if os.path.exists(model_path):
  shutil.rmtree(model_path)

os.makedirs(model_path)

s = tf.Session()
g = mnist.build_graph()
g = mnist.train(s, g)

print('Accuracy: %f' % mnist.accuracy(s, g))

mnist.save(s, model_name=model_name)
print('Saved model to '+ model_path)

mnist.visualize_model(s, g, save=True, path= model_path + '/model.png')
print('Model visualization saved to ' + model_path + '/model.png')

s.close()
