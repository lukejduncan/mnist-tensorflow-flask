from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import idx2numpy
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def reshape_labels(labels):
  reshape = np.zeros((len(labels), 10))

  for i in range(len(labels)):
    reshape[i][labels[i]] = 1
  
  return reshape

def reshape_imgs(img):
  count, width, height = img.shape
  return img.reshape(count, width * height)

def init(path_test_label='data/t10k-labels-idx1-ubyte', path_test_img='data/t10k-images-idx3-ubyte', path_train_label='data/train-labels-idx1-ubyte', path_train_img='data/train-images-idx3-ubyte'):
  global mnist_train_img, mnist_train_label, mnist_test_img, mnist_test_label
  mnist_train_img = reshape_imgs(idx2numpy.convert_from_file(path_train_img))
  mnist_train_label = reshape_labels(idx2numpy.convert_from_file(path_train_label))
  mnist_test_img = reshape_imgs(idx2numpy.convert_from_file(path_test_img))
  mnist_test_label = reshape_labels(idx2numpy.convert_from_file(path_test_label))

  return tf.Session()

def build_graph(learning_rate = 0.5):
  # Create a linear model that is minimizes cross_entropy
  # via gradient descent with a softmax output layer
  # This is similar to logistic regression, except with logistic
  # regression we'd need a model per class.  Softmax allows us to
  # do multi-class classification by normalizing exponentiated inputs
  # resulting in a probability distribution across n-classes, where n
  # is the width of the output layer
  
  # y_hat = xW + b
  # where x is an input parameter of 784 features (pixels in image) and 
  # variable number of examples, W is a matrix of learned coefficients 
  # and b is a vector of bias terms for eclass
  
  x = tf.placeholder(tf.float32, [None, 784], name='x')
  W = tf.Variable(tf.zeros([784, 10]), name='W')
  b = tf.Variable(tf.zeros([10]), name='b')
  y = tf.add(tf.matmul(x, W), b, name='line')
  
  # y_ are the true labels of the training data
  # used for the loss function
  y_ = tf.placeholder(tf.float32, [None,10], name='y_')
  
  loss_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

  # Train the model using gradient decent minimizing the loss function
  # Tensorflow implemenets this via symbol-to-symbol back propagation, appending
  # gradient nodes up the DAG rather than directly computing the gradient
  train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss_function)
 
  # Measure the accuracy of the model
  correct = tf.equal(tf.argmax(y,1), tf.argmax(y_, 1))
  accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name='accuracy')

  return tf.get_default_graph()

def train(sess, graph, batch_size=100, epochs=1000):
  global mnist_train_img, mnist_train_label, mnist_test_img, mnist_test_label
  
  x = graph.get_tensor_by_name('x:0')
  y_ = graph.get_tensor_by_name('y_:0')
  train_step = graph.get_operation_by_name('GradientDescent')

  sess.run(tf.global_variables_initializer())
  
  # Training Loop
  for _ in range(epochs):
    # Stochastic training to make training faster
    batch_idx = np.random.randint(0, len(mnist_train_img), batch_size)
    batch_img = [mnist_train_img[i] for i in batch_idx]
    batch_label = [mnist_train_label[i] for i in batch_idx]
    
    sess.run(train_step, feed_dict={x: batch_img, y_: batch_label})
  
  return tf.get_default_graph()

def get_accuracy(sess, graph):
  x = graph.get_tensor_by_name('x:0')
  y_ = graph.get_tensor_by_name('y_:0')
  accuracy = graph.get_tensor_by_name('accuracy:0')
  
  print("Accuracy: %f" % sess.run(accuracy, feed_dict={x: mnist_test_img, y_: mnist_test_label}))

def close(sess):
  sess.close()
