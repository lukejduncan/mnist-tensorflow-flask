from __future__ import print_function
import mnist.model.softmax as mnist
import numpy as np
import requests

imgs, _ = mnist.mnist_test_img.shape
idx = np.random.choice(range(imgs))

img = ','.join(['%d' % num for num in mnist.mnist_test_img[idx]])
query = {'img': img}

r = requests.post('http://localhost:5000/mnist/classify', data=query)
print("True Class: %d" % np.argmax(mnist.mnist_test_label[idx]))
print("HTTP Response:")
print(r.text)
