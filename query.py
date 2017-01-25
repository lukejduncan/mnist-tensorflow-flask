from __future__ import print_function

import idx2numpy, numpy, png
import mnist.model.softmax as mnist
import numpy as np
import os
import requests

imgs, _ = mnist.mnist_test_img.shape
idx = np.random.choice(range(imgs))

path = '/tmp/mnist-image.png'

png.from_array(mnist.mnist_test_img[idx].reshape(28, 28), 'L').save(path)

f = open(path, 'rb')
files = {'file': open(path, 'rb')}

r = requests.post('http://localhost:5000/mnist/classify', files=files)
print("True Class: %d" % np.argmax(mnist.mnist_test_label[idx]))
print("HTTP Response:")
print(r.text)

f.close()
os.remove(path)
