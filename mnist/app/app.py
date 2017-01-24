from flask import abort, Flask, jsonify, request
from ..model import softmax as mnist
import numpy as np
import tensorflow as tf

app = Flask(__name__)

sess = tf.Session()
graph = mnist.load(sess)

@app.route('/mnist/classify', methods=['POST'])
def classify():
  try:
    img = request.form['img'].split(',')
  except KeyError:
    abort(404)

  if len(img) != 784:
    abort(404)
 
  try:
    img = np.array(img).astype(np.int).reshape(1,784) 
  except ValueError:
    abort(404)

  prediction = mnist.predict(sess, graph, img)
  response = {'classification': prediction[0]}  
  return jsonify(**response)

if __name__ == '__main__':
  app.run(host='0.0.0.0', threaded=True)
