from flask import abort, Flask, jsonify, request
import io
import numpy as np
from ..model import softmax as mnist
import png
import tensorflow as tf

app = Flask(__name__)

sess = tf.Session()
graph = mnist.load(sess)

ALLOWED_EXTENSION = ['png']

def allowed_file(filename):
  return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSION

@app.route('/mnist/classify', methods=['POST'])
def classify():
  if 'file' not in request.files:
    abort(404)

  file = request.files['file']

  if not allowed_file(file.filename):
    abort(404)

  #in_memory_file = io.BytesIO()
  #file.save(in_memory_file)
  r = png.Reader(file=file.stream)
  img = np.array(list(r.read()[2]))
  
  (row, col) = img.shape
  if row*col != 784:
    abort(404)

  img = img.reshape(1,784)

  prediction = mnist.predict(sess, graph, img)
  response = {'classification': prediction[0]}  
  return jsonify(**response)


@app.route('/mnist/classify-vector', methods=['POST'])
def classify_vector():
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
