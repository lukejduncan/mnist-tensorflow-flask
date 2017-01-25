# MNIST classifier and web app

This project provides a simple MNIST classifer using TensorFlow and served via a
Flask webapp.

This is a linear softmax classifer very similar to the TensorFlow simple tutorial.  There are some
differences.  For example, the code is organized for reuse and with the intention that a framework
exists for writing different types of classifiers and allowing clients to swap them in seamlessly.

Additional tools are provided for visualizing the mnist dataset, visualizing the learned model, retraining
models, and simple querying of the HTTP end-point.

## Quick Start
In the repo is a pre-trained model that is ready for immediate use.  To do so you need to do a environment setup.

### Download the mnist data
```shell
$ ./setup.sh
```

### Setup your virtual environment:
```shell
$ virtualenv venv
$ source venv/bin/activate
$ pip install -r requirements.txt
```

### Run the webapp:
```shell
$ ./deploy.sh
```

### Query the endpoint

There are a few ways to query the end-point.  

#### Quick Query

To see results quickly, you can use the tool that randomly selects an image from the mnist test set and issues a query:
```shell
$ python query.py 
True Class: 3

HTTP Response:
{
  "classification": 3
}
```

#### Manual Query with Images

Assuming you have a 28 x 28 pixel png image you can classify it by posting to the API.  If you don't have an appropriate picture,
be sure to checkout /bin/mnist2png.py which will generate pngs for the mnist test set.

The following command shows a classification for the first image in the mnist test set.
```shell
$ curl -X POST -F 'file=@digits_test_sample_0.png' localhost:5000/mnist/classify
{
  "classification": 7
}
```


#### Manual Query With Vectors

Or if you prefer you can directly query it with curl.  The API expects 
a POST to /mnist/classify with a param named `img` that is a csv of a flattened mnist compatible
image.  Fair warning, the code below is > 784 characters and scrolls far to the right in github.

The following command shows a classification for the first image in the mnist test set.

```shell
$ curl -X POST -F 'img=0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,84,185,159,151,60,36,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,222,254,254,254,254,241,198,198,198,198,198,198,198,198,170,52,0,0,0,0,0,0,0,0,0,0,0,0,67,114,72,114,163,227,254,225,254,254,254,250,229,254,254,140,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,17,66,14,67,67,67,59,21,236,254,106,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,83,253,209,18,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,22,233,255,83,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,129,254,238,44,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,59,249,254,62,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,133,254,187,5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,9,205,248,58,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,126,254,182,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,75,251,240,57,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,19,221,254,166,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3,203,254,219,35,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,38,254,254,77,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,31,224,254,115,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,133,254,254,52,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,61,242,254,254,52,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,121,254,254,219,40,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,121,254,207,18,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0' localhost:5000/mnist/classify-vector
```

## Retraining the model
If you'd like to retrain the model you can do so using the provided train.py script.  Models are persisted as a collection 
of tensorflow files with a shared prefix.  After retraining you can find them at `model/<model_name>/*`.  You'll also 
find the visualization of the model at `model/<model_name>/model.png`.

To retrain run:
```shell
$ python train.py --name <model_name>
```

## Testing

To run the provided unit tests execute the following command.
```shell
$ ./test.sh
```
