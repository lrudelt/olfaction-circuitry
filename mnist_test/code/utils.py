import numpy as np
from keras.datasets import mnist
import skimage
from tensorflow._api.v2.image import resize

def create_mnist_inputs(num_inputs, s, number_bound=3, train=True):
    (train_X, train_y), (test_X, test_y) = mnist.load_data()
    image_size = 28
    n = int(np.sqrt(s.n_stim))
    if train:
        images, labels = train_X, train_y
    else:
        images, labels = test_X, test_y
    input = np.zeros((num_inputs, n*n))
    no_images = len(images)
    images_used = 0
    counter = 0
    while images_used < num_inputs:
        if counter >= no_images:
            counter = 0
        if labels[counter] >= number_bound:
            counter += 1
            continue
        image = images[counter].reshape((image_size, image_size))
        #downsized = skimage.transform.resize(image, (n,n))
        downsized = np.array(resize(image[..., np.newaxis], [n,n]))
        input[images_used] = downsized.flatten()/255 ## original downsized in 0-255
        images_used += 1
        counter += 1
    return input

def fade(cur, new, t, length):
    if t > 1-length:
        return new*(1-(1-t)/length) + cur*(1-t)/length
    else:
        return cur
    

def fade_images(inputs, t, s):
    l = s.presentation_length
    fade_length = s.fade_fraction
    image = int(t/l)
    inp = inputs[image]
    new = inputs[min(image + 1, len(inputs)-1)]
    return fade(inp, new, (t % l) / l, fade_length)