from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import numpy as np

mnist = input_data.read_data_sets('./MNIST_data', one_hot=True)
plt.ion()
for i in range(100):
    plt.imshow(mnist.train.images[i].reshape((28, 28)), cmap='gray')
    plt.title('%i' % np.argmax(mnist.train.labels[i]))
    plt.show()
    plt.pause(1)
