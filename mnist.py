from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder("float", [None,10])

W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x,W) + b)

cross_entropy = -tf.reduce_sum(y_*tf.log(y))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

for i in range(2000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    if i%50 == 0:
        print (sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

for i in range(40000,40100):
    image = np.array(mnist.train.images[i])
    image.resize(1,784)
    print('预测的值',sess.run(tf.argmax(y,1),feed_dict={x: image}))
    print('真实的值',np.argmax(mnist.train.labels[i],0))
    
