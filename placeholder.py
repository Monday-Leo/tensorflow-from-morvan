#定义placeholder
import tensorflow as tf

input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)
#相当于两个没有值的变量，在session中需要输入值
output = tf.multiply(input1,input2)

with tf.Session() as sess:
    a = sess.run(output,feed_dict={input1:7,input2:8})
    #输入的值需要是字典类型
    print(a)
