#定义变量及运算
import tensorflow as tf

var = tf.Variable(0)
#定义一个变量，初值为0

add_operation = tf.add(var, 1)
# add_operation = var + 1
update_operation = tf.assign(var, add_operation)
# var = add_operation 

init = tf.global_variables_initializer()
#定义了变量，必须对所有变量初始化

with tf.Session() as sess:
    
    sess.run(init)      #执行初始化
    for _ in range(3):
        sess.run(update_operation)
        print(sess.run(var))
