import tensorflow as tf
import numpy as np

#创造一个训练集
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data*0.2 + 0.3

#添加权重和偏差
Weights = tf.Variable(tf.random.uniform([1],-1.0,1.0))
biases = tf.Variable(tf.zeros(1))

y = x_data*Weights + biases

#计算损失函数
loss = tf.reduce_mean(tf.square(y-y_data))

#运动梯度下降法，学习速率为0.5，减少梯度
optimizer = tf.compat.v1.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

#将以上模型初始化
init = tf.compat.v1.global_variables_initializer()

#真正的开始初始化和训练
sess = tf.compat.v1.Session()
sess.run(init)

for step in range(200):
    sess.run(train)
    if step % 20 == 0:
        print(step,sess.run(Weights),sess.run(biases))
