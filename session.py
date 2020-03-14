#session的作用

import tensorflow as tf

m1 = tf.constant([[2, 2]])  #创建常量矩阵
m2 = tf.constant([[3],
                  [3]])
dot_operation = tf.matmul(m1, m2)

print(dot_operation)  # 没有真正运算，打印出类型

# method1 use session
sess = tf.Session()
result = sess.run(dot_operation)
print(result)
#用run()来显示运算后的值
sess.close()

# method2 use session
with tf.Session() as sess:
    result_ = sess.run(dot_operation)
    print(result_)
