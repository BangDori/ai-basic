import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

loaded_data = np.loadtxt('./data-01.csv', delimiter=',')

x_data = loaded_data[:, 0:-1]
t_data = loaded_data[:, [-1]]

print("x_data.shape =", x_data.shape)
print("t_data.shape =", t_data.shape)

W = tf.Variable(tf.random_normal([3, 1]))
b = tf.Variable(tf.random_normal([1]))

X = tf.placeholder(tf.float32, [None, 3])
T = tf.placeholder(tf.float32, [None, 1])

y = tf.matmul(X, W) + b # 현재 X, W, b, 를 바탕으로 계산된 값

loss = tf.reduce_mean(tf.square(y - T)) # MSE 손실함수 정의

learning_rate = 1e-5 # 학습율

optimizer = tf.train.GradientDescentOptimizer(learning_rate)

train = optimizer.minimize(loss)

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer()) # 변수 노드(tf.Variable) 초기화

	for step in range(8001):
		loss_val, y_val, _ = sess.run([loss, y, train], feed_dict={X: x_data, T: t_data})

		if step % 400 == 0:
			print("step =", step, ", loss_val =", loss_val)

	print("\nPrediction is ", sess.run(y, feed_dict={X: [ [100, 98, 81] ]}))