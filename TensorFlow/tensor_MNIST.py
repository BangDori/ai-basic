import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow.examples.tutorials.mnist import input_data
tf.disable_v2_behavior()

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

print("\n", mnist.train.num_examples, mnist.test.num_examples, mnist.validation.num_examples)

print("\ntrain image shape =", np.shape(mnist.train.images))
print("train label shape =", np.shape(mnist.train.labels))
print("test image shape =", np.shape(mnist.test.images))
print("test label shape =", np.shape(mnist.test.labels))

learning_rate = 0.1 # 학습율
epochs = 100 # 반복 횟수
batch_size = 100 # 한번에 입력으로 주어지는 MNIST 개수

input_nodes = 784 # 입력노드 개수
hidden_nodes = 100 # 은닉노드 개수
output_nodes = 10 # 출력노드 개수

X = tf.placeholder(tf.float32, [None, input_nodes])
T = tf.placeholder(tf.float32, [None, output_nodes])

W2 = tf.Variable(tf.random_normal([input_nodes, hidden_nodes])) # 은닉층 가중치 노드
b2 = tf.Variable(tf.random_normal([hidden_nodes])) # 은닉층 바이어스 노드

W3 = tf.Variable(tf.random_normal([hidden_nodes, output_nodes])) # 출력층 가중치 노드
b3 = tf.Variable(tf.random_normal([output_nodes])) # 출력층 바이어스 노드

Z2 = tf.matmul(X, W2) + b2 # 선형회귀 값 Z2
A2 = tf.nn.relu(Z2) # 은닉층 출력 값 A2, sigmoid 대신 relu 사용

# 출력층 선형회귀 값 Z3, 즉, softmax에 들어가는 입력 값
Z3 = logits = tf.matmul(A2, W3) + b3

y = A3 = tf.nn.softmax(Z3)

# 출력층 선형회귀 값(logits) Z3와 정답 T를 이용하여 손실함수 크로스 엔트로피 계산
loss = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits_v2(logits=Z3, labels=T) )

optimizer = tf.train.GradientDescentOptimizer(learning_rate)

train = optimizer.minimize(loss)

# batch_size X 10 데이터에 대해 argmax를 통해 행단위로 비교
# 출력층의 계산 값 A3와 정답 T에 대해, 행 기준으로 값을 비교
predicted_val = tf.equal( tf.argmax(A3, 1), tf.argmax(T, 1) )

# batch_size X 10의 True, False를 1 또는 0으로 변환
accuracy = tf.reduce_mean(tf.cast(predicted_val, dtype=tf.float32))

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer()) # 변수 노드 (tf.Variable) 초기화

	for i in range(epochs): # 100번 반복수행
		total_batch = int(mnist.train.num_examples / batch_size) # 55,000 / 100
		for step in range(total_batch):
			batch_x_data, batch_t_data = mnist.train.next_batch(batch_size)

			loss_val, _ = sess.run([loss, train], feed_dict={X: batch_x_data, T: batch_t_data})
			if step % 100 == 0:
				print("step =", step, ", loss_val =", loss_val)

	# Accuracy 확인
	test_x_data = mnist.test.images # 10000 X 784
	test_t_data = mnist.test.labels # 10000 X 10

	accuracy_val = sess.run(accuracy, feed_dict={X: test_x_data, T: test_t_data})

	print("\nAccuracy =", accuracy_val)