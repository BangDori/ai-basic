import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from datetime import datetime # datetime.now()를 이용하여 학습 경과 시간 측정

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

print("\n", mnist.train.num_examples, mnist.test.num_examples, mnist.validation.num_examples)

print("\ntrain image shape =", np.shape(mnist.train.images))
print("train label shape =", np.shape(mnist.train.labels))
print("test image shape =", np.shae(mnist.test.images))
print("test label shape =", np.shape(mnist.test.labels))

# Hyper-Parameter
learning_rate = 0.001 # 학습율
epochs = 30 # 반복횟수
batch_size = 100 # 한번에 입력으로 주어지는 MNIST 개수

# 입력과 정답을 위한 플레이스홀더 정의
X = tf.placeholder(tf.float32, [None, 784])

T = tf.placeholder(tf.float32, [None, 10])

# 입력층의 출력 값, 컨볼루션 연산을 위해 reshape 시킴
A1 = X_img = tf.reshape(X, [-1, 28, 28, 1]) # image 28 X 28 X 1 (black/white)

# 1번째 컨볼루션 층
# 3X3 크기를 가지는 32개의 필터를 적용

F2 = tf.Variable(tf.random_normal([3, 3, 1, 32], stddev=0.01)) # stddev 표준편차
b2 = tf.Variable(tf.constant(0.1, shape=[32]))

# 1번째 컨볼루션 연산을 통해 28 X 28 X 1 => 28 X 28 X 32
C2 = tf.nn.conv2d(A1, F2, strides=[1, 1, 1, 1], padding='SAME')

# relu
Z2 = tf.nn.relu(C2+b2)

# 1번째 max pooling을 통해 28 X 28 X 32 => 14 X 14 X 32
A2 = P2 = tf.nn.max_pool(Z2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# 2번째 컨볼루션 층
# 3X3 크기를 가지는 64개의 필터를 적용
F3 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01))
b3 = tf.Variable(tf.constant(0.1, shape=[64]))

# 2번째 컨볼루션 연산을 통해 14 X 14 X 32 => 14 X 14 X 64
C3 = tf.nn.conv2d(A2, F3, strides=[1, 1, 1, 1], padding='SAME')

# relu
Z3 = tf.nn.relu(C3+b3)

# 2번째 max pooling을 통해 14 X 14 X 64 => 7 X 7 X 64
A3 = P3 = tf.nn.max_pool(Z3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# 3번째 컨볼루션 층
# 3X3 크기를 가지는 128개의 필터를 적용
F4 = tf.Variable(tf.random_normal([3, 3, 64, 128], stddev=0.01))
b4 = tf.Variable(tf.constant(0.1, shape=[128]))

# 3번째 컨볼루션 연산을 통해 7 X 7 X 64 => 7 X 7 X 128
C4 = tf.nn.conv2d(A3, F4, strides=[1, 1, 1, 1], padding='SAME')

# relu
Z4 = tf.nn.relu(C4+b4)

# 3번째 max pooling을 통해 7 X 7 X 128 => 4 X 4 X 128
A4 = P4 = tf.nn.max_pool(Z4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# 완전연결층

# 4X4 크기를 가진 128개의 activation map을 flatten 시킴
# 2,048개 (128*4*4) 노드를 가지도록 reshape 시켜줌.
# 즉 완전연결층(2048개 노드)과 출력층(10개 노드)은 2048 X 10개 노드가 완전연결 되어 있음
A4_flat = P4_flat = tf.reshape(A4, [-1, 128*4*4])

# 출력층
W5 = tf.Variable(tf.random_normal([128*4*4, 10], stddev=0.01))
b5 = tf.Variable(tf.random_normal([10]))

# 출력층 선형회귀 값 Z5, 즉 softmax에 들어가는 입력 값
Z5 = logits = tf.matmul(A4_flat, W5) + b5 # 선형회귀 값 Z5

y = A5 = tf.nn.softmax(Z5)

# 출력층 선형회귀 값(logits) Z5와 정답 T를 이용하여 손실함수 크로스 엔트로피 계산
loss = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits_v2(logits=Z5, labels=T) )

# 성능 개선을 위해 AdamOptimizer 사용
optimizer = tf.train.AdamOptimzer(learning_rate)

train = optimizer.minimize(loss)

# batch_size X 10 데이터에 대해 argmax를 통해 행단위로 비교
# 출력층의 계산 값 A5와 정답 T에 대해, 행 단위로 값을 비교함
predicted_val = tf.equal( tf.argmax(A5, 1), tf.argmax(T, 1) )

# batch_size X 10의 True, False를 1또는 0으로 변환
accuracy = tf.reduce_mean(tf.cast(predicted_val, dtype=tf.float32))

with tf.Session as sess:
	sess.run(tf.global_variables_initializer()) # 변수 노드(tf.Variable) 초기화
	start_time = datetime.now()

	for i in range(epochs): # 30번 반복수행
		total_batch = int(mnist.train.num_examples / batch_size) # 55,000 / 100

		for step in range(total_batch):
			batch_x_data, batch_t_data = mnist.train.next_batch(batch_size)
			loss_val, _ = sess.run([loss, train], feed_dict={X: batch_x_data, T: batch_t_data})

			if step % 100 == 0:
				print("epochs =", i, ", step =", step, ", loss_val =", loss_val)

	end_time = datetime.now()
	print("\nelapsed time =", end_time - start_time)

	# Accuracy 확인
	test_x_data = mnist.test.images # 10000 * 784
	test_t_data = mnist.test.labels # 10000 * 10

	accuracy_val = sess.run(accuracy, feed_dict={X: test_x_data, T: test_t_data})
	print("\nAccuracy =", accuracy_val)