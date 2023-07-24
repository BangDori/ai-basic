import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

loaded_data = np.loadtxt('./diabetes.csv', delimiter=',')

x_data = loaded_data[:, 0:-1] # 입력데이터 1~8열
t_data = loaded_data[:, [-1]] # 정답데이터 9열

print("loaded_data =", loaded_data.shape)
print("x_data =", x_data.shape, ", t_data =", t_data.shape)

# 확장성을 위하여 None으로 설정
X = tf.placeholder(tf.float32, [None, 8])
T = tf.placeholder(tf.float32, [None, 1])

# 가중치 노드 W는 행렬 곱을 위해 8X1로 정의
W = tf.Variable(tf.random_normal([8,1]))
b = tf.Variable(tf.random_normal([1]))

z = tf.matmul(X, W) + b # 선형회귀 값 z
y = tf.sigmoid(z) # 시그모이드로 계산 값

# 손실함수는 Cross-Entropy
loss = -tf.reduce_mean( T*tf.log(y) + (1-T)*tf.log(1-y) )

learning_rate = 0.01 # 학습율

# 경사 하강법 적용
optimizer = tf.train.GradientDescentOptimizer(learning_rate)

train = optimizer.minimize(loss)

# 시그모이드 값 y형상(shape)은 (759X8) * (8X1) = 759X1
# 즉, y > 0.5 라는 것은 759개의 모든 데이터에 대해 y > 0.5 비교하여 총 759개의 True 또는 False 리턴
predicted = tf.cast(y > 0.5, dtype=tf.float32)

# predicted와 T가 같으면 True, 아니면 False를 리턴하르모
# tf.cast를 이용하여 1또는 0으로 변환해서 총 759개의 1 또는 0을 가
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, T), dtype=tf.float32))

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer()) # 변수 노드(tf.Variable) 초기화

	for step in range(20001):
		loss_val, _ = sess.run([loss, train], feed_dict={X: x_data, T: t_data})

		if step % 500 == 0:
			print("step =", step, ", loss_val =", loss_val)

	# Accuracy 확인
	y_val, predicted_val, accuracy_val = sess.run([y, predicted, accuracy], feed_dict={X: x_data, T: t_data})

	print("\ny_val.shape =", y_val.shape, ", predicted_val =", predicted_val.shape)
	print("\nAccuracy =", accuracy_val)