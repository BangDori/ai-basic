import numpy as np
from datetime import datetime # datetime.now() 를 이용하여 학습 경과 시간 측정

training_data = np.loadtxt('./mnist_train.csv', delimiter=',', dtype=np.float32)
test_data = np.loadtxt('./mnist_test.csv', delimiter=',', dtype=np.float32)

def sigmoid(x):
	return 1 / (1 + np.exp(-x))

class NeuralNetwork:
	def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
		self.input_nodes = input_nodes
		self.hidden_nodes = hidden_nodes
		self.output_nodes = output_nodes

		# 은닉층 가중치 W2 = 784 X 100
		self.W2 = np.random.randn(self.input_nodes, self.hidden_nodes) / np.sqrt(self.input_nodes/2)
		self.b2 = np.random.rand(self.hidden_nodes)

		# 출력층 가중치는 W3 = 100 X 10
		self.W3 = np.random.randn(self.hidden_nodes, self.output_nodes) / np.sqrt(self.hidden_nodes/2)
		self.b3 = np.random.rand(self.output_nodes)

		# 출력층 선형회귀 값 Z3, 출력값 A3 정의
		self.Z3 = np.zeros([1, output_nodes])
		self.A3 = np.zeros([1, output_nodes])

		# 은닉층 선형회귀 값 Z2, 출력값 A2 정의
		self.Z2 = np.zeros([1, hidden_nodes])
		self.A2 = np.zeros([1, hidden_nodes])

		# 입력층 선형회기 값 Z1, 출력값 A1 정의
		self.Z1 = np.zeros([1, input_nodes])
		self.A1 = np.zeros([1, input_nodes])

		# 학습률 learning rate 초기화
		self.learning_rate = learning_rate
	
	def feed_forward(self):
		delta = 1e-7 # log 무한대 발산 방지

		# 입력층 선형회귀 값 Z1, 출력값 A1 계산
		self.Z1 = self.input_data
		self.A1 = self.input_data

		# 은닉층 선형회귀 값 Z2, 출력값 A2 계산
		self.Z2 = np.dot(self.A1, self.W2) + self.b2
		self.A2 = sigmoid(self.Z2)

		# 출력층 선형회귀 값 Z3, 출력값 A3 계산
		self.Z3 = np.dot(self.A2, self.W3) + self.b3
		self.A3 = sigmoid(self.Z3)

		return -np.sum( self.target_data*np.log(self.A3 + delta) + (1-self.target_data)*np.log((1-self.A3)+delta) )

	def loss_val(self):
		delta = 1e-7 # log 무한대 발산 방지

		# 입력층 선형회귀 값 Z1, 출력값 A1 계산
		self.Z1 = self.input_data
		self.A1 = self.input_data

		# 은닉층 선형회귀 값 Z2, 출력값 A2 계산
		self.Z2 = np.dot(self.A1, self.W2) + self.b2
		self.A2 = sigmoid(self.Z2)

		# 출력층 선형회귀 값 Z3, 출력값 A3 계산
		self.Z3 = np.dot(self.A2, self.W3) + self.b3
		self.A3 = sigmoid(self.Z3)

		return -np.sum( self.target_data*np.log(self.A3 + delta) + (1-self.target_data)*np.log((1-self.A3)+delta))
	
	def train(self, input_data, target_data): # input_data 784개, target_data 10개
		self.target_data = target_data
		self.input_data = input_data

		# 먼저 feed forward를 통해서 최종 출력값과 이를 바탕으로 현재의 에러 값 계산
		loss_val = self.feed_forward()

		# 출력층 loss 인 loss_3 구함
		loss_3 = (self.A3-self.target_data) * self.A3 * (1-self.A3)

		# 출력층 가중치 W3, 출력층 바이어스 b3 업데이트
		self.W3 = self.W3 - self.learning_rate * np.dot(self.A2.T, loss_3)
		self.b3 = self.b3 - self.learning_rate * loss_3

		# 은닉층 loss인 loss_2 구함
		loss_2 = np.dot(loss_3, self.W3.T) * self.A2 * (1-self.A2)

		# 은닉층 가중치 W2, 은닉층 바이어스 b2 업데이트
		self.W2 = self.W2 - self.learning_rate * np.dot(self.A1.T, loss_2)
		self.b2 = self.b2 - self.learning_rate * loss_2

	def predict(self, input_data): # input_data는 행렬로 입력됨 즉, (1, 784) shape을 가짐
		Z2 = np.dot(input_data, self.W2) + self.b2
		A2 = sigmoid(Z2)

		Z3 = np.dot(A2, self.W3) + self.b3
		A3 = sigmoid(Z3)

		predicted_num = np.argmax(A3)

		return predicted_num
	
	# 정확도 측정 함수
	def accuracy(self, test_data): # MNIST test_data (10,000 X 785)
		matched_list = []
		not_matched_list = []

		for index in range(len(test_data)):
			label = int(test_data[index, 0])

			# one-hot encoding을 위한 데이터 정규화 (data normalize)
			data = (test_data[index, 1:] / 255.0 * 0.99) + 0.01

			# predict를 위해서 vector을 matrix로 변환하여 인수로 넘겨줌
			predicted_num = self.predict(np.array(data, ndmin=2))

			if label == predicted_num:
				matched_list.append(index)
			else:
				not_matched_list.append(index)

		print("Current Accuracy =", 100*(len(matched_list) / len(test_data)), " %")
		return matched_list, not_matched_list
	
input_nodes = 784
hidden_nodes = 100
output_nodes = 10
learning_rate = 0.3
epochs = 1

nn = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

start_time = datetime.now()

for i in range(epochs):
	for step in range(len(training_data)): # train
		# input_data, target_data normalize
		target_data = np.zeros(output_nodes) + 0.01
		target_data[int(training_data[step, 0])] = 0.99

		input_data = ((training_data[step, 1:] / 255.0) * 0.99) + 0.01

		nn.train( np.array(input_data, ndmin=2), np.array(target_data, ndmin=2) )

		if step % 400 == 0:
			print("step=", step, ", loss_val=", nn.loss_val())

end_time = datetime.now()
print("\nelapsed time=", end_time - start_time)

nn.accuracy(test_data) # epochs == 1인 경우
nn.accuracy(test_data) # epochs == 5인 경우