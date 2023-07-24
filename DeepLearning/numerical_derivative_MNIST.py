import numpy as np

training_data = np.loadtxt('./mnist_train.csv', delimiter=',', dtype=np.float32)
test_data = np.loadtxt('./mnist_test.csv', delimiter=',', dtype=np.float32)

print("training_data.shape=", training_data.shape, ", test_data.shape=", test_data.shape)

# sigmoid 함수
def sigmoid(x):
	return 1 / (1 + np.exp(-x))

# 수치미분 함수
def numerical_derivative(f, x):
	delta_x = 1e-4 # 0.001
	grad = np.zeros_like(x)

	it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])

	while not it.finished:
		idx = it.multi_index
		tmp_val = x[idx]

		x[idx] = float(tmp_val) + delta_x
		fx1 = f(x) # f(x+delta_x)

		x[idx] = tmp_val - delta_x
		fx2 = f(x) # f(x-delta_x)

		grad[idx] = (fx1 - fx2) / (2 * delta_x)

		x[idx] = tmp_val
		it.iternext()

	return grad

class NeuralNetwork:

	# 생성자
	def __init__(self, input_nodes, hidden_nodes, output_nodes):
		self.input_nodes = input_nodes
		self.hidden_nodes = hidden_nodes
		self.output_nodes = output_nodes

		# 2층 hidden layer unit
		# 가중치 W, 바이어스 b 초기화
		self.W2 = np.random.rand(self.input_nodes, self.hidden_nodes) # W2 = (784X100)
		self.b2 = np.random.rand(self.hidden_nodes) # b2 = (100,)

		# 3층 output layer unit
		self.W3 = np.random.rand(self.hidden_nodes, self.output_nodes) # W3 = (100X10)
		self.b3 = np.random.rand(self.output_nodes) # b3 = (10,)

		# 학습률 learning rate 초기화
		self.learning_rate = 1e-4

	# feed forward를 이용하여 입력층에서 부터 출력층까지 데이터를 전달하고 손실 함수 값 계산
	# loss_val(self) 메서드와 동일한 코드. loss_val(self)은 외부 출력용으로 사용
	def feed_forward(self):
		delta = 1e-7 # log 무한대 발산 방지

		z1 = np.dot(self.input_data, self.W2) + self.b2
		y1 = sigmoid(z1)

		z2 = np.dot(y1, self.W3) + self.b3
		y = sigmoid(z2)

		# cross-entropy
		return -np.sum( self.target_data*np.log(y+delta) + (1-self.target_data)*np.log((1-y)+delta) )

	# 손실 값 계산
	def loss_val(self):
		delta = 1e-7    # log 무한대 발산 방지

		z1 = np.dot(self.input_data, self.W2) + self.b2
		y1 = sigmoid(z1)
		
		z2 = np.dot(y1, self.W3) + self.b3
		y = sigmoid(z2)
		
		# cross-entropy 
		return  -np.sum( self.target_data*np.log(y + delta) + (1-self.target_data)*np.log((1 - y)+delta ) )
	
    	# input_data : 784개, traget_data : 10개
	def train(self, training_data):
		# normalize
		self.target_data = np.zeros(output_nodes) + 0.01 # one-hot encoding을 위한 10개의 노드 0.01 초기화
		self.target_data[int(training_data[0])] = 0.99 # 정답을 나타내는 인덱스에 가장 큰 값인 0.99로 초기화

		# 입력 데이터는 0~255이기 때문에, 가끔 overflow 발생
		# 따라서, 모든 입력 값을 0~1 사이의 값으로 nomalize 함
		self.input_data = (training_data[1:] / 255.0 * 0.99) + 0.0

		f = lambda x : self.feed_forward()

		self.W2 -= self.learning_rate * numerical_derivative(f, self.W2)
		self.b2 -= self.learning_rate * numerical_derivative(f, self.b2)
		self.W3 -= self.learning_rate * numerical_derivative(f, self.W3)
		self.b3 -= self.learning_rate * numerical_derivative(f, self.b3)

	# query, 즉 미래 값 예측 함수
	def predict(self, input_data):
		z1 = np.dot(input_data, self.W2) + self.b2
		y1 = sigmoid(z1)

		z2 = np.dot(y1, self.W3) + self.b3
		y = sigmoid(z2)

		# 가장 큰 값을 가지는 인덱스를 정답으로 인식(argmax)
		# 즉, one-hot encoding을 구현
		predicted_num = np.argmax(y)

		return predicted_num
	
    	# 정확도 측정 함수
	def accuracy(self, test_data):
		matched_list = []
		not_matched_list = []
	
		for index in range(len(test_data)):
			label = int(test_data[index, 0]) # Test_data의 1열에 있는 정답 분리
	
			# normalize
			data = (test_data[index, 1:] / 255.0 * 0.99) + 0.01
	
			predicted_num = self.predict(data)
	
			if label == predicted_num: # 정답과 예측 값이 맞으면 matched_list에 추가
				matched_list.append(index)
			else: # 정답과 예측 값이 틀리면 not_matched_list에 추가
				not_matched_list.append(index)
	
		# 정확도 계산 (정답 데이터 / 전체 테스트 데이터)
		print("Current Accuracy=", 100*(len(matched_list)/(len(test_data))), " %")
	
		return matched_list, not_matched_list
	

input_nodes = 784
hidden_nodes = 100
output_nodes = 10

nn = NeuralNetwork(input_nodes, hidden_nodes, output_nodes)

for step in range(30001): # 전체 training data 50%
	
	# 총 60,000개의 training data 가운데 random하게 30,000개 선택
	index = np.random.randint(0, len(training_data)-1)

	nn.train(training_data[index])

	if step % 400 == 0:
		print("step=", step, ", loss_val=", nn.loss_val())