import numpy as np

loaded_data = np.loadtxt('./data-01-test-score.csv', delimiter=',', dtype=np.float32)

x_data = loaded_data[:, 0:-1]
t_data = loaded_data[:, [-1]]

W = np.random.rand(3, 1) # 3X1 행렬
b = np.random.rand(1)
print("W=", W, ", W.shape=", W.shape, ", b=", b, ", b.shape=", b.shape)

def loss_func(x, t):
	y = np.dot(x, W) + b

	return ( np.sum( (t-y)**2 ) ) / ( len(x) )

def numerical_derivative(f, x):
	delta_x = 1e-4 # 0.000
	grad = np.zeros_like(x)

	it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])

	while not it.finished:
		idx = it.multi_index
		
		tmp_val = x[idx]
		x[idx] = float(tmp_val) + delta_x
		fx1 = f(x) # f(x+delta_x)

		x[idx] = tmp_val - delta_x
		fx2 = f(x) # f(x-delta_x)
		grad[idx] = (fx1 - fx2) / (2*delta_x)

		x[idx] = tmp_val
		it.iternext()

	return grad

# 손실함수 값 계산 함수
# 입력변수 x, t : numpy type
def error_val(x, t):
	y = np.dot(x, W) + b

	return ( np.sum((t-y)**2) ) / ( len(x) )

# 학습을 마친 후, 임의의 데이터에 대해 미래 값 예측 함수
# 입력변수 x : numpy type
def predict(x):
	y = np.dot(x, W) + b

	return y

learning_rate = 1e-5 # 1e-2, 1e-3 은 손실함수 값 발산

f = lambda x : loss_func(x_data, t_data) # f(x) = loss_func(x_data, t_data)

print('initial error value=', error_val(x_data, t_data), "\ninitial W=", W, ", b=", b)

for step in range(8001):
	W -= learning_rate * numerical_derivative(f, W)
	b -= learning_rate * numerical_derivative(f, b)
		
test_data = np.array([100, 98, 81])
print("predict:", predict(test_data)) # [100, 98, 81] 의 미래값 예측