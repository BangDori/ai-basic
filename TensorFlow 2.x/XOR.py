import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.optimizers import Adam

import numpy as np

print(tf.__version__)

# 데이터 생성
x_data = np.array([ [0, 0], [0, 1], [1, 0], [1, 1]])
t_data = np.array([[0], [1], [1], [0]])

# 모델 구축
model = Sequential()
model.add(Flatten(input_shape=(2,)))
model.add(Dense(2, activation='sigmoid'))
model.add(Dense(1, activation='sigmoid'))

# 모델 컴파일
model.compile(optimizer=Adam(learning_rate=0.1), loss='mse', metrics=['mse', 'binary_accuracy'])
model.summary()

# 모델 학습
hist = model.fit(x_data, t_data, epochs=500)

# 예측
result = model.predict([ [0,0], [0, 1], [1, 0], [1, 1] ])
print(result)