import tensorflow as tf
import pandas as pd
import numpy as np

data = pd.read_csv('gpascore.csv')

# 전처리를 하는 이유는 공백 데이터를 채울려고 이다. 그래서 평균값 또는 행을 삭제하는 거를 통해서 공백을 채울 수 있음

print(data.isnull().sum())  # 빈칸 세줌
data.dropna(inplace=True)  # 빈칸 있는거 제거
# data.fillna() # 빈칸에 값 투입
# print(data['gpa'].min())  # gpa 최솟값 출력

y_data = data['admit'].values  # 리스트를 y_data에 싹 담아줌
x_data = []

for i, rows in data.iterrows():
    x_data.append([rows['gre'], rows['gpa'], rows['rank']])

# exit()  # 코드 잠깐 정지

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation='tanh'),  # 그림에서 보던 레이어  안에 숫자는 노드의 갯수  결과가 잘나올때까지 실험적으로 파악
    tf.keras.layers.Dense(128, activation='tanh'),
    tf.keras.layers.Dense(1, activation='sigmoid'),  # 이거는 결과층 결과는 1개 시그모이드는 모든 숫자를 0에서 1로 압축
    # 즉 확률로 해석
])  # 딥러닝 모델 디자인 하기

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# 확률일때 loss 함수는 binary_cross... 써야됨


model.fit(np.array(x_data), np.array(y_data), epochs=100000)  # 모델 학습 리스트를 넘파이 배열로 바꿔서 넣어야됨

# 예측
predict = model.predict(np.array([[800, 4, 1], [100, 1.5, 4],[800, 3.98, 1], [700, 3.78, 2]]))
print(predict)
