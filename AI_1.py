import tensorflow as tf

텐서 = tf.constant([3, 4, 5])
텐서2 = tf.constant([6, 4, 5])
print(텐서 + 텐서2)  # 숫자한번에 더하기 스킬!
tf.add(텐서, 텐서2)
텐서4 = tf.zeros(10)  # 0으로 10개찬 텐서
print(텐서4)

텐서5 = tf.zeros([2, 2])
print(텐서5)

텐서6 = tf.zeros([2, 2, 3])  # 뒤에서 부터 해석 shape
print(텐서6)
print(텐서6.shape)

텐서3 = tf.constant([[1, 2],
                   [3, 4]])  # 행렬 4개 된거 생각하면됨

# dtype은 데이터의 타입을 의미한다. 하지만 실수 타입으로 표현하는게 중요함!
tf.cast(텐서3, float)  # 자료형 변환

w = tf.Variable(1.0)  # 딥러닝 상에서는 가중치 즉 w값 저장 항상 variable에 저장해됨
print(w.numpy()) #variable 저장된 값을 불러옴
w.assign(2) # w값 할당


