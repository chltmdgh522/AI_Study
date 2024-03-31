import tensorflow as tf

# 문제: 키와 신발사이즈는 어떤관련이 있을까?

키1 = [170, 180, 175, 160]
신발1 = [260, 270, 265, 255]
# 신발= a*키 + b


키 = tf.constant(키1, dtype=tf.float32)
신발 = tf.constant(신발1, dtype=tf.float32)

a = tf.Variable(0.1)
b = tf.Variable(0.1)


def loss_function():  # 손실함수는 그냥 실제값과 예측값의 차!
    return tf.square(신발 - (키 * a + b))  # 실제값, 예측값


opt = tf.keras.optimizers.Adam()  # 경사하강법을 이용해서 w를 업데이트!

# opt.minimize(loss_function, var_list=[a, b])  # 경사하강법 실행 var_list는 경사하강법을 통해서 바뀔 변수들

# 최적화 실행
for epoch in range(10000):  # 적절한 반복 횟수 선택
    with tf.GradientTape() as tape:
        loss = loss_function()
    gradients = tape.gradient(loss, [a, b])
    opt.apply_gradients(zip(gradients, [a, b]))

print(a.numpy(), b.numpy())

print("신발 사이즈: ", a.numpy() * 180 + b.numpy())
