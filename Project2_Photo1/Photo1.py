import tensorflow as tf
import matplotlib.pyplot as plt

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()

# 정답데이터 즉 레이블
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# 구글에서 제공해주는 패션 데이터
# (train_images, train_labels), (test_images, test_labels) = 쇼핑몰 이미지 데이터, 6만개 정답 데이터 6만개

# print(train_images.shape) # (60000,28,28)

# print(test_labels)  # 9 2 1 티셔츠면 0

# plt.imshow(train_images[1])
# plt.gray() # 흑백
# plt.colorbar()  # 컬러
# plt.show()

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, input_shape=(28, 28), activation="relu"),
    tf.keras.layers.Dense(128, activation="relu"),  # relu는 음수 다 0으로 만듬 convolution layer에서 자주씀
    tf.keras.layers.Flatten(), # 출력증 1D로 나열
    tf.keras.layers.Dense(10, activation="softmax"),  # 시그모이드랑 차이점은 binary 예측 문제에서 사용 즉 0인지 1인지 이거는 다름 즉 개인지 고양이인지
    # 즉 소프트 맥스는 여러개의 카테고리중 뭐가 높을지 예측 즉 어디 속할지  그리고 다 더하면 총합 1나옴
])

model.summary() #dense층에 inputshape 놓기 모델의 겉모양 보고 싶으면
exit()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])  # 이 loss는 정수로 인코딩 되어있을때 이거 사용
model.fit(train_images, train_labels, epochs=5)
