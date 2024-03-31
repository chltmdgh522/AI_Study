import tensorflow as tf
import matplotlib.pyplot as plt

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()

# 정답데이터 즉 레이블
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# 구글에서 제공해주는 패션 데이터
# (train_images, train_labels), (test_images, test_labels) = 쇼핑몰 이미지 데이터, 6만개 정답 데이터 6만개

# print(train_images.shape) # (60000,28,28)

# print(test_labels)  # 9 2 1 티셔츠면 0

plt.imshow(train_images[1])
# plt.gray() # 흑백
plt.colorbar()  # 컬러
plt.show()
