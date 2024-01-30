from keras.layers import InputLayer, Conv2D, UpSampling2D
from keras.models import Sequential
from keras.preprocessing.image import array_to_img
from keras.utils import plot_model
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
from sklearn.model_selection import train_test_split


def create_bw_images():
    color_images_folder = 'landscape_images'
    bw_images_folder = 'bw_images'
    os.makedirs(bw_images_folder, exist_ok=True)

    color_images = []
    bw_images = []
    for filename in os.listdir(color_images_folder):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            color_img = cv2.imread(os.path.join(color_images_folder, filename))
            color_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)
            color_img = cv2.resize(color_img, (256, 256))

            bw_image = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
            output_path = os.path.join(bw_images_folder, filename)
            cv2.imwrite(output_path, bw_image)

            color_images.append(color_img)
            bw_images.append(bw_image)

    return bw_images, color_images


def plot_images(original, decoded, n=6):
    plt.figure(figsize=(12, 4))
    for i in range(n):
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(original[i].reshape(256, 256), cmap='gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(decoded[i].reshape(256, 256, 3))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()


bw_images, color_images = create_bw_images()

print("Wczytano kolorowe obrazki i utworzono czarnobiałe: ", len(color_images))

height, width = 256, 256

bw_images_np = np.array(bw_images).reshape(-1, height, width, 1).astype('float32') / 255.0
color_images_np = np.array(color_images).reshape(-1, height, width, 3).astype('float32') / 255.0

X_bw_train, X_bw_test, X_color_train, X_color_test = train_test_split(bw_images_np, color_images_np, test_size=0.2,
                                                                      random_state=42)

model = Sequential()

# Enkoder
model.add(InputLayer(input_shape=(None, None, 1)))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same', strides=2))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))

# Dekoder
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(3, (3, 3), activation='sigmoid', padding='same'))


model.compile(optimizer='adam', loss='mean_squared_error')

model.fit(X_bw_train, X_color_train, validation_data=(X_bw_test, X_color_test), epochs=30,
          batch_size=32)

decoded_imgs = model.predict(X_bw_test)

# tanh
#decoded_imgs = (decoded_imgs + 1.0) / 2.0

os.makedirs("decoded_images", exist_ok=True)

for i, img_array in enumerate(decoded_imgs):
    img = array_to_img(img_array)
    img.save(os.path.join("decoded_images", f'decoded_img_{i}.png'))


plot_images(X_bw_test, decoded_imgs)

