import os
import cv2
import numpy as np
import tensorflow as tf

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28,28)))
model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(units=10, activation=tf.nn.softmax))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=3)
model.save("AI_Model/mnist_model.keras")



model = tf.keras.models.load_model("AI_Model/mnist_model.keras")


loss, accuracy = model.evaluate(x_test, y_test)
print(loss)
print(accuracy)

image_directory = 'Test_Data'
image_files = [f for f in os.listdir(image_directory) if f.endswith('.png') or f.endswith('.jpg')]

for image_file in image_files:
    try:
        img_path = os.path.join(image_directory, image_file)
        img = cv2.imread(img_path)[:,:,0]
        img = np.invert(np.array([img]))
        prediction = model.predict(img)
        print(f"The number in {image_file} is probably a {np.argmax(prediction)}")
    except Exception as e:
        print(f"Error reading image {image_file}! Proceeding to the next one...")
        print(f"Error details: {e}")
