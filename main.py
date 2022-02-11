import tensorflow as tf
from deepface import DeepFace

print("TensorFlow version:", tf.__version__)
# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import cv2

print("Open CV Version: ", cv2.__version__)


def TenserFlow():
    fashion_mnist = tf.keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    len(train_labels)
    len(test_labels)
    plt.figure()
    plt.imshow(train_images[0])
    plt.colorbar()
    plt.grid(False)
    plt.show()

    train_images = train_images / 255.0

    test_images = test_images / 255.0

    plt.figure(figsize=(10, 10))
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(train_images[i], cmap=plt.cm.binary)
        plt.xlabel(class_names[train_labels[i]])
    plt.show()

    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10)
    ])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    model.fit(train_images, train_labels, epochs=10)

    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)

    print('\nTest accuracy:', test_acc)


def openCV():
    # Load the cascade
    face_cascade = cv2.CascadeClassifier('face_detector.xml')
    imageSelected = input("Please select image 1-5.")
    if 0 < int(imageSelected) < 6:
        img = cv2.imread("images/" + imageSelected + ".jpg")

        # Detect faces
        faces = face_cascade.detectMultiScale(img, 1.1, 4)

        # Draw rectangle around the faces
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 6)

        # Export the result
        cv2.imwrite("opencv" + imageSelected + ".png", img)
        print('Successfully saved')
    else:
        print("Input Not Correct.")


def deepFace():
    imageSelected = input("Please select image 1-5.")
    if 0 < int(imageSelected) < 6:
        # Create an Object with the results of deep face analyze
        obj = DeepFace.analyze(img_path="images/" + imageSelected + ".jpg",
                               actions=['age', 'gender', 'race', 'emotion'])
        print(obj)

    else:
        print("Input Not Correct.")


if __name__ == "__main__":
    print("Hello, Welcome to the AI demonstrator.\n")
    print("Here I will do three different AI examples.\n")
    print("1. TenserFlow.\n2. Open CV.\n3. DeepFace.\n")
    userSelect = input()

    if int(userSelect) == 1:
        TenserFlow()
    if int(userSelect) == 2:
        openCV()
    if int(userSelect) == 3:
        deepFace()
    else:
        print("Not a Valid input")
