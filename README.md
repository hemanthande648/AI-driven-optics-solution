import cv2
import tensorflow as tf
from tensorflow import keras
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt

# Load dataset of images of optical lenses with defects
train_dir = 'path/to/train/directory'
test_dir = 'path/to/test/directory'

# Define constants
IMG_WIDTH, IMG_HEIGHT = 256, 256
BATCH_SIZE = 32

# Load and preprocess training data
train_datagen = keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# Load and preprocess testing data
test_datagen = keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# Define multi-modal generative AI model
def build_model():
    # Define encoder model
    encoder = keras.Sequential([
        keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation='relu')
    ])

    # Define decoder model
    decoder = keras.Sequential([
        keras.layers.Dense(128, activation='relu', input_shape=(128,)),
        keras.layers.Reshape((4, 4, 32)),
        keras.layers.Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same'),
        keras.layers.Conv2D(3, (3, 3), activation='tanh', padding='same')
    ])

    # Define discriminator model
    discriminator = keras.Sequential([
        keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')
    ])

    # Define generator model
    generator = keras.Sequential([
        encoder,
        decoder
    ])

    # Define GAN model
    gan = keras.Sequential([
        generator,
        discriminator
    ])

    return gan

# Build and compile GAN model
gan = build_model()
gan.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train GAN model
history = gan.fit(train_generator, epochs=10, validation_data=test_generator)

# Evaluate GAN model
loss, accuracy = gan.evaluate(test_generator)
print(f'Test loss: {loss:.3f}, Test accuracy: {accuracy:.3f}')

# Use GAN model to detect and classify defects
def detect_defects(image):
    # Preprocess image
    image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
    image = image / 255.0

    # Run image through GAN model
    output = gan.predict(image)

    # Threshold output to determine defect presence
    threshold = 0.5
    defect_present = output > threshold

    # Classify defect type
    defect_type = np.argmax(output, axis=1)

    return defect_present, defect_type

# Test defect detection and classification
image = cv2.imread('path/to/test/image.jpg')
defect_present, defect_type = detect_defects(image)
print(f'Defect present: {defect_present}, Defect type: {defect_type}')
