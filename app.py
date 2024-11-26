import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
import os
from tensorflow.keras.preprocessing import image

# Path lokal ke dataset
dataset_path = './dataset/gambar-hewan/'  # Sesuaikan path dataset Anda

# Membuat data generator untuk training dan validasi
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    dataset_path,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    dataset_path,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# Definisi model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(len(train_generator.class_indices), activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Training model
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=50
)

# Menyimpan model ke file lokal
model.save('./dataset/gambar-hewan/my_model.keras')

# Load model untuk prediksi
model = tf.keras.models.load_model('./dataset/gambar-hewan/my_model.keras')

# Mendapatkan label kelas dari generator
class_labels = list(train_generator.class_indices.keys())

# Direktori untuk gambar yang akan diprediksi
image_dir = './dataset/gambar-hewan/ayam/'  # Ganti dengan direktori gambar Anda

# Loop untuk prediksi setiap gambar di direktori
# Mendapatkan semua file gambar dari direktori
for filename in os.listdir(image_dir):
    if filename.endswith(('.jpg', '.jpeg', '.png')):  # Filter file gambar
        img_path = os.path.join(image_dir, filename)
        img = image.load_img(img_path, target_size=(150, 150))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = x / 255.0

        predictions = model.predict(x)
        predicted_class = np.argmax(predictions[0])

        plt.imshow(img)
        plt.axis('off')
        plt.title(f"File: {filename}\nPredicted: {class_labels[predicted_class]}, Confidence: {predictions[0][predicted_class]:.2f}")
        plt.show()

# Visualisasi akurasi model
plt.figure(figsize=(10, 5))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Model Accuracy')
plt.legend()
plt.show()