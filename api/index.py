from flask import Flask, request, render_template, jsonify
import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)

# Load model
model = tf.keras.models.load_model('../dataset/gambar-hewan/my_model.keras')

# Class labels
class_labels = ['ayam', 'kucing', 'rusa', 'unta']  # Ganti dengan label kelas sebenarnya

@app.route('/')
def home():
    return render_template('../templates/index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file:
        # Simpan file sementara untuk prediksi
        file_path = os.path.join('../uploads', file.filename)
        file.save(file_path)

        # Prediksi
        img = image.load_img(file_path, target_size=(150, 150))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = x / 255.0

        predictions = model.predict(x)
        predicted_class = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class]

        # Hapus file sementara
        os.remove(file_path)

        # Kirim hasil prediksi sebagai JSON
        return jsonify({
            "prediction": class_labels[predicted_class],
            "confidence": float(confidence)
        })

if __name__ == "__main__":
    app.run(debug=True)