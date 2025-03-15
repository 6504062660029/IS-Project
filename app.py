from flask import Flask, render_template, request
import pickle
import numpy as np
import tensorflow as tf
from PIL import Image  # เพิ่มการใช้ PIL เพื่อจัดการภาพ

# โหลดโมเดล Machine Learning
ml_model = pickle.load(open('house_price_model.sav', 'rb'))

# โหลดโมเดล Neural Network
nn_model = tf.keras.models.load_model("mnist_model.h5")

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ml_model')
def ml_model_page():
    return render_template('ml_model.html')

@app.route('/predict_ml', methods=['GET', 'POST'])
def predict_ml():
    prediction = None
    if request.method == 'POST':
        data = [float(x) for x in request.form.values()]
        prediction = ml_model.predict([data])[0]
    return render_template('predict_ml.html', prediction=prediction)

@app.route('/predict_nn', methods=['GET', 'POST'])
def predict_nn():
    prediction = None
    if request.method == 'POST':
        # อ่านไฟล์ภาพจากผู้ใช้
        image_file = request.files['image']
        
        # ใช้ Pillow เพื่อเปิดภาพและแปลงเป็นขนาด 28x28 pixels (Grayscale)
        image = Image.open(image_file).convert('L')  # 'L' = Grayscale
        image = image.resize((28, 28))  # ปรับขนาดเป็น 28x28
        
        # แปลงภาพเป็น numpy array และ normalize เป็นค่า 0-1
        image_array = np.array(image).astype('float32') / 255.0
        
        # Reshape เป็น (1, 784) เพื่อให้ตรงกับโมเดล
        image_array = image_array.reshape(1, 784)

        # ทำนายผลลัพธ์
        prediction = np.argmax(nn_model.predict(image_array))

    return render_template('predict_nn.html', prediction=prediction)


if __name__ == '__main__':
    app.run(debug=True)
