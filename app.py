from flask import Flask, render_template, request
import numpy as np
import pickle
import tensorflow as tf

app = Flask(__name__)

# Load model dan scaler
model = tf.keras.models.load_model('model/model.h5', compile=False)
scaler_X = pickle.load(open('model/scaler_X.pkl', 'rb'))
scaler_y = pickle.load(open('model/scaler_y.pkl', 'rb'))

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        try:
            # Ambil input dari form
            input_data = [
                float(request.form['season']),
                float(request.form['yr']),
                float(request.form['mnth']),
                float(request.form['holiday']),
                float(request.form['weekday']),
                float(request.form['workingday']),
                float(request.form['weathersit']),
                float(request.form['temp']),
                float(request.form['atemp']),
                float(request.form['hum']),
                float(request.form['windspeed']),
            ]
            input_array = np.array([input_data])
            input_scaled = scaler_X.transform(input_array)
            result_scaled = model.predict(input_scaled)
            prediction = scaler_y.inverse_transform(result_scaled)[0][0]
        except:
            prediction = "Terjadi kesalahan input!"
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
