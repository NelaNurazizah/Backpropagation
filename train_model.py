import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Load dataset
df = pd.read_csv('day.csv')

# Fitur dan target
features = [
    'season', 'yr', 'mnth', 'holiday', 'weekday', 'workingday',
    'weathersit', 'temp', 'atemp', 'hum', 'windspeed'
]
target = 'cnt'

X = df[features]
y = df[[target]]

# Normalisasi
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_scaled, test_size=0.2, random_state=42
)

# Model Jaringan Saraf Tiruan
model = Sequential([
    Dense(64, activation='relu', input_dim=X.shape[1]),
    Dense(32, activation='relu'),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=100, batch_size=16, verbose=1)

# Simpan model dan scaler
model.save('model/model.h5')
with open('model/scaler_X.pkl', 'wb') as f:
    pickle.dump(scaler_X, f)
with open('model/scaler_y.pkl', 'wb') as f:
    pickle.dump(scaler_y, f)

print("Model dan scaler berhasil disimpan ke folder 'model'.")
