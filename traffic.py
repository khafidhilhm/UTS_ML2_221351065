import streamlit as st
import numpy as np
import pickle
import tensorflow as tf

# Load model dan preprocessing
@st.cache(allow_output_mutation=True)
def load_artifacts():
    with open('label_encoder.pkl', 'rb') as f:
        le = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        sc = pickle.load(f)
    
    interpreter = tf.lite.Interpreter(model_path='traffic_model.tflite')
    interpreter.allocate_tensors()
    
    return le, sc, interpreter

le, sc, interpreter = load_artifacts()

# Streamlit UI
st.title('Prediksi Situasi Lalu Lintas')

car_count = st.number_input('Jumlah Mobil', min_value=0)
bike_count = st.number_input('Jumlah Sepeda', min_value=0)
bus_count = st.number_input('Jumlah Bus', min_value=0)
truck_count = st.number_input('Jumlah Truk', min_value=0)
total = st.number_input('Total Kendaraan', min_value=0)
hour = st.number_input('Jam (0-23)', min_value=0, max_value=23)

if st.button('Prediksi'):
    input_data = np.array([[car_count, bike_count, bus_count, truck_count, total, hour]])
    input_scaled = sc.transform(input_data).astype(np.float32)
    
    # Predict
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    interpreter.set_tensor(input_details[0]['index'], input_scaled)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    
    pred_class = np.argmax(output_data)
    situation = le.inverse_transform([pred_class])[0]
    
    st.success(f'Situasi Lalu Lintas Diprediksi: {situation}')