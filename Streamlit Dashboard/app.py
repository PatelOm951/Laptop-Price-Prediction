# import streamlit as st
# import pickle
# import numpy as np

# # Load the model and dataframe
# pipe = pickle.load(open('pipe.pkl', 'rb'))
# df = pickle.load(open('df.pkl', 'rb'))

# st.set_page_config(page_title="Laptop Price Predictor", layout="centered")
# st.markdown("""
#     <style>
#         .main-title {
#             font-size: 36px;
#             text-align: center;
#             font-weight: bold;
#             margin-bottom: 20px;
#         }
#         .section-title {
#             font-size: 20px;
#             margin-top: 20px;
#             margin-bottom: 10px;
#             font-weight: bold;
#         }
#         .prediction-box {
#             padding: 20px;
#             border-radius: 15px;
#             background-color: #f0f2f6;
#             border: 2px solid #0073e6;
#             text-align: center;
#             margin-top: 30px;
#         }
#         .prediction-text {
#             font-size: 28px;
#             font-weight: bold;
#             color: #0073e6;
#         }
#     </style>
# """, unsafe_allow_html=True)

# st.markdown('<div class="main-title">💻 Laptop Price Predictor</div>', unsafe_allow_html=True)

# st.markdown('<div class="section-title">📋 Select Configuration</div>', unsafe_allow_html=True)

# # Inputs in two columns
# col1, col2 = st.columns(2)

# with col1:
#     company = st.selectbox('Brand', df['Company'].unique())
#     filtered_df = df[df['Company'] == company]
#     type = st.selectbox('Type', filtered_df['TypeName'].unique())
#     ram = st.selectbox('RAM (in GB)', [2, 4, 6, 8, 12, 16, 24, 32, 64])
#     weight = st.number_input('Weight (kg)', format="%.2f")
#     touchscreen = st.radio('Touchscreen', ['No', 'Yes'], horizontal=True)
#     ips = st.radio('IPS Display', ['No', 'Yes'], horizontal=True)

# with col2:
#     screen_size = st.slider('Screen Size (in inches)', 10.0, 18.0, 13.3)
#     resolution = st.selectbox(
#         'Resolution',
#         ['1920x1080', '1366x768', '1600x900', '3840x2160', '1440x900',
#          '3200x1800', '2880x1800', '2560x1600', '2560x1440', '2304x1440']
#     )
#     cpu = st.selectbox('CPU Brand', filtered_df['Cpu Brand'].unique())
#     hdd = st.selectbox('HDD (GB)', [0, 128, 256, 512, 1024, 2048])
#     ssd = st.selectbox('SSD (GB)', [0, 8, 128, 256, 512, 1024])
#     gpu = st.selectbox('GPU Brand', filtered_df['Gpu Brand'].unique())
#     os = st.selectbox('Operating System', filtered_df['os'].unique())

# # Prediction
# if st.button('🔮 Predict Price'):
#     # Convert touchscreen and ips to binary
#     touchscreen_bin = 1 if touchscreen == 'Yes' else 0
#     ips_bin = 1 if ips == 'Yes' else 0
#     X_res = int(resolution.split('x')[0])
#     Y_res = int(resolution.split('x')[1])
#     ppi = ((X_res ** 2) + (Y_res ** 2)) ** 0.5 / screen_size

#     # Check if configuration exists in dataset (excluding PPI/weight)
#     match = df[
#     (df['Company'] == company) &
#     (df['TypeName'] == type) &
#     (df['Ram'] == ram) &
#     (df['Touchscreen'] == touchscreen_bin) &
#     (df['Ips'] == ips_bin) &
#     (df['Cpu Brand'] == cpu) &
#     (df['Gpu Brand'] == gpu) &
#     (df['os'] == os) &
#     (df['HDD'] == hdd) &  # more flexible check
#     (df['SSD'] == ssd)
# ]

#     if match.empty:
#         st.error("❌ This configuration doesn't exist in the dataset. Please adjust your selections.")
#     else:
#         query = np.array([company, type, ram, weight, touchscreen_bin, ips_bin, ppi,
#                           cpu, hdd, ssd, gpu, os]).reshape(1, 12)

#         predicted_price = int(np.exp(pipe.predict(query)[0]))

#         st.markdown(f"""
#             <div class="prediction-box">
#                 <div class="prediction-text">
#                     🎯 Estimated Price: ₹ {predicted_price}
#                 </div>
#                 <p style="color: #4d4d4d; font-size: 16px;">
#                     Based on your selected configuration.
#                 </p>
#             </div>
#         """, unsafe_allow_html=True)

import streamlit as st
import pickle
import numpy as np

# Load the model and dataframe
pipe = pickle.load(open('pipe.pkl', 'rb'))
df = pickle.load(open('df.pkl', 'rb'))

st.set_page_config(page_title="Laptop Price Predictor", layout="centered")
st.markdown("""
    <style>
        .main-title {
            font-size: 36px;
            text-align: center;
            font-weight: bold;
            margin-bottom: 20px;
        }
        .section-title {
            font-size: 20px;
            margin-top: 20px;
            margin-bottom: 10px;
            font-weight: bold;
        }
        .prediction-box {
            padding: 20px;
            border-radius: 15px;
            background-color: #f0f2f6;
            border: 2px solid #0073e6;
            text-align: center;
            margin-top: 30px;
        }
        .prediction-text {
            font-size: 28px;
            font-weight: bold;
            color: #0073e6;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">💻 Laptop Price Predictor</div>', unsafe_allow_html=True)

st.markdown('<div class="section-title">📋 Select Configuration</div>', unsafe_allow_html=True)

# Inputs in two columns
col1, col2 = st.columns(2)

with col1:
    company = st.selectbox('Brand', df['Company'].unique())
    filtered_df = df[df['Company'] == company]
    type = st.selectbox('Type', filtered_df['TypeName'].unique())
    ram = st.selectbox('RAM (in GB)', [2, 4, 6, 8, 12, 16, 24, 32, 64])
    weight = st.number_input('Weight (kg)', format="%.2f")
    touchscreen = st.radio('Touchscreen', ['No', 'Yes'], horizontal=True)
    ips = st.radio('IPS Display', ['No', 'Yes'], horizontal=True)

with col2:
    screen_size = st.slider('Screen Size (in inches)', 10.0, 18.0, 13.3)
    resolution = st.selectbox(
        'Resolution',
        ['1920x1080','1366x768','1600x900','3840x2160','3200x1800','2880x1800','2560x1600','2560x1440','2304x1440']
    )
    cpu = st.selectbox('CPU Brand', filtered_df['Cpu Brand'].unique())
    hdd = st.selectbox('HDD (GB)', [0, 128, 256, 500, 512, 1000, 1024, 2000, 2048])
    ssd = st.selectbox('SSD (GB)', [0, 8, 128, 256, 500, 512, 1000, 1024])
    gpu = st.selectbox('GPU Brand', filtered_df['Gpu Brand'].unique())
    os = st.selectbox('Operating System', filtered_df['os'].unique())

# Prediction
if st.button('🔮 Predict Price'):
    # Convert touchscreen and ips to binary
    touchscreen_bin = 1 if touchscreen == 'Yes' else 0
    ips_bin = 1 if ips == 'Yes' else 0
    X_res = int(resolution.split('x')[0])
    Y_res = int(resolution.split('x')[1])
    ppi = ((X_res ** 2) + (Y_res ** 2)) ** 0.5 / screen_size

    # Check if configuration exists in dataset (excluding PPI/weight)
    match = df[
    (df['Company'] == company) &
    (df['TypeName'] == type) &
    (df['Ram'] == ram) &
    (df['Touchscreen'] == touchscreen_bin) &
    (df['Ips'] == ips_bin) &
    (df['Cpu Brand'] == cpu) &
    (df['Gpu Brand'] == gpu) &
    (df['os'] == os) &
    (df['HDD'] == hdd) &  # more flexible check
    (df['SSD'] == ssd)
]

    if match.empty:
        st.error("❌ This configuration doesn't exist in the dataset. Please adjust your selections.")
    else:
        query = np.array([company, type, ram, weight, touchscreen_bin, ips_bin, ppi,
                          cpu, hdd, ssd, gpu, os]).reshape(1, 12)

        predicted_price = int(np.exp(pipe.predict(query)[0]))

        st.markdown(f"""
            <div class="prediction-box">
                <div class="prediction-text">
                    🎯 Estimated Price: ₹ {predicted_price}
                </div>
                <p style="color: #4d4d4d; font-size: 16px;">
                    Based on your selected configuration.
                </p>
            </div>
        """, unsafe_allow_html=True)