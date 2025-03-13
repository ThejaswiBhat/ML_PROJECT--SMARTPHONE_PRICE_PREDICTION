import streamlit as st
import pickle
from PIL import Image 
import pandas as pd
import numpy as np
import os
import base64

# Creating a function to get Images:

Base_Image_Path = r"F:\Python Studies\Luminar\Project Files\Machine Learning\Phone_Images"

def get_image_path(Brand, Model):
    model_folder = os.path.join(Base_Image_Path, Brand)
    return os.path.join(model_folder, f"{Model}.jpg")


# Creating function for user input:

def user_input():
    with st.sidebar:

        with st.sidebar:
            st.markdown(
        """
        <style>
        [data-testid="stSidebar"] {
            background-color: #2a2a2a;
            border-right: 2px solid #FF5733;
            padding: 15px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

        st.markdown('<h2 style="color:#FF5733; font-weight: bold;">Select Your Preferred Features</h2>', unsafe_allow_html=True)

        image_sidebar = Image.open('1.jpg')
        st.image(image_sidebar, use_container_width=True)

        brand_models = {
        "Vivo": ["V23", "V21", "X70", "X60"],
        "Google": ["Pixel 7", "Pixel 6", "Pixel 5", "Pixel 6a"],
        "Samsung": ["Galaxy A52", "Galaxy S21", "Galaxy S22", "Galaxy M32"],
        "Oppo": ["Reno 7", "Reno 6", "Reno 5", "Find X3"],
        "Xiaomi": ["Mi 11", "Redmi Note 11", "Redmi Note 10", "Poco X3"],
        "OnePlus": ["OnePlus 9", "OnePlus 10T", "OnePlus 8T", "OnePlus Nord"],
        "Realme": ["Realme GT", "Realme 9", "Realme 8", "Realme X7"],
        "Apple": ["iPhone 13", "iPhone SE", "iPhone 12", "iPhone 14"]
        }

        Brand = st.selectbox('Select Your Brand', list(brand_models.keys()))
        
        Model = st.selectbox('Select Your Model',brand_models[Brand])

        RAM = st.selectbox('RAM (GB) ',[4,6,8,12])

        Storage = st.selectbox('Storage (GB)',[64,128,256,512])

        Camera_Quality = st.selectbox('Select Camera Quality (MP)',[12,48,64,108])

        processor_options = {
        "Apple": ["A15 Bionic", "A14 Bionic"],
        "Other": ["MediaTek Dimensity 1200", "Exynos 2100", "Snapdragon 888"]
        }

        if Brand == 'Apple':
            Processor_Type = st.selectbox('Select Processor Type',processor_options["Apple"])

        else:
            Processor_Type = st.selectbox('Select Processor Type',processor_options["Other"])

        
        Screen_Size = st.selectbox('Select Screen Size',[5.8,6.1,6.4,6.7,6.9])

        Battery_Life = st. selectbox('Select Battery Capacity (mAH)',[3500,4000,4500,5000,6000])

        os_options = {
            'Apple':['iOS 14','iOS 15','iOS 16'],
            'Other':['Android 11','Android 12']
        }

        if Brand == 'Apple':
            OS_Version = st.selectbox('Select OS Version',os_options["Apple"])
        
        else:
            OS_Version = st.selectbox('Select OS Version',os_options["Other"])

        
        Release_Year = st.number_input('Select Release Year',min_value=2020,max_value=2023,step=1)

        Network = st.selectbox('Select Network Type',['4G','5G'])


        return {
            'Brand': Brand,
            'Model': Model,
            'RAM': RAM,
            'Storage': Storage,
            'Camera_Quality': Camera_Quality,
            'Processor_Type': Processor_Type,
            'Screen_Size': Screen_Size,
            'Battery_Life': Battery_Life,
            'OS_Version': OS_Version,
            'Release_Year': Release_Year,
            'Network': Network
        }

    
# Creating the main function for making prediciton:

def main():
    st.set_page_config(
        page_title='Phone Price Prediction',
        layout='wide',
        initial_sidebar_state='expanded'
    )
    

    # For Background:
    
    def set_background(background_path):
        with open(background_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode()

        st.markdown(
            f"""
            <style>
            .stApp {{
                background-image: url("data:image/jpg;base64,{encoded_string}");
                background-size: cover;
                background-repeat: no-repeat;
                background-attachment: fixed;
            }}
            </style>
            """,
            unsafe_allow_html=True
        )

    
    set_background(r"F:\Python Studies\Luminar\Project Files\Machine Learning\Background\3.jpg")

    # For Main Title 
    st.markdown(
        """
        <div style="background-color:rgba(255,255,255,0.15); padding:15px; border-radius:10px;">
            <h3 style="color:#FFFFFF; text-align:center;">About This App</h3>
            <p style="color:#F1F1F1; text-align:center;">
            This application helps you estimate the price of your phone based on various features such as brand, RAM, storage, camera quality, and more. Select your preferred specifications and get an instant price prediction!
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )


    st.markdown(
    """
    <style>
        .title {
            font-size: 40px;
            font-weight: bold;
            text-align: center;
            background: linear-gradient(to right, #ff7e5f, #feb47b);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
    </style>
    <h1 class="title">Unlock Your Phone's True Worth! </h1>
    """,
    unsafe_allow_html=True
    )

    user_data = user_input()

    features = pd.DataFrame([user_data])

    # Importing Encoding:

    le1 = pickle.load(open('le1.sav','rb'))
    one = pickle.load(open('one.sav','rb'))
    one1 = pickle.load(open('one1.sav','rb'))
    one2 = pickle.load(open('one2.sav','rb'))
    one3 = pickle.load(open('one3.sav','rb'))

    brand_df = pd.DataFrame(one.transform(features[['Brand']]),columns=one.get_feature_names_out())
    model_df = pd.DataFrame(one1.transform(features[['Model']]),columns=one1.get_feature_names_out())
    processor_df = pd.DataFrame(one2.transform(features[['Processor_Type']]),columns=one2.get_feature_names_out())
    network_df = pd.DataFrame(one3.transform(features[['Network']]),columns=one3.get_feature_names_out())

    features.drop(['Brand','Model','Processor_Type','Network'],axis=1,inplace=True)

    features['OS_Version'] = le1.transform(features['OS_Version'])

    features = pd.concat([features,brand_df,model_df,processor_df,network_df],axis=1)

    # Importing Model and Scaler:
    model = pickle.load(open('model.sav','rb'))

    scaler = pickle.load(open('scaler.sav','rb'))

    # Creating 2 columns:

    col1 , col2 , col3 = st.columns([1,2,1])

    # for col1:

    with col1:
        st.markdown(
            """
            <h3 style="
                color: #FFD700; /* Light Yellow */
                text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.8);
                font-weight: bold;
            ">
                Specifications
            </h3>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            f"""
            <div style="background-color:#1e1e1e; padding:10px; border-radius:10px;">
                <p><strong>Brand:</strong> {user_data['Brand']}</p>
                <p><strong>Model:</strong> {user_data['Model']}</p>
                <p><strong>RAM:</strong> {user_data['RAM']} GB</p>
                <p><strong>Storage:</strong> {user_data['Storage']} GB</p>
                <p><strong>Battery:</strong> {user_data['Battery_Life']} mAH</p>
                <p><strong>Network:</strong> {user_data['Network']}</p>
            </div>
            """,
            unsafe_allow_html=True
        )


    # For clo2 (Image):
    image_path = os.path.join(Base_Image_Path, user_data['Brand'], f"{user_data['Model']}.jpg")


    if os.path.isfile(image_path):
        with col2:
            st.markdown(
                f"""
                <div style="text-align: center;">
                    <h3 style="color:#FFD700;"></h3>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            image = Image.open(image_path)
            new_width = 300
            new_height = 400  # Adjust this value as needed
            image = image.resize((new_width, new_height))

            # Display resized image
            st.image(image, use_container_width=False)

    with col3:

        # For Button:

        st.markdown(
            """
            <style>
                div.stButton > button {
                    background-color: #FF5733;
                    color: white;
                    font-size: 22px;  /* Increased font size */
                    font-weight: bold;
                    padding: 16px 28px;  /* Increased padding */
                    border: none;
                    border-radius: 12px;  /* Slightly larger border-radius */
                    width: 100%;
                    cursor: pointer;
                    transition: all 0.3s ease-in-out;
                    text-align: center;
                    display: inline-block;
                }
                div.stButton > button:hover {
                    background-color: #E04E2A;
                    transform: scale(1.08); /* Slightly bigger hover effect */
                    box-shadow: 0px 6px 12px rgba(255, 87, 51, 0.5);
                }
            </style>
            """,
            unsafe_allow_html=True
        )


        pred = st.button('PREDICT')

        if pred:
            features = scaler.transform(features)
            predict = model.predict(features)
            st.success(f'ESTIMATED PRICE IS: ₹{predict[0]:,.2f}')




main()