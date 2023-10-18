import streamlit as st
import joblib
import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image

# Load the machine learning models
model_upper = joblib.load("FSC_U.pkl")
model_lower = joblib.load("FSC_B.pkl")

# Function to process the uploaded image
def process_image(uploaded_image, model):
    # Read the uploaded image
    image = Image.open(uploaded_image)
    image_array = np.array(image)

    if image_array.shape[-1] == 3:
        image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)

    image_array = image_array.reshape(-1)
    
    # Apply the palette
    img_processed = apply_palette(image_array, palette='NCS7')

    # Transform the image for prediction
    X = transform_row(img_processed, palette='NCS7')

    # Make the prediction
    prediction = run_model(model, X)
    
    return image, prediction

def transform_row(img_df, palette='NCS7'):
    '''
    '''
    # identify unique colors
    unique_colors = np.unique(img_df.values)
    
    # calculate total cell
    total_cell = img_df.shape[0]*img_df.shape[1]
    
    # choose training features according to palette choice
    if palette == 'NCS7': new_row = {'0':0.,'2':0.,'4':0.,'6':0.,'8':0.,'10':0.,'12':0.,'14':0.,'16':0.,'18':0.}
    else: pass

    # calculate color distribution for each color and store these distribution in dictionary
    for unique_color in unique_colors:
        each_count = img_df[img_df == unique_color].count().sum()
        each_percent = each_count / total_cell
        new_row.update({str(unique_color):round(each_percent,2)})
    
    # convert color distribution dictionary to one row dataframe
    X = pd.DataFrame([new_row])
    
    return X
    
def apply_palette(df, palette='NCS7'):
    '''
    '''
    # cluster grayscaled pixels between 0-255 into color interval of palette
    if palette == 'NCS7':
        # black is equal to 0
        for i in range(250,257): df = df.replace(i,0)
            
        # light green is equal to 4  
        for i in range(110,119): df = df.replace(i,4)
            
        # cyan is equal to 6
        for i in range(170,188): df = df.replace(i,6)
        
        # pink is equal to 8
        for i in range(98,111): df = df.replace(i,8)
            
        # yellow is equal to 10
        for i in range(218,235): df = df.replace(i,10)
            
        # blue is equal to 12
        for i in range(20,37): df = df.replace(i,12)
        
        # green is equal to 14
        for i in range(141,158): df = df.replace(i,14)
            
        # red is equal to 16
        for i in range(69,85): df = df.replace(i,16)
            
        return df
    elif palette == 'NCS14':
        pass
    else:
        pass

# Define the Streamlit app
def main():
    st.title("Image Classifier App")
    
    # Sidebar for model selection
    model_option = st.sidebar.radio("Select a Model", ["Upper", "Lower"])
    
    # File uploader
    uploaded_image = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])
    
    if uploaded_image is not None:
        st.subheader("Uploaded Image")
        st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
        
        if model_option == "Upper":
            model = model_upper
        else:
            model = model_lower
        
        # Process the image and make predictions
        image, prediction = process_image(uploaded_image, model)
        
        st.subheader("Prediction Result")
        if prediction == 1:
            st.write("The model predicts 'Upper'.")
        else:
            st.write("The model predicts 'Lower'.")
        
if __name__ == "__main__":
    main()