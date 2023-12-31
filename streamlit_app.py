import streamlit as st
import joblib
import pickle
import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image

# Main functionalities about the page to be set at first,optional
st.set_page_config(page_title = "Automatic Cooking Evaluation Tool", page_icon = None, layout = "centered",
initial_sidebar_state = "collapsed", menu_items = None)

# Changing font sizes on the sidebar with CSS injection
st.markdown(
    """
    <style>
    .sidebar-toggle {
        font-size: 45px;
    }
    .stRadio div label {
        font-size: 16px;
    }
    </style>
    """,
    unsafe_allow_html = True,
)

# All the models that is working behind the app. They should be in the same directory with the script in the repo.
model_paths = {
    "Upper": "FSC_U.pkl",
    "Lower": "FSC_B.pkl",
    "Pizza": "PIZZA_U.pkl",
    "Apple Pie": None,
    "Grilling White Bread": "GWB_U.pkl"
}

# Function to process the uploaded image
def process_image(uploaded_image, model_path):
    # Read the uploaded image
    image = Image.open(uploaded_image)
    image_array = np.array(image)

    if image_array.shape[-1] == 3:
        image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)

    # Ensure it's a 2D array
    if len(image_array.shape) > 2:
        image_array = image_array.reshape(-1, image_array.shape[-1])

    image_df = pd.DataFrame(image_array)

    # Apply the palette
    img_processed = apply_palette(image_df, palette='NCS7')

    # Transform the image for prediction
    X = transform_row(img_processed, palette='NCS7')

    # Make the prediction
    prediction = run_model(model_path, X)
    
    return prediction

def run_model(model_object_path, X):
    '''
    '''
    # load the model
    with open(model_object_path, 'rb') as model_file:
        model = pickle.load(model_file)
    
    prediction = int(model.predict(X))
    
    return prediction

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

        if str(unique_color) in new_row:
            new_row[str(unique_color)] = round(each_percent, 2)

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
    st.title("Automatic Cooking Evaluation")
    
    # Sidebar for model selection
    model_option = st.sidebar.radio("Select a Product", ["Upper", "Lower", "Pizza", "Apple Pie", "Grilling White Bread"])
    model_path = model_paths[model_option]

    # File uploader
    uploaded_image = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])

    # If the state of the uploaded_image changes to other than None, trigger the process
    if uploaded_image is not None: 
        # Display the uploaded image, create a header, draw a seperator line to the end
        st.subheader("Uploaded Image")
        st.image(uploaded_image, use_column_width=True)
        st.divider()

        # Trigger the process to make all the operations on image and return the final prediction
        prediction = process_image(uploaded_image, model_path)
        
        st.subheader("Evaluation Result")

        # If prediction is been made successfully, display the result as succes box. If not, throw an error box.
        if prediction is not None:
            st.success(f"{prediction}")
        else:
            st.error("Something went wrong. Try again.")
        
if __name__ == "__main__":
    main()