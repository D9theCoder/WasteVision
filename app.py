import streamlit as st
import numpy as np
import cv2
from skimage.feature import local_binary_pattern, hog
from skimage import color, filters
import pickle
from PIL import Image
import io
from sklearn.decomposition import PCA
from skimage.transform import resize

# Constants for feature extraction
LBP_RADIUS = 3
LBP_POINTS = 8 * LBP_RADIUS
HOG_PARAMS = {
    'orientations': 8,
    'pixels_per_cell': (16, 16), 
    'cells_per_block': (2, 2),
    'block_norm': 'L2-Hys'
}

def preprocess_image(image):
    if image.shape[-1] == 4:
        image = image[:, :, :3] 
    gray_image = color.rgb2gray(image)
    blurred_image = filters.gaussian(gray_image, sigma=1)
    
    resized_image = resize(blurred_image, (384, 512)) # dimensi dari gambar
    
    normalized_image = resized_image / 255.0
    return normalized_image

def extract_features(image):
    preprocessed_img = preprocess_image(image)
    
    # LBP features
    lbp = local_binary_pattern(preprocessed_img, LBP_POINTS, LBP_RADIUS, method="uniform")
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, LBP_POINTS + 3), density=True)

    # HOG features
    hog_features = hog(preprocessed_img, **HOG_PARAMS)

    # Combine features
    combined_features = np.hstack((lbp_hist, hog_features))
    return combined_features.reshape(1, -1)

# Load models and PCA
@st.cache_resource
def load_models():
    models = {
        'SVM': pickle.load(open('models/svm_model.pkl', 'rb')),
        'KNN': pickle.load(open('models/knn_model.pkl', 'rb')),
        'Random Forest': pickle.load(open('models/rf_model.pkl', 'rb'))
    }
    # Load PCA
    pca = pickle.load(open('models/pca_model.pkl', 'rb'))
    return models, pca

def main():
    st.title("Waste Classification App")
    
    # Load models and PCA
    models, pca = load_models()
    
    # Model selection
    selected_model = st.selectbox(
        "Select Classification Model",
        ["SVM", "KNN", "Random Forest"]
    )
    
    # File uploader
    uploaded_file = st.file_uploader("Choose an image file", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file is not None:
        # Convert uploaded file to image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        
        # Display the uploaded image
        st.image(uploaded_file, caption='Uploaded Image', use_container_width=True)
        
        # Make prediction button
        if st.button('Classify Image'):
            # Extract features
            features = extract_features(image)
            
            # Apply PCA transformation
            features_pca = pca.transform(features)
            
            # Make prediction
            model = models[selected_model]
            prediction = model.predict(features_pca)[0]
            
            # Display result
            st.success(f'Prediction: This is {prediction} waste')

if __name__ == "__main__":
    main()
