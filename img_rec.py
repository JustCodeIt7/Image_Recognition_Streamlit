# =========================================
# Imports
# =========================================
import streamlit as st
import torch
import torchvision
from torchvision import transforms
from PIL import Image
import requests
import io
import json
from typing import List, Tuple, Optional, Any # Added Any for model/preprocess return

# =========================================
# Model and Data Loading Functions (Helpers)
# =========================================

@st.cache_data
def load_imagenet_labels() -> Optional[List[str]]:
    """Loads the ImageNet class labels from a standard URL.

    Returns:
        Optional[List[str]]: A list of label strings, or None if loading fails.
    """
    LABELS_URL = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
    try:
        response = requests.get(LABELS_URL)
        response.raise_for_status()
        labels = response.json()
        return labels
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching ImageNet labels: {e}")
        return None
    except json.JSONDecodeError:
        st.error("Error decoding ImageNet labels JSON.")
        return None

@st.cache_resource
def load_pytorch_model(model_name: str = "resnet18") -> Tuple[Optional[Any], Optional[Any]]:
    """Loads a specified pre-trained PyTorch model and its preprocessing pipeline.

    Args:
        model_name (str): The name of the model to load (e.g., "resnet18").

    Returns:
        Tuple[Optional[Any], Optional[Any]]: A tuple containing the loaded model
                                             and its preprocessing transform, or (None, None)
                                             if loading fails.
    """
    st.write(f"Loading {model_name} model...")
    weights = None
    model = None

    try:
        if model_name == "resnet18":
            weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1
            model = torchvision.models.resnet18(weights=weights)
        elif model_name == "resnet50":
            weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V2
            model = torchvision.models.resnet50(weights=weights)
        elif model_name == "efficientnet_b0":
            weights = torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1
            model = torchvision.models.efficientnet_b0(weights=weights)
        else:
            st.error(f"Model {model_name} not supported.")
            return None, None

        model.eval() # Set to evaluation mode
        preprocess = weights.transforms()
        st.write("Model loaded successfully!")
        return model, preprocess

    except Exception as e:
        st.error(f"An error occurred loading model {model_name}: {e}")
        return None, None


# =========================================
# Prediction Function (Helper)
# =========================================

def predict(model, preprocess, image: Image.Image, labels: List[str], top_k: int = 5) -> Optional[List[Tuple[str, float]]]:
    """Processes an image and returns the top K predictions.

    Args:
        model: The loaded PyTorch model.
        preprocess: The preprocessing transform associated with the model.
        image (Image.Image): The input image.
        labels (List[str]): The list of class labels.
        top_k (int): The number of top predictions to return.

    Returns:
        Optional[List[Tuple[str, float]]]: A list of (label, probability) tuples,
                                           or None if prediction fails.
    """
    try:
        if image.mode != "RGB":
            image = image.convert("RGB")

        input_tensor = preprocess(image)
        input_batch = input_tensor.unsqueeze(0)

        if torch.cuda.is_available():
            input_batch = input_batch.to('cuda')
            model.to('cuda')

        with torch.no_grad():
            output = model(input_batch)

        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        top_prob, top_indices = torch.topk(probabilities, top_k)

        predictions = [
            (labels[idx], prob.item())
            for idx, prob in zip(top_indices, top_prob)
        ]
        return predictions
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        return None

# =========================================
# UI Component Functions
# =========================================

def setup_sidebar() -> Tuple[str, int]:
    """Sets up the sidebar widgets and returns the selected options."""
    st.sidebar.header("⚙️ Options")
    model_choice = st.sidebar.selectbox(
        "Choose a Model:",
        ("resnet18", "resnet50", "efficientnet_b0"),
        index=0,
        key="model_select" # Added key for potential state management later
    )
    top_k_slider = st.sidebar.slider(
        "Number of Predictions to Show:",
        min_value=1,
        max_value=10,
        value=5,
        key="top_k_select" # Added key
    )
    return model_choice, top_k_slider

def display_main_interface() -> Optional[st.runtime.uploaded_file_manager.UploadedFile]:
    """Displays the main page title, description, and file uploader."""
    st.title("🖼️ PyTorch Image Recognition")
    st.write(
        "Upload an image, and this app will classify it using a pre-trained "
        "PyTorch model."
    )
    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=["jpg", "jpeg", "png"],
        key="file_uploader" # Added key
    )
    return uploaded_file

def display_prediction_results(image: Image.Image, predictions: Optional[List[Tuple[str, float]]], top_k: int):
    """Displays the uploaded image and the prediction results side-by-side."""
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption='Uploaded Image', use_column_width=True)

    with col2:
        st.subheader(f"Top {top_k} Predictions:")
        if predictions:
            for label, probability in predictions:
                st.write(f"- {label}: {probability:.2%}")
        else:
            # Error during prediction is usually handled in predict(),
            # but display an error if predictions list is None for other reasons.
             st.error("Could not retrieve predictions.")


# =========================================
# Main Application Function
# =========================================
def main():
    """Runs the main Streamlit application."""
    # Configure page - Must be the first Streamlit command
    st.set_page_config(
        page_title="PyTorch Image Recognizer",
        page_icon="🖼️",
        layout="centered",
    )

    # Setup UI components
    model_choice, top_k_slider = setup_sidebar()
    uploaded_file = display_main_interface()

    # Load essential data (labels)
    imagenet_labels = load_imagenet_labels()
    if not imagenet_labels:
        st.error("Cannot proceed without ImageNet labels. Please check the connection or URL.")
        st.stop() # Stop if labels are essential and failed to load

    # Load model based on choice (cached)
    selected_model, preprocess_pipeline = load_pytorch_model(model_choice)

    # Process uploaded file if available and model loaded
    if uploaded_file is not None:
        if selected_model and preprocess_pipeline:
            try:
                # Open and validate image
                image = Image.open(uploaded_file)

                # Perform prediction
                with st.spinner('🧠 Classifying...'):
                    predictions = predict(
                        selected_model,
                        preprocess_pipeline,
                        image,
                        imagenet_labels,
                        top_k=top_k_slider
                    )

                # Display results
                display_prediction_results(image, predictions, top_k_slider)

            except Exception as e:
                # Catch errors related to opening/processing the image itself
                st.error(f"Error processing image file: {e}")
                st.error("Please try uploading a valid image file (JPG, JPEG, PNG).")
        else:
            # Handle case where model loading failed earlier
            st.error(f"Model '{model_choice}' could not be loaded. Cannot classify the image.")

    elif uploaded_file is None:
        st.info("👆 Upload an image file to get started.")

# =========================================
# Script Entry Point
# =========================================
if __name__ == "__main__":
    main()