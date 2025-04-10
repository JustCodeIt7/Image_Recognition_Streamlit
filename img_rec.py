# =========================================
# Tutorial Step 1: Imports
# =========================================
# Core Streamlit library
import streamlit as st
# PyTorch for deep learning models and tensors
import torch
# Torchvision for pre-trained models and image transforms
import torchvision
from torchvision import transforms
# PIL (Pillow) for image manipulation
from PIL import Image
# Requests for fetching data from URLs (like labels)
import requests
# io for handling byte streams (used with file uploader)
import io
# json for parsing labels data
import json
# Typing for type hints (good practice)
from typing import List, Tuple, Optional

# =========================================
# Tutorial Step 2: Page Configuration (Optional)
# =========================================
# Set the title and icon shown in the browser tab
st.set_page_config(
    page_title="PyTorch Image Recognizer",
    page_icon="🖼️",
    layout="centered",
)

# =========================================
# Tutorial Step 3: Helper Functions (Model, Labels, Prediction)
# =========================================

# --- Function to Load ImageNet Labels ---
# Use @st.cache_data for functions returning serializable data (like lists, dicts)
# This prevents re-downloading labels on every interaction.
@st.cache_data
def load_imagenet_labels() -> Optional[List[str]]:
    """Loads the ImageNet class labels from a standard URL."""
    LABELS_URL = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
    try:
        response = requests.get(LABELS_URL)
        response.raise_for_status() # Check for download errors
        labels = response.json()
        return labels
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching ImageNet labels: {e}")
        return None
    except json.JSONDecodeError:
        st.error("Error decoding ImageNet labels JSON.")
        return None

# --- Function to Load PyTorch Model ---
# Use @st.cache_resource for non-serializable objects like ML models
# This prevents reloading the large model file on every interaction.
@st.cache_resource
def load_pytorch_model(model_name: str = "resnet18"):
    """Loads a specified pre-trained PyTorch model and its preprocessing pipeline."""
    st.write(f"Loading {model_name} model...") # Show loading message
    weights = None
    model = None

    # Select the appropriate weights and model based on the name
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
        return None, None # Return None if model is not found

    # Crucial: Set the model to evaluation mode (disables dropout, uses batch norm stats)
    model.eval()

    # Get the preprocessing steps recommended for this model
    preprocess = weights.transforms()
    st.write("Model loaded successfully!")
    return model, preprocess

# --- Function to Make Predictions ---
def predict(model, preprocess, image: Image.Image, labels: List[str], top_k: int = 5) -> Optional[List[Tuple[str, float]]]:
    """Processes an image and returns the top K predictions."""
    # 1. Ensure image is RGB
    if image.mode != "RGB":
        image = image.convert("RGB")

    # 2. Apply preprocessing transformations
    input_tensor = preprocess(image)

    # 3. Add a batch dimension (models expect batches)
    input_batch = input_tensor.unsqueeze(0)

    # 4. Optional: Move tensor to GPU if available for faster inference
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model.to('cuda') # Also move the model to GPU

    # 5. Perform inference without calculating gradients
    with torch.no_grad():
        output = model(input_batch)

    # 6. Get probabilities using softmax
    probabilities = torch.nn.functional.softmax(output[0], dim=0)

    # 7. Get the top K probabilities and their corresponding class indices
    top_prob, top_indices = torch.topk(probabilities, top_k)

    # 8. Map indices to class labels and format the output
    predictions = [
        (labels[idx], prob.item())  # .item() gets the Python number from a tensor
        for idx, prob in zip(top_indices, top_prob)
    ]

    return predictions


# =========================================
# Tutorial Step 4: Main App Interface Setup
# =========================================
st.title("🖼️ PyTorch Image Recognition")
st.write(
    "Upload an image, and this app will classify it using a pre-trained "
    "PyTorch model (ResNet or EfficientNet)."
)

# =========================================
# Tutorial Step 5: Sidebar for Options (Introduce Later?)
# =========================================
# You can initially skip the sidebar and hardcode model_name = "resnet18"
# and top_k = 5, then introduce this section later in the tutorial.
st.sidebar.header("⚙️ Options")

# Model selection dropdown
model_choice = st.sidebar.selectbox(
    "Choose a Model:",
    ("resnet18", "resnet50", "efficientnet_b0"),
    index=0 # Default selection (resnet18)
)

# Slider to choose number of predictions
top_k_slider = st.sidebar.slider(
    "Number of Predictions to Show:",
    min_value=1,
    max_value=10,
    value=5 # Default value
)

# =========================================
# Tutorial Step 6: Load Model and Labels based on Choice
# =========================================
# Load the labels first (required for prediction function)
imagenet_labels = load_imagenet_labels()

# Load the selected model and its preprocessing function
# This will only run once per model choice due to caching
if imagenet_labels:
    # Only attempt to load model if labels are available
    selected_model, preprocess_pipeline = load_pytorch_model(model_choice)
else:
    st.error("Cannot proceed without ImageNet labels. Please check the connection or URL.")
    st.stop() # Stop the app script if labels can't be loaded

# =========================================
# Tutorial Step 7: File Uploader
# =========================================
uploaded_file = st.file_uploader(
    "Choose an image...",
    type=["jpg", "jpeg", "png"] # Allowed file types
)

# =========================================
# Tutorial Step 8: Process Upload and Display Results
# =========================================
if uploaded_file is not None:
    # Check if model loading was successful before proceeding
    if selected_model is not None and preprocess_pipeline is not None:
        # Read the uploaded image file
        try:
            image = Image.open(uploaded_file)

            # --- Display Uploaded Image ---
            # Use columns for side-by-side layout (can introduce this layout later)
            col1, col2 = st.columns(2)
            with col1:
                st.image(image, caption='Uploaded Image', use_column_width=True)

            # --- Perform Prediction ---
            # Show a spinner while the model is working
            with st.spinner('🧠 Classifying...'):
                # Call the predict function
                predictions = predict(
                    selected_model,
                    preprocess_pipeline,
                    image,
                    imagenet_labels, # Pass the loaded labels
                    top_k=top_k_slider # Pass the selected K value
                )

            # --- Display Predictions ---
            with col2:
                st.subheader(f"Top {top_k_slider} Predictions:")
                if predictions:
                    # Nicely format the output
                    for label, probability in predictions:
                        st.write(f"- {label}: {probability:.2%}") # Format as percentage
                else:
                    # Handle case where prediction might fail (though less likely here)
                    st.error("Prediction failed.")

        except Exception as e:
            st.error(f"Error processing image: {e}")
            st.error("Please try uploading a valid image file (JPG, JPEG, PNG).")
    else:
        # This message shows if model loading failed earlier
        st.error(f"Model '{model_choice}' could not be loaded. Cannot classify the image.")

elif uploaded_file is None:
    # Initial message when no file is uploaded
    st.info("👆 Upload an image file to get started.")

