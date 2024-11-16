import os
import torch
import torch.nn as nn
import torch.nn.functional as F  # Import for softmax function
from torchvision.models import resnet18, ResNet18_Weights
import streamlit as st
from PIL import Image
from torchvision import transforms

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load class names (ensure this matches your model's output)
class_names = ['daisy', 'dandelion']  # Update as per your classes

# Load the pre-trained model
model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(class_names))
model.load_state_dict(torch.load('flower_classifier.pth', map_location=device))
model = model.to(device)
model.eval()  # Set model to evaluation mode

# Define transformations for input image
data_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Streamlit interface v1
st.title("Flower Classifier: Dandelion vs Daisy")

# File uploader
uploaded_file = st.file_uploader("Choose an image of a flower", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open and display the image
    image = Image.open(uploaded_file).convert("RGB") # Convert image to RGB to ensure 3 channels

    # Preprocess the image
    input_tensor = data_transforms(image).unsqueeze(0).to(device)

    # Make prediction
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = F.softmax(outputs, dim=1)  # Calculate softmax probabilities
        _, preds = torch.max(outputs, 1)
        confidence = probabilities[0][preds[0]].item()  # Get the confidence score for the predicted class

    # Display prediction with confidence percentage
    # label = class_names[preds[0]]
    # st.write(f"Prediction: {label} ({confidence * 100:.2f}% confidence)")

    if confidence >= 0.85:
        label = class_names[preds[0]]
        st.write(f"Prediction: {label} ({confidence * 100:.2f}% confidence)")
    else:
        st.write(f"The model is not confident that the image with a confidence level of {confidence * 100:.2f}% is a flower (daisy or dandelion).")


    # Display the image please
    st.image(image, caption='Uploaded Image', use_column_width=True)

