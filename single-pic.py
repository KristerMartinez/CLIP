# Imports
import torch
import clip
from PIL import Image
"""
torch: PyTorch is used for tensor operations and for running the model.
clip: The clip module from OpenAI provides access to the CLIP model and its utilities.
PIL (Pillow): Used to load the image that will be processed.
"""

# Load Model and Device Selection
# Choose the model to use; available models are "ViT-B/32", "ViT-B/16", "RN50", etc.
model_name = "ViT-B/32"

# Load the model and tokenizer
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load(model_name, device=device)
'''
Model Choice: The code uses the "ViT-B/32" version of the CLIP model. This is a Vision Transformer-based model that splits the input image into 32x32 patches.
Device Selection: The code checks whether a GPU (CUDA) is available. If yes, it uses "cuda"; otherwise, it falls back to "cpu".
Load Model: The clip.load() function loads both the CLIP model and the preprocessing function (preprocess) for the specified model type.
'''

# Image Loading and Preprocessing
# Load an image from file
image = preprocess(Image.open("MonaLisa-LeonardodaVinci.jpg")).unsqueeze(0).to(device)
'''
Load Image: The image "MonaLisa-LeonardodaVinci.jpg" is loaded using Image.open().
Preprocessing: The image is preprocessed using preprocess to transform it into the appropriate size, color format, and tensor shape for the model.
Add Batch Dimension: .unsqueezme(0) adds a batch dimension to the image tensor. This is needed because the model expects input tensors to have a batch size.
Move to Device: The image is moved to the device (cpu or cuda) to be used for inference.
'''

# Text Prompts Tokenization
# Define text prompts
text = clip.tokenize(
    ("An smiling lady", "A lady posing for a picture", "A lady dressed in old style clothes", "Is this a replica of the MonaLisa", "Lady waiting for her husband", "The Monalisa")).to(device)
'''
Define Prompts: Several text prompts are defined, each representing a different possible description of the image.
Tokenize Prompts: The clip.tokenize() function converts the text prompts into tokens that the model can understand.
Move to Device: The tokenized text prompts are also moved to the appropriate device.
'''

# Model Inference
# Run the model
with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
'''
No Gradient Calculation: torch.no_grad() is used because we are not training the model, so gradients do not need to be calculated, which saves memory and speeds up computation.
Encode Image: The model.encode_image() method generates a feature vector (representation) for the input image.
Encode Text: The model.encode_text() method generates feature vectors for each text prompt.
'''

# Normalize Feature Vectors
# Normalize the features to unit length
image_features /= image_features.norm(dim=-1, keepdim=True)
text_features /= text_features.norm(dim=-1, keepdim=True)
'''
Normalize to Unit Length: The feature vectors are normalized to unit length (magnitude = 1). 
This makes it easier to calculate cosine similarity, as the resulting similarity is in the range between -1 and 1.
'''

# Calculate Cosine Similarity
# Calculate similarity (cosine similarity)
similarity = (image_features @ text_features.T).squeeze(0)
'''
Cosine Similarity: The similarity is calculated using the dot product between the normalized image and text feature vectors. 
Since they are normalized, the dot product is equivalent to the cosine similarity.
Transpose Text Features: .T transposes the text features tensor to make it compatible for matrix multiplication.
Remove Extra Dimension: .squeeze(0) removes the extra batch dimension for easier handling of the similarity scores.
'''

# Print Similarity Scores
# Print out the similarity scores
for i, score in enumerate(similarity):
    print(f"Prompt {i}: {score.item():.4f}")
'''
Iterate Through Scores: The loop iterates through each similarity score and prints it.
Print Scores: The scores represent how similar each text prompt is to the given image, with higher scores indicating a better match.
'''

# Example Output Explanation
'''
If the output shows:

Prompt 0: 0.2534
Prompt 1: 0.2085
Prompt 2: 0.3456
Prompt 3: 0.5789
This means that the fourth prompt, "The Mona Lisa," is the best match for the image (with the highest score), suggesting that the CLIP model associates the image most closely with that description.

Summary
The code loads a pre-trained CLIP model and an image of the Mona Lisa.
It then compares the image to several text descriptions using cosine similarity.
The similarity scores indicate how well the image matches each of the given text prompts, with higher scores indicating stronger similarity.
Let me know if you have more questions or need further explanation!
'''