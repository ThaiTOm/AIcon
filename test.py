# ==============================================================================
# IMPORT LIBRARIES
# ==============================================================================
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
import sys

# ==============================================================================
# CONFIGURATION
# ==============================================================================
# Path to the trained weights file saved from train_vit.py
WEIGHTS_PATH = "bike_motorbike_vit_weights.pth"

# Path to the image you want to classify
# --->>> CHANGE THIS TO YOUR IMAGE FILE <<<---
IMAGE_TO_TEST = "test.jpg"

# The classes the model was trained on. MUST match the order from training.
CLASS_NAMES = ['bike', 'motorbike']

# Device (use GPU if available)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")


# ==============================================================================
# MODEL AND TRANSFORMS DEFINITION
# ==============================================================================

def load_trained_model(weights_path, num_classes):
    """
    Loads the ViT model architecture and the trained weights.
    """
    # Load the ViT-B-16 model structure without pre-trained weights
    model = models.vit_b_16(weights=None)

    # Replace the final classification layer to match our number of classes (2)
    num_ftrs = model.heads.head.in_features
    model.heads.head = nn.Linear(num_ftrs, num_classes)

    # Load the state dictionary from the saved weights file
    # map_location ensures it works even if you trained on GPU and are testing on CPU
    model.load_state_dict(torch.load(weights_path, map_location=DEVICE))

    # Move the model to the selected device
    model = model.to(DEVICE)

    # Set the model to evaluation mode (important for inference)
    model.eval()

    print("Model loaded and set to evaluation mode.")
    return model


def get_image_transforms():
    """
    Returns the same image transformations used for the validation set during training.
    This ensures the input for inference is processed identically.
    """
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


# ==============================================================================
# PREDICTION FUNCTION
# ==============================================================================

def predict_image(model, image_path, transforms, class_names):
    """
    Takes a model and an image path, and returns the predicted class and confidence.
    """
    try:
        # Open the image using PIL
        img = Image.open(image_path).convert('RGB')
    except FileNotFoundError:
        print(f"Error: Image file not found at '{image_path}'")
        return None, None

    # Apply the transformations to the image
    img_transformed = transforms(img)

    # Add a batch dimension (models expect a batch of images)
    # The result should be of shape [1, 3, 224, 224]
    batch_img = img_transformed.unsqueeze(0)

    # Move the input tensor to the device
    batch_img = batch_img.to(DEVICE)

    # Make a prediction without calculating gradients
    with torch.no_grad():
        outputs = model(batch_img)

        # Apply Softmax to get probabilities
        probabilities = torch.nn.functional.softmax(outputs, dim=1)

        # Get the top prediction
        confidence, predicted_index = torch.max(probabilities, 1)

    # Get the class name using the predicted index
    predicted_class = class_names[predicted_index.item()]

    return predicted_class, confidence.item()


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

if __name__ == '__main__':
    # 1. Check if the weights file exists
    if not os.path.exists(WEIGHTS_PATH):
        print(f"Error: Weights file not found at '{WEIGHTS_PATH}'")
        print("Please make sure you have run the training script and the file is in the correct location.")
        sys.exit(1)
    # 2. Load the trained model
    model = load_trained_model(WEIGHTS_PATH, len(CLASS_NAMES))

    # 3. Get the necessary image transformations
    image_transforms = get_image_transforms()

    # 4. Perform the prediction on the test image
    predicted_class, confidence = predict_image(model, IMAGE_TO_TEST, image_transforms, CLASS_NAMES)

    # 5. Display the result
    if predicted_class and confidence:
        print("\n" + "=" * 30)
        print("          PREDICTION RESULT")
        print("=" * 30)
        print(f"Image:     '{IMAGE_TO_TEST}'")
        print(f"Predicted: '{predicted_class}'")
        print(f"Confidence: {confidence:.2%}")
        print("=" * 30)