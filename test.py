from transformers import ConvNextFeatureExtractor, ConvNextForImageClassification
import torch
from PIL import Image
import cv2

# ------------------- STEP 1: DEFINE THE MAPPING -------------------
# Create a dictionary to group the detailed ImageNet classes into the classes you want.
# You can add or remove classes from this list if you wish.
CUSTOM_CLASSES = {
    'car': [
        'sports car, sport car', 'convertible', 'jeep, landrover',
        'limousine, limo', 'minivan', 'racer, race car, racing car',
        'cab, hack, taxi, taxicab', 'ambulance',
        'police van, police wagon, paddy wagon, patrol wagon, wagon, black Maria',
        'recreational vehicle, RV, R.V.', 'station wagon, wagon, estate car, beach wagon, station waggon, waggon',
        'passenger car, coach, carriage'
    ],
    'motorcycle': [
        'motor scooter, scooter', 'moped', 'motorcycle'
    ],
    'bicycle': [
        'mountain bike, all-terrain bike, off-roader', 'bicycle-built-for-two, tandem bicycle, tandem',
        'unicycle, monocycle'
    ],
    'person': [
        'scuba diver', 'groom, bridegroom', 'baseball player, ballplayer', 'skier'
        # Warning: The 'person' class on ImageNet is very limited and not reliable.
    ]
}

# Create a reverse mapping dictionary for faster lookup
# Example: {'sports car, sport car': 'car', 'moped': 'motorcycle', ...}
imagenet_to_custom_map = {}
for custom_class, imagenet_labels in CUSTOM_CLASSES.items():
    for label in imagenet_labels:
        imagenet_to_custom_map[label] = custom_class
# --------------------------------------------------------------------

# 1. Load the model and feature extractor
feature_extractor = ConvNextFeatureExtractor.from_pretrained("facebook/convnext-large-384-22k-1k")
model = ConvNextForImageClassification.from_pretrained("facebook/convnext-large-384-22k-1k")
model.eval()

# 2. Preprocess the image
img_path = "test.jpg"  # <-- Change your image file name here
try:
    img_cv = cv2.imread(img_path)
    if img_cv is None:
        raise FileNotFoundError(f"Cannot read image file: {img_path}")

    img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)

    # The feature extractor will handle the necessary transformations
    inputs = feature_extractor(img_pil, return_tensors="pt")

    # 3. Make a prediction
    with torch.no_grad():
        logits = model(**inputs).logits


    # 4. Get the original result from ImageNet
    prediction_index = logits.argmax(-1).item()
    predicted_imagenet_label = model.config.id2label[prediction_index]


    # ------ STEP 2: APPLY THE MAPPING TO GET THE FINAL RESULT ------
    # Use the created map to find the custom class. If not found, return "Other".
    final_prediction = imagenet_to_custom_map.get(
        predicted_imagenet_label,
        "Not one of the classes of interest (Other)"
    )
    # ----------------------------------------------------------------

    print(f"Original prediction from ConvNext (ImageNet): '{predicted_imagenet_label}'")
    print("-" * 30)
    print(f"==> Result after filtering: {final_prediction}")


except FileNotFoundError as e:
    print(e)
    print("Please check the path and file name 'test.jpg' again.")