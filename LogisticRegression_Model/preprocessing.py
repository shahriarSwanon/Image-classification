import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import torch
import torchvision.transforms as transforms
import torchvision.models as models

def load_data(dataset_dir, img_size):
    print(f"Loading images and extracting deep features from: {dataset_dir}")
    
    # 1. Setup Pre-trained Feature Extractor (MobileNetV2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} for feature extraction...")
    
    # Load MobileNetV2 and grab just the feature layers (ignore the classifier)
    weights = models.MobileNet_V2_Weights.IMAGENET1K_V1
    feature_extractor = models.mobilenet_v2(weights=weights).features
    pool = torch.nn.AdaptiveAvgPool2d((1, 1)) # Flattens the feature map
    
    feature_extractor.to(device)
    feature_extractor.eval() # Set to evaluation mode

    # 2. Image Preprocessing matching the model's requirements
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    X = []
    y = []
    class_names = []
    
    # Iterate through all category folders in the dataset directory
    for label_idx, class_name in enumerate(sorted(os.listdir(dataset_dir))):
        class_dir = os.path.join(dataset_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
            
        class_names.append(class_name)
        print(f"Processing class: {class_name}")
        
        with torch.no_grad():
            for file_name in os.listdir(class_dir):
                file_path = os.path.join(class_dir, file_name)
                try:
                    with Image.open(file_path).convert('RGB') as img:
                        # Process image to PyTorch tensor
                        input_tensor = preprocess(img).unsqueeze(0).to(device)
                        
                        # Extract features (shape output is typically [1, 1280, 7, 7])
                        features = feature_extractor(input_tensor)
                        
                        # Pool down to exactly 1280 robust features
                        features_pooled = pool(features).flatten().cpu().numpy()
                        
                        X.append(features_pooled)
                        y.append(label_idx)
                except Exception as e:
                    print(f"Skipping {file_name}: {e}")
                
    return np.array(X), np.array(y), class_names

def prepare_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
