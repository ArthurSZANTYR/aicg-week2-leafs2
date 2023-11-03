import torch
from torchvision import models
import torchvision.transforms as T
import matplotlib.pyplot as plt
import os
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Import your custom model class
from train import LeafDataset

# Define a function to load the model
def load_model(model_path):
    model = models.resnet18(pretrained=False)  # Create a ResNet model (same architecture as trained)
    in_features_cnn = model.fc.in_features
    model.fc = torch.nn.Linear(in_features_cnn, out_features=4)  # Change the last layer to match your number of classes
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set the model to evaluation mode
    return model

# Define a function to perform inference
def predict(model, image_path, transform):
    # Load and preprocess the image
    image = plt.imread(image_path)
    image = transform(image=image)['image']
    image = image.unsqueeze(0)  # Add a batch dimension

    # Make a prediction
    with torch.no_grad():
        output = model(image)

    # Convert the output to probabilities or class predictions as needed
    probabilities = torch.softmax(output, dim=1)
    predicted_class = torch.argmax(output, dim=1)

    return probabilities, predicted_class

if __name__ == '__main__':
    # Load your trained model
    model_path = 'leaf_cnn_mlp.pt'
    model = load_model(model_path)

    # Define the same transformation as in your training script
    transform = A.Compose([
        A.RandomResizedCrop(height=200, width=200, p=1.0), #au lieu de 500 - sinon le cpu ne suit pas - et kill tout les kernels python
        A.Transpose(p=1.0), 
        A.Normalize(p=1.0),  
        ToTensorV2(p=1.0),
        ], p=1.0)
    # Example inference on an image
    image_path = 'dataset/plant-pathology-2020-fgvc7/images/Train_4.jpg'
    probabilities, predicted_class = predict(model, image_path, transform)

    # You can use the predicted_class and probabilities as needed in your application
    print(f'Predicted Class: {predicted_class.item()}')
    print(f'Probabilities: {probabilities.numpy()}')
