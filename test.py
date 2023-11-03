import torch
import matplotlib.pyplot as plt
import os
import albumentations as A
from albumentations.pytorch import ToTensorV2

import warnings  
warnings.filterwarnings('ignore')  #pour enlever les warning message

from glob import glob #pour parcourir dossier

# Import your custom model class
from train import Model

# Define the same transformation as in your training script
transform = A.Compose([
            A.RandomResizedCrop(height=200, width=200, p=1.0), #au lieu de 500 - sinon le cpu ne suit pas - et kill tout les kernels python
            A.Transpose(p=1.0), 
            A.Normalize(p=1.0),  
            ToTensorV2(p=1.0),
            ], p=1.0)

# Define a function to perform inference
def predict(model, image_path, transform):
    # Load and preprocess the image
    image = plt.imread(image_path)
    image = transform(image=image)['image']
    image = image.unsqueeze(0)  # Add a batch dimension

    # Make a prediction
    with torch.no_grad():    #inference mode ppur la pred
        output = model(image)

    # Convert the output to probabilities or class predictions as needed
    probabilities = torch.softmax(output, dim=1)
    predicted_class = torch.argmax(output, dim=1)

    return probabilities, predicted_class


if __name__ == '__main__':
    # Load your trained model
    model_path = 'leaf_cnn_mlp.pt'
    model = Model()
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set the model to evaluation mode

    # Example inference on an image
    image_folder = 'dataset/plant-pathology-2020-fgvc7/images'
    image_files = glob(os.path.join(image_folder, '*.jpg'))  #pour creer un objet pour parcourir toutes les images "fichier qui termine par jpg"

    for image_file in image_files:
        file_name = os.path.basename(image_file) 

        probabilities, predicted_class = predict(model, image_file, transform)
        print(f'{file_name} : {predicted_class}')
   
