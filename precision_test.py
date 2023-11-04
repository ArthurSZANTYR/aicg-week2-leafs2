import torch
import matplotlib.pyplot as plt
import os
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader

import warnings  
warnings.filterwarnings('ignore')  #pour enlever les warning message

from glob import glob #pour parcourir dossier

from train import Model, LeafDataset


# Define a function to perform inference
def predict(model, image):
    # Load and preprocess the image
    #image = plt.imread(image_path)
    #les images sont déja ransformé avec l'appel de LeafDataset
    print(image.shape)
    image = image.unsqueeze(0)  # Add a batch dimension

    with torch.no_grad():    #inference mode ppur la pred
        output = model(image)

    probabilities = torch.softmax(output, dim=1)
    predicted_class = torch.argmax(output, dim=1)

    return probabilities, predicted_class


if __name__ == '__main__':

    test_set = LeafDataset()

    model_path = 'leaf_cnn_mlp.pt'
    model = Model()
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set the model to evaluation mode

    correct_predictions = 0
    total_predictions = 0

    for index in range(len(test_set)-1):
        image, label = test_set[index]

        _, predicted_class = predict(model, image)
        true_class = label.item()


        if true_class == predicted_class.item():
            correct_predictions += 1
        
        total_predictions += 1
        print(total_predictions)  #pour suivre l'évolution

    accuracy = correct_predictions / total_predictions
    print(f'Accuracy: {accuracy * 100:.2f}%')


    ## Example inference on an image
    #image_folder = 'dataset/plant-pathology-2020-fgvc7/images'
    #image_files = glob(os.path.join(image_folder, 'Train*'))  #pour creer un objet pour parcourir toutes les images "fichier qui termine par jpg"
#
    #for image_file in image_files:
    #    file_name = os.path.basename(image_file) 
#
    #    probabilities, predicted_class = predict(model, image_file, transform)
    #    print(f'{file_name} : {predicted_class}')
   
