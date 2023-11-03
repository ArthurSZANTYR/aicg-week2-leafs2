import matplotlib.pyplot as plt
import os
import pandas as pd
import torch 
import numpy as np
from tqdm import tqdm
from itertools import cycle

import albumentations as A
from albumentations.pytorch import ToTensorV2

from torch.utils.data import (DataLoader, Dataset)
import torchvision.transforms as T

import torch.nn as nn

from torch.optim import Optimizer
import torch.nn.functional as F
from itertools import cycle

from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

from torchvision import models


class LeafDataset(Dataset):
    def __init__(self) -> None:
        super().__init__()

        self.path = "dataset/plant-pathology-2020-fgvc7"

        self.path_dir_X = os.path.join(self.path, 'images')

        self.path_Y = os.path.join(self.path, 'train.csv')
        self.dataframe_Y = pd.read_csv(self.path_Y)
        self.labels = self.dataframe_Y.loc[:, 'healthy':'scab']

        self.transform = A.Compose([
        A.RandomResizedCrop(height=200, width=200, p=1.0), #au lieu de 500 - sinon le cpu ne suit pas - et kill tout les kernels python
        A.Rotate(20, p=1.0), 
        A.Flip(p=1.0),
        A.Transpose(p=1.0), 
        A.Normalize(p=1.0),  
        ToTensorV2(p=1.0),
        ], p=1.0)

    def __getitem__(self, index): #on defini le dataset et les transfo ici - car quand je vais appeller le dataloader - ca va parcourir toutes le simages donc passé par getitem
        img_name = self.dataframe_Y.loc[index, 'image_id']   #image_id,healthy,multiple_diseases,rust,scab
                                                             #Train_0,0,0,0,1      -> dans train.csv on a le nom du fichier ex : df[0]['image_id] = Train_0
        img_path = f"{self.path_dir_X}/{img_name}.jpg"
        image = plt.imread(img_path)

        image = self.transform(image = image)['image'] #resize / normalized / ....  #on prend ["image"] car renvoi un dictionnaire a la base 
        
        #test pour voir image de sorti
        #permute_transfo_image = image.permute(1, 2, 0)   #pour pouvoir l'afficher en plotlib
        #plt.imshow(permute_transfo_image)
        #plt.show()

        label = torch.tensor(np.argmax(self.labels.loc[index,:].values))  #on obtient la label avec argmax
        #print(f'label : {label}')   #maintenant on aplus que la label et plus le tableau 
                                    #on peut maitnent calculer une loss - on pouvait pas avant avec array : tensor([0, 0, 1, 0])

        return image, label
    
    def __len__(self):
        return len(self.dataframe_Y)

class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.model = models.resnet18(pretrained= True)   #le cnn #on utilise fine tunning pour le cnn et la detection de features - resnet
        for param in self.model.parameters():
            param.requires_grad = False         #on ne suit pas le gradient de cette couche - on ne modifie pas ses poids - pour etre plus rapide

        out_features_cnn = self.model.fc.in_features
        out_features_model = 4  #le nombre de classe 

        #on def le nombre de couches
        self.model.fc = nn.Linear(out_features_cnn,out_features_model)

    def forward(self, X):
        X = self.model(X)   
        #X = self.fc1(X)
        return X

    def fit_model(self, loader: DataLoader, optimizer: Optimizer, scheduler, epochs: int):
        self.model.train()   
        batches = iter(cycle(loader))
        for _ in tqdm(range(epochs * len(loader)), desc= 'fitting'):
            batch_X, batch_Y = next(batches) 
            batch_Y_pred = self(batch_X)   #forward pass
            loss = nn.CrossEntropyLoss()(batch_Y_pred, batch_Y)     #cross entropy fait direct le softmax
            loss.backward() 
            optimizer.step() 
            scheduler.step()
            optimizer.zero_grad(set_to_none=True) 

if __name__ == '__main__':

    epochs = 2    #pour reduire le temps avec le cpu - sinon trop long
    batch_size = 12
    lr = 1e-2
    
    train_set = LeafDataset()
    train_loader = DataLoader(train_set, batch_size= batch_size, shuffle= True)

    model = Model()
    
    optimizer = AdamW(model.parameters(), lr=lr, betas=(0.7, 0.9)) 
    scheduler = OneCycleLR(optimizer, max_lr=lr, total_steps=int(((len(train_set) - 1) // batch_size + 1) * epochs))
    
    model.fit_model(train_loader, optimizer, scheduler, epochs)
    torch.save(model.state_dict(), 'leaf_cnn_mlp.pt')
    
    