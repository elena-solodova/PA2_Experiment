############################################################################
## (C)Copyright 2021-2023 Hewlett Packard Enterprise Development LP
## Licensed under the Apache License, Version 2.0 (the "License"); you may
## not use this file except in compliance with the License. You may obtain
## a copy of the License at
##
##    http://www.apache.org/licenses/LICENSE-2.0
##
## Unless required by applicable law or agreed to in writing, software
## distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
## WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
## License for the specific language governing permissions and limitations
## under the License.
############################################################################

import os
import pandas as pd
import re
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler,Normalizer
from sklearn.metrics import accuracy_score, classification_report
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset,TensorDataset, DataLoader,random_split

import datetime

from swarmlearning.pyt import SwarmCallback
from torchvision import datasets, transforms
import time

import torch.nn.functional as F
import glob
import random

import pdb
import re

default_max_epochs = 500
default_min_peers = 2
trainPrint = True
# Gibt an, nach wie vielen Batches Swarm synchronisieren soll.
swSyncInterval = 24

class LinearSVM(nn.Module):
    """
    Linearer SVM, implementiert mit PyTorch.
    Verwendet Hinge Loss für das Training.
    """
    def __init__(self, input_dim):
        super(LinearSVM, self).__init__()
        # Eine lineare Schicht ordnet die Merkmale einem einzelnen Ausgabewert f(x) zu
        self.linear = nn.Linear(input_dim, 1)
    
    def forward(self, x):
        return self.linear(x)

# Hinge Loss für SVM
def hinge_loss(model,outputs, labels, C=1.0):
    """
    Hinge Loss: max(0, 1 - y * f(x))
    C ist der Regularisierungsparameter
    """
    # Labels müssen -1 oder +1 für SVM sein
    labels = 2 * labels - 1  # Wandelt 0/1 in -1/+1 um
    # Standard Hinge Loss Berechnung
    loss = torch.mean(torch.clamp(1 - labels * outputs, min=0))
    
    # L2-Regularisierung nur auf die Gewichte (nicht den Bias)
    l2_reg = torch.tensor(0., requires_grad=True)
    for name, param in model.named_parameters():
        if 'weight' in name:
            l2_reg = l2_reg + torch.norm(param, 2)**2
    
    # Der Faktor (1 / (2 * C)) wird in der primalen SVM-Formulierung verwendet
    return loss + (1 / (2 * C)) * l2_reg



df_all = pd.read_csv('data/all_stat.csv')

X = df_all.drop(columns=["Label", "subject_id"])#.fillna(0) # Handle potential NaNs
y = df_all["Label"].map({"AD": 1, "CN": 0})



# ============================================================
# 4. Datenaufteilung und Skalierung
# ============================================================
scaler = MinMaxScaler() #RobustScaler() # Normalizer() # MinMaxScaler() #StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42, stratify=y
)
print(X_train)
# Umwandlung in Tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

# ============================================================
# 5. Modell und Training
# ============================================================
input_dim = X.shape[1]

# Hyperparameter
C = 1.0  # Regularisierungsparameter
epochs = 50

print("Training Linear SVM mit PyTorch...")
print(f"Input Dimension: {input_dim}")
print(f"Training Samples: {len(X_train)}, Test Samples: {len(X_test)}")
print(f"C Parameter: {C}\n")



def doTrainBatch(model,device,trainLoader,optimizer,epoch,swarmCallback):
    model.train()
    for batchIdx, (data, labels) in enumerate(trainLoader):
        optimizer.zero_grad()
        output = model(data)
        #loss = F.nll_loss(output, target)
        loss = hinge_loss(model,output, labels, C=C)
        loss.backward()
        optimizer.step()
        if trainPrint and batchIdx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                  epoch, batchIdx * len(data), len(trainLoader.dataset),
                  100. * batchIdx / len(trainLoader), loss.item()))
        # Swarm Learning Schnittstelle
        if swarmCallback is not None:
            swarmCallback.on_batch_end()        

def test(model, device, data, true_label):
    model.eval()
    testLoss = 0

    with torch.no_grad():
        output = model(data)
            
        pred = (output > 0).float()
        acc = accuracy_score(true_label, pred)
    print(f"\n{'='*50}")
    print(f"Test Accuracy: {acc:.3f}")
    print(f"{'='*50}\n")
    
    # Detaillierter Klassifikationsbericht
    print("Classification Report:")
    print(classification_report(
        y_test_tensor.numpy(), 
        pred.numpy(), 
        target_names=['CN', 'AD']
    ))
    

def main():
    
    max_epochs = int(os.getenv('MAX_EPOCHS', str(default_max_epochs)))
    min_peers = int(os.getenv('MIN_PEERS', str(default_min_peers)))
    batchSz = 16
    useCuda = torch.cuda.is_available()
    
    if useCuda:
        print("Cuda ist verfügbar")
    else:
        print("Cuda ist nicht verfügbar")
        
    device = torch.device("cuda" if useCuda else "cpu")  
   
    model = LinearSVM(input_dim)
    
    opt = optim.SGD(model.parameters(), lr=0.01, weight_decay=0.0001) 


    trainLoader = torch.utils.data.DataLoader(train_dataset, batch_size=batchSz,shuffle=True)
    testLoader = torch.utils.data.DataLoader(test_dataset, batch_size=batchSz,shuffle=True)
    
    # Erstelle Swarm Callback
    swarmCallback = None

    # Im SwarmCallback werden folgende Parameter bereitgestellt, um den Trainingsfortschritt
    # oder die ETA des Trainings im SLM UI anzuzeigen.
    # 'lossFunction'      - Der String 'lossFunction' sollte mit der Loss-Funktion-Klasse in PyTorch übereinstimmen -
    #                       https://pytorch.org/docs/stable/nn.html#loss-functions
    # 'lossFunctionArgs'  - Dictionary mit benannten Argumenten für lossFunction wie unten gezeigt.
    # 'metricFunction'    - Der String 'metricFunction' sollte mit der Metrik-Funktionsklasse in torchmetrics übereinstimmen -
    #                       https://torchmetrics.readthedocs.io/en/stable/all-metrics.html
    # 'metricFunctionArgs'- Dictionary mit benannten Argumenten für metricFunction wie unten gezeigt.
    # 'totalEpochs'       - Gesamtanzahl der Epochen im lokalen Training.

    lFArgsDict={}
    lFArgsDict['reduction']='sum'

    mFArgsDict={}
    mFArgsDict['task']="multiclass"
    mFArgsDict['num_classes']=2

    swarmCallback = SwarmCallback(syncFrequency=swSyncInterval,
                                  minPeers=min_peers,
                                  useAdaptiveSync=False,
                                  adsValData=testLoader,
                                  adsValBatchSize=batchSz,
                                  model=model,
                                  totalEpochs=max_epochs,
                                  lossFunction="BCEWithLogitsLoss", 
                                  lossFunctionArgs=lFArgsDict,
                                  metricFunction="Accuracy",
                                  metricFunctionArgs=mFArgsDict)
                                  
    # Initialisiere swarmCallback und führe erstes Sync durch
    swarmCallback.on_train_begin()
        
    for epoch in range(1, epochs + 1):
        print('Training beginnt')
        doTrainBatch(model,device,trainLoader,opt,epoch,swarmCallback)   
        print('Test beginnt')   
        test(model,device,X_test_tensor,y_test_tensor)
        swarmCallback.on_epoch_end(epoch)

      
    swarmCallback.on_train_end()

 
if __name__ == '__main__':
  main()
