"""
Reproduce my leaderboard results using the saved model (i.e no model training)
"""
import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt


# load model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(24, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 11),
        )
    def forward(self, x):
        return self.linear_relu_stack(x)

model = NeuralNetwork()
model.load_state_dict(torch.load('mymodel1.pt', weights_only=True))
model.eval()

# load data
data_path = '/kaggle/input/c/stock-price-prediction-challenge/test/'
stock_names = os.listdir(data_path)
stock_names.sort()


Y = []
for jt in range(5):
    df = pd.read_csv(data_path+stock_names[jt],index_col=0)
    
    # feature engineering
    df['Vlog'] = np.log10(df['Volume'].values)
    price_keys = ['Open', 'High', 'Low', 'Close', 'Adjusted', 'Vlog']
    X_keys = price_keys+[]
    y_keys = []

    for key in price_keys:
        X_keys.append(key+'_fl')
        df[X_keys[-1]]=df[key].rolling(6).mean()

    for it in range(1,3):
        for key in price_keys:
            X_keys.append(key+'_%d'%(it))
            df[X_keys[-1]] = df[key].shift(it)

    # y data
    for it in range(1,12):
        y_keys.append('Returns_%d'%(it))
        df[y_keys[-1]] = df['Returns'].shift(-it)

    # extract a single x data
    x = torch.FloatTensor(df.iloc[-1][X_keys].values.astype(float)).unsqueeze(0)

    # model prediction
    with torch.no_grad():
        y_pred=model(x)

    Y.append(y_pred.squeeze(0)) 

Y = np.array(Y).T

# submission file
file = pd.read_csv('/kaggle/input/c/stock-price-prediction-challenge/sample_submission.csv',index_col=0)
for it in range(1,6):
    file['Returns_%d'%(it)] = Y[1:,it-1]
    
file.to_csv('my_submission.csv')
print(file)