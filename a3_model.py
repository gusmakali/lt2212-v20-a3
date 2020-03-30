import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch 
from torch import nn 
from torch import optim
from sklearn import preprocessing 
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from collections import OrderedDict
import csv 
from sklearn.utils import shuffle
from sklearn.metrics import classification_report, accuracy_score
# Whatever other imports you need

# You can implement classes and helper functions here too.

def read_file_df(file):
    with open(file, "r") as data:
        df = pd.read_csv(data)
    
    return df

rd = np.random.RandomState()

def sample(dt):
    d = dt.sample(n=2, random_state=rd)
    doc1 = d.iloc[0].values.tolist()
    doc2 = d.iloc[1].values.tolist()
    
    if doc1[-1] == doc2[-1]:
        return (doc1[:-1], doc2[:-1], 1)
    else:
        return (doc1[:-1], doc2[:-1], 0)
        
def build_samples(dt, n= 1000):
    sample_0 = []
    sample_1 = []

    while len(sample_1) < n or len(sample_0) <n:
        s = sample(dt)
        if s[2] == 1 and s not in sample_1 and len(sample_1) < n:
            sample_1.append(s)
        elif s[2] == 0 and s not in sample_0 and len(sample_0) < n:
            sample_0.append(s)

    return sample_0 + sample_1

class Train_Data(Dataset):
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data
    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]
    def __len__(self):
        return len(self.X_data)

class Test_Data(Dataset):
    def __init__(self, X_data):
        self.X_data = X_data
    def __getitem__(self, index):
        return self.X_data[index]

    def __len__(self):
        return len(self.X_data)

class ClassModel(nn.Module):
    def __init__(self, input, hidden = None, nonlin = None):
        super(ClassModel,self).__init__()

        if hidden is not None:
            self.layer_1 = nn.Linear(input, hidden)
            self.layer_hidden = nn.Linear(hidden, hidden)
            self.layer_out = nn.Linear(hidden, 1)

        else:
            self.layer_1 = nn.Linear(input, 64)
            self.layer_hidden = None
            self.layer_out = nn.Linear(64,1)
        
        if nonlin == 1:
            self.nonlin = nn.ReLU()
        else:
            self.nonlin = nn.Softmax(dim=1)
    def forward(self, inputs):
        x = self.layer_1(inputs)
        if self.layer_hidden is not None:
            x = self.nonlin(x)
        x = self.layer_out(x) 
        return x

def actual_acc(y_pred, y_test):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))

    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum/y_test.shape[0]
    acc = torch.round(acc * 100)
    
    return acc


def build_train_model(featurefile, hidden_layers=None, choice=None, should_print=True):
    if should_print:
        print("Reading {}...".format(featurefile))

    df = read_file_df(featurefile)

    train = df[df.columns.difference(['target'])][df['target'] == 'train']
    test = df[df.columns.difference(['target'])][df['target'] == 'test']

    train_samples = shuffle(build_samples(train, 1600))
    test_samples = shuffle(build_samples(test, 400))
    
    X_train = np.array([v[0] + v[1] for v in train_samples])
    X_test = np.array([v[0] + v[1] for v in test_samples]) 
    y_train = np.array([v[2] for v in train_samples])
    y_test = np.array([v[2] for v in test_samples])
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)

    epochs = 30
    batch_size = 32 
    learning_rate = 0.001

    train_data = Train_Data(torch.FloatTensor(X_train),
                        torch.FloatTensor(y_train))

    test_data = Test_Data(torch.FloatTensor(X_test))
    train_load  = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    test_load = DataLoader(dataset=test_data, batch_size=1)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = ClassModel(X_train.shape[1], hidden_layers, choice)
    model.to(device)
    if should_print:
        print(model)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate) # stochastic optim 

    # model.train()
    
    for e in range(1, epochs+1):  
        epochs_l = 0
        epochs_acc = 0
        for batch in train_load: 
            model.train()
            X_batch, y_batch = batch 
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_batch = torch.unsqueeze(y_batch, 1)

            y_pred = model(X_batch)
            
            l = criterion(y_pred, y_batch)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()

            acc = actual_acc(y_pred, y_batch)
            epochs_l += l.item()
            epochs_acc += acc.item()    

        if should_print:
            print("Epoch {}:  |  Loss: {}:  |  Accuracy: {}".format(e, epochs_l/len(train_load), epochs_acc/len(train_load)))
    
    y_pred_list = []
    model.eval()
    for_bonus = []
    with torch.no_grad():
        for X_batch in test_load:
            X_batch = X_batch.to(device)

            y_pred = model(X_batch)
           
            y_pred = torch.sigmoid(y_pred)
            
            y_pred_tag = torch.round(y_pred)
            y_pred_list.append(y_pred_tag.cpu().numpy())
    
    y_pred_list = [a.squeeze().tolist() for a in y_pred_list]
    
    r1 = classification_report(y_test, y_pred_list, output_dict=True)
    r2 = accuracy_score(y_test, y_pred_list)
    if should_print:
        print(pd.DataFrame(r1), "\naccuracy:", r2)

    return (model, y_pred_list, y_test)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and test a model on features.")
    parser.add_argument("featurefile", type=str, help="The file containing the table of instances and features.")
    parser.add_argument("--hl", type=int, help="size of hidden layer, if not given, then no hl")
    parser.add_argument("--choice", type=int, help="choice 1 (relu), or 2 (softmax) of non linearity, if hd is activated")

    # Add options here for part 3 -- hidden layer and nonlinearity,
    # and any other options you may think you want/need.  Document
    # everything.
    
    args = parser.parse_args()

    build_train_model(args.featurefile, args.hl, args.choice)
