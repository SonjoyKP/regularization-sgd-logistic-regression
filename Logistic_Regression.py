from libauc.losses import CrossEntropyLoss

import torch
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import top_k_accuracy_score
from sklearn.datasets import load_svmlight_file
from sklearn.preprocessing import StandardScaler
import numpy as np

"""
    This script provides an example of how to learn a logistic regression model for a binary classification problem
    on the "breast-cancer" dataset using SGD optimizer with a constant learning rate. You are asked to complete the 
    code and explore different schedulers and compare them. 
    Please try the following 3 schedulers:
    step decay, cosine decay, polynomial decay. You should implement schedulers yourself.
"""

class LibSVMDataset(Dataset):
    def __init__(self, data, targets):
       self.data = data
       self.targets = targets
       self.targets[targets==2] = 0 # convert 2 to 0 for breast-cancer dataset only
       self.targets[targets==4] = 1 # convert 4 to 1 for breast-cancer dataset only

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        data = self.data[idx].astype(np.float32)
        target = self.targets[idx].astype(np.int64)
        return data, target

#A mini example of SGD optimizer
class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=0.1, weight_decay=0.0):
        self.lr = lr
        defaults = dict(lr=lr, weight_decay=weight_decay)
        super(SGD, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            self.lr = group['lr']
            for p in group['params']:
                d_p = p.grad
                # update the model parameters
                ### YOUR CODE HERE

                ### YOUR CODE HERE

        return loss

#A simple linear model
class linear_model(torch.nn.Module):
    def __init__(self, input_dim=10, output_dim=1):
        super().__init__()
        self.classifer = torch.nn.Linear(input_dim, output_dim)
    def forward(self, x):
        return self.classifer(x)
    
# Hyper Parameters
BATCH_SIZE = 16
total_epochs = 30
lr = 1

def main():
    # load data, labels as numpy arrays
    X_train, y_train = load_svmlight_file("breast-cancer_scale")

    # normalization
    scaler = StandardScaler()
    X_train = X_train.toarray()
    X_train = scaler.fit_transform(X_train)

    # dataset && dataloaders
    trainSet = LibSVMDataset(X_train, y_train)
    trainloader = DataLoader(trainSet, batch_size=BATCH_SIZE, shuffle=True)

    # define model
    feat_dim = X_train.shape[-1]
    model = linear_model(input_dim=feat_dim, output_dim=1)

    # define loss and optimizer
    loss_fn = CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=lr)

    print ('Start Training')
    print ('-'*30)

    train_log = []
    for epoch in range(total_epochs):
        # training
        train_loss = []
        model.train()
        for data, targets in trainloader:
            y_pred = model(data)
            loss = loss_fn(y_pred, targets.float().reshape(-1, 1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
            #Decay the learning rate per iteration
            ### YOUR CODE HERE

            ### YOUR CODE HERE

        # evaluation
        model.eval()
        train_pred_list = []
        train_true_list = []
        for train_data, train_targets in trainloader:
            train_pred = model(train_data)
            train_pred_list.append(train_pred.detach().numpy())
            train_true_list.append(train_targets.numpy())
        train_true = np.concatenate(train_true_list)
        train_pred = np.concatenate(train_pred_list)
        train_acc = top_k_accuracy_score(train_true, train_pred, k=1)
        train_loss = np.mean(train_loss)

        # print results
        print("epoch: %s, train_loss: %.4f, train_acc: %.4f lr: %.4f"%(epoch, train_loss, train_acc, optimizer.lr))
        train_log.append(train_loss)


    # Plot the Training loss curve
    ### YOUR CODE HERE

    ### YOUR CODE HERE


main()