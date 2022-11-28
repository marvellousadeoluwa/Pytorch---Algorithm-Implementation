
# imorting libraries
import torch
import torch.nn as nn

import numpy as np

from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
#%matplotlib inline

# prepare dataset
bc = datasets.load_breast_cancer()

X, y = bc.data, bc.target

n_samples, n_features = X.shape
print(n_samples, n_features)

# split data
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                    test_size=0.2,
                                    random_state=42)

# scale
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# convert numpy to tensor
# convert to tensor
X_train = torch.from_numpy(X_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32))
y_test = torch.from_numpy(y_test.astype(np.float32))

# reshape target to one col
y_train = y_train.view(y_train.shape[0], 1)
y_test = y_test.view(y_test.shape[0], 1)


# Step 1 - Model
# NB: model - linear combo of weight and bias

class LogisticRegression(nn.Module):
    def __init__(self, n_input_features):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(n_input_features, 1) # one class label at the end
        
    def forward(self, x):
        
        y_predicted = torch.sigmoid(self.linear(x))
        
        return y_predicted
    
model = LogisticRegression(n_features)

# Step 2 - Loss and Optimizer
learning_rate = 0.01 
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(),
                            lr = learning_rate)

# Step 3 - Training Loop
num_epochs =  100
print('Training...')
for epoch in range(num_epochs):

    # forward pass
    y_prediction = model(X_train)
    loss = criterion(y_prediction, y_train)

    # backward pass
    loss.backward()

    # weight update
    optimizer.step()

    optimizer.zero_grad()

    if (epoch + 1) % 10 == 0:
        print(f'epoch: {epoch + 1}, loss = {loss.item():.4f}')

# model Evaluation - accuracy

with torch.no_grad():
    y_test_pred = model(X_test)
    y_test_pred_cls = y_test_pred.round()

    # accuracy
    acc = y_test_pred_cls.eq(y_test).sum(

    ) / float(y_test.shape[0]) * 100
    print(f'Accuracy = {acc:.2f} %')
