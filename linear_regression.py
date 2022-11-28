
# imorting libraries
import torch
import torch.nn as nn

import numpy as np
from sklearn import datasets

import matplotlib.pyplot as plt
#%matplotlib inline

# Data
X_numpy, y_numpy = datasets.make_regression(n_samples=100,
                                            n_features=1,
                                            noise=20,
                                           random_state=42)
# convert numpy to tensor
X = torch.from_numpy(X_numpy.astype(np.float32))
y = torch.from_numpy(y_numpy.astype(np.float32))

# reshape
y = y.view(y.shape[0], 1) # 1 column

n_samples, n_features = X.shape

print(n_samples, n_features)

# Step 1 - Model

model = nn.Linear(in_features = n_features, 
                    out_features = 1)


# Step 2 - Loss and Optimizer
learning_rate = 0.1 
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(),
                            lr = learning_rate)

# Step 3 - Training Loop
num_epochs =  100
print('Training...')
for epoch in range(num_epochs):
    # forward pass
    y_prediction = model(X)
    loss = criterion(y_prediction, y)

    # backward pass
    loss.backward()

    # weight update
    optimizer.step()

    optimizer.zero_grad()

    if (epoch + 1) % 10 == 0:
        print(f'epoch: {epoch + 1}, loss = {loss.item():.4f}')

# plotting regression

print("\n Plotting...")
predicted = model(X).detach()  # makes requires_grad = False

plt.plot(X_numpy, y_numpy, 'ro')
plt.plot(X_numpy, predicted, 'b')

plt.show()

