import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch import optim
from torchvision.models import resnet18
from torchvision.transforms import transforms
from load_cifar10_data import get_data, get_test_data

class CustomTensorDataset(TensorDataset):
    """Dataset wrapping tensors.
    Extended to support image transformations
    """

    def __init__(self, *tensors, transform):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors
        self.transform = transform

    def __getitem__(self, index):
        pictures = self.transform(self.tensors[0][index])
        labels = self.tensors[1][index]
        return pictures, labels

    def __len__(self):
        return self.tensors[0].size(0)


'''Reference blog post https://towardsdatascience.com/transfer-learning-with-convolutional-neural-networks-in-pytorch-dd09190245ce'''
use_gpu = torch.cuda.is_available()

# pick a trained model from pytorch
model = resnet18(pretrained=True)

# We can choose to freeze model weights, by setting required grad to False
for param in model.parameters():
    param.requires_grad = True

# change the classifier, map the internal values to 10 output classes
model.fc = nn.Linear(512, 10)

if use_gpu:
    model.cuda()

# Find total parameters and trainable parameters(most paramaters are not trainable, which speed up training a lot)
total_params = sum(p.numel() for p in model.parameters())
print(f'{total_params:,} total parameters.')
total_trainable_params = sum(
    p.numel() for p in model.parameters() if p.requires_grad)
print(f'{total_trainable_params:,} training parameters.')

# Loss and optimizer
criteration = nn.CrossEntropyLoss()
optimizer = optim.Adadelta(model.parameters(), lr=0.001)
# dynamically reduce lr if loss not improving
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2000, verbose=True)

# train the model
data = get_data()
test_data = get_test_data()
X_train = data[b'data']
X_test = test_data[b'data']
# transpose for pytorch
X_train = np.transpose(X_train, (0, 3, 1, 2))
X_test = np.transpose(X_test, (0, 3, 1, 2))
y_train = data[b'labels']
y_test = test_data[b'labels']

# normalize data(this is a very important step, which increases accuracy a lot), also according to pytorch docu, image size needs to be at least 224
# https://pytorch.org/docs/stable/torchvision/models.html
normalize = transforms.Normalize(mean=[0.45, 0.45, 0.45],
                                 std=[0.226, 0.226, 0.226])
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(224),
    transforms.ToTensor(),
    normalize
])
# create data and dataloder
data = CustomTensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).long(), transform=transform)
test_data = CustomTensorDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).long(), transform=transform)
dataloader = DataLoader(data, batch_size=64, shuffle=True, pin_memory=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True, pin_memory=True)

reuse = True

if os.path.exists('model.pt') and reuse:
    print('Loading model')
    model = torch.load('model.pt',map_location=lambda storage, loc: storage)

else:
    print('Training model')
    # set model for training
    model.train()
    for epoch in range(10):
        t_loss = 0.0
        total_train_images = 0.0
        total_train_correct = 0.0
        total_test_images = 0.0
        total_test_correct = 0.0

        for data, targets in dataloader:
            if use_gpu:
                data = data.cuda()
                targets = targets.cuda()

            # Generate predictions
            out = model(data)
            # Calculate loss
            labels = torch.argmax(targets, dim=1)
            loss = criteration(out, labels)
            right_count = float(torch.sum(torch.argmax(out, dim=1) == labels))

            loss.backward()
            # Update model parameters
            optimizer.step()
            scheduler.step(loss)

            t_loss += loss.item()
            total_train_correct += right_count
            total_train_images += data.shape[0]

        # eval using test set
        model.eval()
        for data, targets in test_dataloader:
            if use_gpu:
                data = data.cuda()
                targets = targets.cuda()

            # Generate predictions
            out = model(data)
            labels = torch.argmax(targets, dim=1)
            right_count = float(torch.sum(torch.argmax(out, dim=1) == labels))
            total_test_correct += right_count
            total_test_images += data.shape[0]

        model.train()
        torch.save(model, 'model.pt')
        print(
            f'{epoch} - Training Loss: {t_loss}, Training Accuracy: {total_train_correct / total_train_images}, Test Accuracy: {total_test_correct / total_test_images}')

# set model for evaluation(changes batch layers etc.)
model.eval()
