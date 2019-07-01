from dataset import FashionDataset
from model import get_instance_segmentation_model
import torch
from engine import train_one_epoch, evaluate
import utils
from apex import amp
import transforms as T


def get_transform(train):
    transforms = []
    # converts the image, a PIL image, into a PyTorch Tensor
    transforms.append(T.ToTensor())
    if train:
        # during training, randomly flip the training images
        # and ground-truth for data augmentation
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

num_classes = 46 + 1
device = torch.device('cuda:0')

dataset_train = FashionDataset("../input/train/", "../input/train_kfolds.csv", 1024, 1024,
                               folds=[0, 1, 2, 3, 4], transforms=get_transform(train=True))
#dataset_val = FashionDataset("../input/train/", "../input/train_kfolds.csv", 512, 512,
#                             folds=[0], transforms=get_transform(train=False))

model_ft = get_instance_segmentation_model(num_classes)
model_ft.to(device)

# define training and validation data loaders
data_loader = torch.utils.data.DataLoader(
    dataset_train, batch_size=4, shuffle=True, num_workers=8,
    collate_fn=utils.collate_fn)

#data_loader_test = torch.utils.data.DataLoader(
#    dataset_val, batch_size=1, shuffle=False, num_workers=4,
#    collate_fn=utils.collate_fn)

# construct an optimizer
params = [p for p in model_ft.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005,
                            momentum=0.9, weight_decay=0.0005)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                               step_size=5,
                                               gamma=0.1)
num_epochs = 8
for epoch in range(num_epochs):
    # train for one epoch, printing every 10 iterations
    train_one_epoch(model_ft, optimizer, data_loader, device, epoch, print_freq=10)
    # update the learning rate
    lr_scheduler.step()
    torch.save(model_ft.state_dict(), "model.bin")
    # evaluate on the test dataset
    # evaluate(model_ft, data_loader_test, device=device)
torch.save(model_ft.state_dict(), "model.bin")

optimizer = torch.optim.SGD(params, lr=0.0005,
                            momentum=0.9, weight_decay=0.0005)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                               step_size=5,
                                               gamma=0.1)

num_epochs = 8
for epoch in range(num_epochs):
    # train for one epoch, printing every 10 iterations
    train_one_epoch(model_ft, optimizer, data_loader, device, epoch, print_freq=10)
    # update the learning rate
    lr_scheduler.step()
    torch.save(model_ft.state_dict(), "model.bin")
    # evaluate on the test dataset
    # evaluate(model_ft, data_loader_test, device=device)
torch.save(model_ft.state_dict(), "model.bin")

optimizer = torch.optim.SGD(params, lr=0.00005,
                            momentum=0.9, weight_decay=0.0005)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                               step_size=5,
                                               gamma=0.1)
num_epochs = 8
for epoch in range(num_epochs):
    # train for one epoch, printing every 10 iterations
    train_one_epoch(model_ft, optimizer, data_loader, device, epoch, print_freq=10)
    # update the learning rate
    lr_scheduler.step()
    torch.save(model_ft.state_dict(), "model.bin")
    # evaluate on the test dataset
    # evaluate(model_ft, data_loader_test, device=device)
torch.save(model_ft.state_dict(), "model.bin")
