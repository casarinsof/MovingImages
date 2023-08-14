import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from network import ResNet18
import os
from video_train import accuracy, AverageMeter
from utils.augmented_dataset import AugmentedDataset
from utils.tiny import TinyImageNetDataset
from torchvision.models import resnet152
# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters

batch_size = 64
learning_rate = 0.001
num_epochs = 200
WEIGHT_DECAY = 1E-4

print('+++++++++++++++++++++++++++++++++++ HYPER PARAMETERS +++++++++++++++++++++++++++++++++++++++++++\n'
      '+                                                                                              +\n '
     f'+                lr = {learning_rate}, wd = {WEIGHT_DECAY}, epochs = {num_epochs}              +\n '
      '+                                                                                              +\n '
      '++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')


import argparse as arg

parser = arg.ArgumentParser(description="Choose the dataset to use to generate videos")
parser.add_argument("--dataset", type=str, help="Name of the dataset")

args = parser.parse_args()
global train_dataset, test_dataset

dataset = args.dataset
if dataset == 'mnist':
    dir_path = f'data/raw-data/{dataset}'
    file_name = os.path.join(dir_path, 'mnist.tar.gz')
    # Load the MNIST dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = datasets.MNIST(root=file_name, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root=file_name, train=False, download=True, transform=transform)
    num_classes = 10
    rgb = False
    model = ResNet18(rgb=rgb, num_classes=num_classes).to(device)

elif dataset == 'fashion-mnist':
    dir_path = f'data/raw-data/{dataset}'
    # Load the MNIST dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = datasets.FashionMNIST(dir_path, train=True, download=True, transform=transform)
    test_dataset = datasets.FashionMNIST(dir_path, train=False, download=True, transform=transform)
    num_classes = 10
    rgb = False
    model = ResNet18(rgb=rgb, num_classes=num_classes).to(device)

elif dataset == 'emnist':
    dir_path = f'data/raw-data/{dataset}'
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_dataset = datasets.EMNIST(dir_path, split='letters', train=True, download=True, transform=transform)
    test_dataset = datasets.EMNIST(dir_path, split='letters', train=False, download=True, transform=transform)

    test_y = test_dataset.targets
    train_y = train_dataset.targets

    train_labels = [label - 1 for label in train_y]
    test_labels = [label - 1 for label in test_y]

    test_dataset.targets = test_labels
    train_dataset.targets = train_labels

    num_classes = 26
    rgb = False
    model = ResNet18(rgb=rgb, num_classes=num_classes).to(device)

elif dataset == 'cifar-10':
    transform_train = transforms.Compose([
      #  transforms.RandomCrop(32, padding=4),
      #  transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

#    dir_path = f'data/raw-data/{dataset}'
    data_folder = '/data/users/scasarin/MovingImages/data/moving_cifar-10'
    train_dataset = AugmentedDataset(root_dir=f'{data_folder}', txt_file=f'{data_folder}/train_videofolder.txt',
                                     transform=transform_train)
    test_dataset = AugmentedDataset(root_dir=f'{data_folder}', txt_file=f'{data_folder}/val_videofolder.txt',
                                     transform=transform_train)
   # train_dataset = datasets.CIFAR10(dir_path, train=True, download=True, transform=transform_train)
   # test_dataset = datasets.CIFAR10(dir_path, train=False, download=True, transform=transform_test)
    num_classes = 10
    rgb = True

elif dataset == 'cifar-100':
    dir_path = f'data/raw-data/{dataset}'
    target_transforms = transforms.Compose([
        transforms.Resize(232, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    train_dataset = datasets.CIFAR100(dir_path, train=True, download=True, transform=target_transforms)
    test_dataset = datasets.CIFAR100(dir_path, train=False, download=True, transform=target_transforms)
    num_classes = 100
    model = resnet152(weights='DEFAULT')
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

elif dataset == 'tiny':
    dir_path = f'data/raw-data/{dataset}'
    target_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(232, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

    train_dataset = TinyImageNetDataset(root_dir=dir_path, mode='train', preload=False,
                                        load_transform=None,
                                        transform=target_transforms, download=True, max_samples=None)
    test_dataset = TinyImageNetDataset(root_dir=dir_path, mode='val', preload=False,
                                       load_transform=None,
                                       transform=target_transforms, download=True, max_samples=None)
    # in realta' e' validation
    num_classes = 200
    model = resnet152(weights='DEFAULT')
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
model = model.cuda()
# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=WEIGHT_DECAY)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
# Training loop
total_step = len(train_loader)
model.train()
for epoch in range(num_epochs):
    correct = 0
    total = 0
    top1 = AverageMeter()
    top5 = AverageMeter()
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        prec1, prec5 = accuracy(outputs.data, labels, topk=(1, 5))

        top1.update(prec1, images.size(0))
        top5.update(prec5, images.size(0))

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print training progress
        if (i + 1) % 100 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{total_step}], Loss: {loss.item():.4f}")

            output = ('Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, i, len(train_loader), top1=top1, top5=top5,
                lr=optimizer.param_groups[-1]['lr']))

            print(output)
    lr_scheduler.step()
    if epoch % 2 == 0:

        # Test the model
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            test_accuracy = 100 * correct / total
            print(f"Accuracy on the test set: {test_accuracy:.2f}%")


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res
