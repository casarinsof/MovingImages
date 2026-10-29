import torch
import torch.nn as nn
import torch.optim as optim
import optuna
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.models import resnet152
import os
from optuna.trial import TrialState
from scripts import video_train
from scripts.video_train import AverageMeter
from movingimages.utils.augmented_dataset import AugmentedDataset
from movingimages.utils.tiny import TinyImageNetDataset
# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# Hyperparameters
import argparse as arg

parser = arg.ArgumentParser(description="Choose the dataset to use to generate videos")
parser.add_argument("--dataset", type=str, help="Name of the dataset")
def objective(trial):
    # Define the hyperparameter search space
    lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    wd = trial.suggest_float('wd', 1e-6, 1e-3, log=True)
    num_epochs = trial.suggest_int('epochs', 120, 200)
    batch_size = 64



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

        dir_path = f'data/raw-data/{dataset}'
        train_dataset = datasets.CIFAR10(dir_path, train=True, download=True, transform=transform_train)
        test_dataset = datasets.CIFAR10(dir_path, train=False, download=True, transform=transform_test)
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

    model = resnet152(weights='DEFAULT')
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    model = model.to(device)
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    # Training loop
    total_step = len(train_loader)
    model.train()
    for epoch in range(num_epochs):
        top1 = AverageMeter()
        top5 = AverageMeter()
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            prec1, prec5 = video_train.accuracy(outputs.data, labels, topk=(1, 5))
            top1.update(prec1, images.size(0))
            top5.update(prec5, images.size(0))

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Print training progress
            if (i + 1) % 100 == 0:
                print(f"Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{total_step}], Loss: {loss.item():.4f}")

                # output = ('Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t'
                #           'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                #           'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                #         epoch, i, len(train_loader), top1=top1, top5=top5,
                #         lr=optimizer.param_groups[-1]['lr']))
                #
                # print(output)

        model.eval()
        vtop1 = AverageMeter()
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                val_loss = criterion(outputs, labels)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                vprec1, _ = video_train.accuracy(outputs.data, labels, topk=(1, 5))
                vtop1.update(vprec1, images.size(0))

            accuracy = 100 * correct / total
            # print(f"Accuracy on last sampls on the test set: {accuracy:.2f}%")
            # output = ('Epoch: [{0}][{1}/{2}], '
            #           'Prec@1 {top1.avg:.3f}\t'
            #           .format(epoch, i, len(train_loader), top1=vtop1))
            # print(output)

            quella_che_uso = vtop1.avg
            trial.report(quella_che_uso, epoch)

            # Handle pruning based on the intermediate value.
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

    # Test the model
        lr_scheduler.step(val_loss)

    return vtop1.avg

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=20)

pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])



# Get the best hyperparameters and their values
best_params = study.best_params
best_lr = best_params['lr']
best_wd = best_params['wd']
best_epochs = best_params['epochs']

print("Best LR: ", best_lr)
print("Best WD: ", best_wd)
print("Best Epochs: ", best_epochs)

print("Study statistics: ")
print("  Number of finished trials: ", len(study.trials))
print("  Number of pruned trials: ", len(pruned_trials))
print("  Number of complete trials: ", len(complete_trials))

print("Best trial:")
trial = study.best_trial

print("  Value: ", trial.value)

print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))