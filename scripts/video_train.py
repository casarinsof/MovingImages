import time
import shutil
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from torch.nn.utils import clip_grad_norm_
from movingimages.models.video_network import ResNet18, VideoResnet152
import torchvision.transforms as transforms
from movingimages.utils.transforms import ToTorchFormatTensor, GroupNormalize, Stack, GroupMultiScaleCrop, GroupScale, GroupCenterCrop
from movingimages.utils.dataset import VideoDataset
from datetime import datetime
import os
from torch.utils.data import DataLoader
from torch.cuda import amp
from movingimages.utils.helpers import return_dataset
import torch.nn as nn

torch.backends.cudnn.benchmark = True
torch.autograd.set_detect_anomaly(False)
torch.autograd.profiler.profile(False)
torch.autograd.profiler.emit_nvtx(False)

best_prec1 = 0
MOMENTUM = 0.9
NUM_SEGMENTS = 4
BATCH_SIZE = 2
CLIP_GRADIENT = 20
ITER_SIZE = 2
PRINT_FREQ = 200

import argparse as arg

parser = arg.ArgumentParser(description="Choose the dataset to use to generate videos")
parser.add_argument("--dataset", type=str, help="Name of the dataset")
parser.add_argument("--lr", type=float, help="Learning rate")
parser.add_argument("--wd", type=float, help="Weight decay")
parser.add_argument("--epochs", type=int, help="Number of epochs")
parser.add_argument("--root", type=str, help="Root path to the dataset")
parser.add_argument("--resume", type=str, help="Path tot he saved weights")
parser.add_argument("--complex", action="store_true", help= "Parameter for complex motions")

args = parser.parse_args()
DATASET = args.dataset
LR = args.lr
WEIGHT_DECAY = args.wd
EPOCHS = args.epochs
ROOT_DATASET = args.root
complex = args.complex
def main():

    print('+++++++++++++++++++++++++++++++++++ HYPER PARAMETERS +++++++++++++++++++++++++++++++++++++++++++\n'
          '+                                                                                              +\n'
          f'+                lr = {LR}, wd = {WEIGHT_DECAY}, epochs = {EPOCHS}                                          +\n'
          '+                                                                                              +\n'
          '++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    global best_prec1, store_name
    store_name = '_'.join(
        [DATASET, 'resnet18', 'segment%d' % NUM_SEGMENTS])
    train_transform = transforms.Compose([
        Stack(roll=False),
        ToTorchFormatTensor(),
        GroupNormalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # it requires this format
    ])
    test_transform = train_transform

    if DATASET == 'mnist':
        num_class = 10
        rgb_prefix = ''
        rgb_read_format = "frames_{:06d}.png"
        model = ResNet18(num_classes=num_class, num_segments=NUM_SEGMENTS)

    elif DATASET == 'fashion-mnist':
        num_class = 10
        rgb_prefix = ''
        rgb_read_format = "frames_{:06d}.png"
        model = ResNet18(num_classes=num_class, num_segments=NUM_SEGMENTS)

    elif DATASET == 'cifar-10':
        num_class = 10 # = 4 is self_supervised
        rgb_prefix = ''
        rgb_read_format = "frames_{:06d}.png"
        model = ResNet18(num_classes=num_class, num_segments=NUM_SEGMENTS)

    elif DATASET == 'emnist':
        num_class = 26
        rgb_prefix = ''
        rgb_read_format = "frames_{:06d}.png"
        model = ResNet18(num_classes=num_class, num_segments=NUM_SEGMENTS)

    elif DATASET == 'cifar-100':
        num_class = 100
        rgb_prefix = ''
        rgb_read_format = "frames_{:06d}.png"
        train_transform = transforms.Compose([
            GroupMultiScaleCrop(232,  [1, .875, .75, .66]),
            Stack(roll=False),
            ToTorchFormatTensor(),
            GroupNormalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        test_transform = transforms.Compose([
            GroupScale(232),
            GroupCenterCrop(224),
            Stack(False),
            ToTorchFormatTensor(),
            GroupNormalize(0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        ])
        model = VideoResnet152(pretrained=True, num_classes=num_class, num_segments=NUM_SEGMENTS)
    elif DATASET == 'tiny':
        num_class = 200
        rgb_prefix = ''
        rgb_read_format = "frames_{:06d}.png"
        train_transform = transforms.Compose([
            GroupMultiScaleCrop(232,  [1, .875, .75, .66]),
            Stack(roll=False),
            ToTorchFormatTensor(),
            GroupNormalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        test_transform = transforms.Compose([
            GroupScale(232),
            GroupCenterCrop(224),
            Stack(False),
            ToTorchFormatTensor(),
            GroupNormalize(0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        ])
        model = VideoResnet152(pretrained=True, num_classes=num_class, num_segments=NUM_SEGMENTS)

    else:
        raise ValueError('Unknown dataset ' + DATASET)

    global model_dir

    model_dir = os.path.join('experiments', DATASET, f'{model._get_name()}', datetime.now().strftime('%b%d_%H-%M-%S'))
    os.makedirs(model_dir)
    # os.makedirs(os.path.join(model_dir, args.root_log))
    if complex:
        aux_dir = 'complex_'
    else:
        aux_dir = ''
    train_list, val_list, root_path, prefix = return_dataset(args.dataset, aux_dir, ROOT_DATASET)


    # optimizer = torch.optim.SGD(model.parameters(),
    #         LR,
    #        momentum=MOMENTUM,
    #      weight_decay=WEIGHT_DECAY)
    scaler = amp.GradScaler(enabled=True)

    model = model.cuda()
    model = torch.nn.DataParallel(model).cuda()

    if args.resume != '':
        if os.path.isfile(args.resume):
            print(("=> loading checkpoint '{}'".format(args.resume)))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['model_state_dict'])
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
            print("=> loaded checkpoint")
            in_features = model.module.fc.in_features
            model.module.fc = nn.Linear(in_features, 10).cuda()
        else:
            print(("=> no checkpoint found at '{}'".format(args.resume)))


    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    train_loader = DataLoader(VideoDataset(root_path, train_list, num_segments=NUM_SEGMENTS,
                                           image_tmpl=rgb_prefix + rgb_read_format,
                                           transform=train_transform),
                              batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True)

    val_loader = DataLoader(VideoDataset(root_path, val_list, num_segments=NUM_SEGMENTS,
                                         image_tmpl=rgb_prefix + rgb_read_format,
                                         random_shift=False,
                                         transform=train_transform),
                            batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=4, pin_memory=True, drop_last=False)

    # define loss function (criterion) and optimizer
    criterion = torch.nn.CrossEntropyLoss().cuda()

    for epoch in range(0, EPOCHS):

        train_prec1 = train(train_loader, model, criterion, optimizer, epoch, scaler=scaler)
        lr_scheduler.step()

        # evaluate on validation set
        if (epoch + 1) % 1 == 0:
            prec1 = validate(val_loader, model, criterion, epoch=epoch)
            # remember best prec@1 and save checkpoint
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': 'resnet18',
                'model_state_dict': model.state_dict(),
                'scaler_state_dict': scaler.state_dict() if scaler is not None else None,
                'best_prec1': best_prec1,
                'current_prec1': prec1,
                'lr': optimizer.param_groups[-1]['lr'],
            }, is_best, model_dir)
        else:
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': 'resnet18',
                'model_state_dict': model.state_dict(),
                'scaler_state_dict': scaler.state_dict() if scaler is not None else None,
                'best_prec1': best_prec1,
                'current_prec1': train_prec1,
                'lr': optimizer.param_groups[-1]['lr'],
            }, False, model_dir)


def train(train_loader, model, criterion, optimizer, epoch, scaler):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    # switch to train mode
    model.train()
    loss_summ = 0
    for i, (input, target) in enumerate(train_loader):

        target = target.cuda()
        input_var = input.cuda()

        # compute output
        #################amp########################################
        with amp.autocast(enabled=True):
            output = model(input_var)
            loss = criterion(output, target) / ITER_SIZE
            loss_summ += loss.data
            scaler.scale(loss).backward()

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss_summ.data, input.size(0))
        top1.update(prec1, input.size(0))
        top5.update(prec5, input.size(0))

        if (i + 1) % ITER_SIZE == 0:
            # scale down gradients when iter size is functioning
            if CLIP_GRADIENT is not None:
                scaler.unscale_(optimizer)
                total_norm = clip_grad_norm_(model.parameters(), CLIP_GRADIENT)
                if total_norm > CLIP_GRADIENT:
                    print("clipping gradient: {} with coef {}".format(total_norm, CLIP_GRADIENT / total_norm))
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            loss_summ = 0

        if i % PRINT_FREQ == 0:
            output = ('Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'.format(
                epoch, i, len(train_loader), loss=losses, top1=top1, lr=optimizer.param_groups[-1]['lr']))

            print(output)

    return top1.avg


def validate(val_loader, model, criterion, epoch):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):

            target = target.cuda()
            input_var = input.cuda()

            # compute output
            with amp.autocast(enabled=True):
                output = model(input_var)
                loss = criterion(output, target)

            # measure accuracy and record loss
            losses.update(loss.data, input.size(0))
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            top1.update(prec1, input.size(0))
            top5.update(prec5, input.size(0))

            if i % PRINT_FREQ == 0:
                output = ('Test: [{0}/{1}]\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                          'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                          'Prec@5 {top5.val:.3f} (\t avg {top5.avg:.3f})'.format(
                    i, len(val_loader), loss=losses, top1=top1, top5=top5))
                print(output)

    output = ('Testing Results: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Loss {loss.avg:.5f}'
              .format(top1=top1, top5=top5, loss=losses))
    print(output)

    return top1.avg


def save_checkpoint(state, is_best, model_dir):
    torch.save(state, '%s/%s_checkpoint.pth.tar' % (model_dir, store_name))
    if is_best:
        shutil.copyfile('%s/%s_checkpoint.pth.tar' % (model_dir, store_name),
                        '%s/%s_best.pth.tar' % (model_dir, store_name))


def save_checkpoint_epoch(state, epoch, model_dir):
    torch.save(state, '%s/%s_checkpoint_%d.pth.tar' % (model_dir, store_name, epoch))


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


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


if __name__ == '__main__':
    main()
