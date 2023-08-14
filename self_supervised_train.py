import time
import shutil
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from torch.nn.utils import clip_grad_norm_
from video_network import ResNet18
import torchvision.transforms as transforms
from utils.transforms import ToTorchFormatTensor, GroupNormalize, Stack
from utils.dataset import VideoDataset
from datetime import datetime
import os
from torch.utils.data import DataLoader
from torch.cuda import amp

def return_cifar10():
    root_data = './moving_cifar-10/'
    filename_imglist_train = f'./moving_cifar-10/train_videofolder.txt'
    filename_imglist_val = f'./moving_cifar-10/val_videofolder.txt'
    prefix = '{:05d}.png'

    return filename_imglist_train, filename_imglist_val, root_data, prefix


ROOT_DATASET = './self_supervised_data'
def return_mnist():
    root_data = './moving_mnist/'
    filename_imglist_train = './moving_mnist/train_videofolder.txt'
    filename_imglist_val = './moving_mnist/val_videofolder.txt'
    prefix = '{:05d}.png'

    return filename_imglist_train, filename_imglist_val, root_data, prefix

def return_emnist():
    root_data = './moving_emnist/'
    filename_imglist_train = './moving_emnist/train_videofolder.txt'
    filename_imglist_val = './moving_emnist/val_videofolder.txt'
    prefix = '{:05d}.png'

    return filename_imglist_train, filename_imglist_val, root_data, prefix



def return_fashion_mnist():
    root_data = './moving_fashion-mnist/'
    filename_imglist_train = f'./moving_fashion-mnist/train_videofolder.txt'
    filename_imglist_val = f'./moving_fashion-mnist/val_videofolder.txt'
    prefix = '{:05d}.png'

    return filename_imglist_train, filename_imglist_val, root_data, prefix


def return_dataset(dataset):
    dict_single = {'mnist': return_mnist(), 'fashion-mnist' : return_fashion_mnist(), 'cifar-10': return_cifar10(),
                   'emnist': return_emnist()}
    if dataset in dict_single:
            file_imglist_train, file_imglist_val, root_data, prefix = dict_single[dataset]
    else:
        raise ValueError('Unknown dataset '+dataset)
    file_imglist_train = os.path.join(ROOT_DATASET, file_imglist_train)
    file_imglist_val = os.path.join(ROOT_DATASET, file_imglist_val)
    root_data = os.path.join(ROOT_DATASET, root_data)

    return file_imglist_train, file_imglist_val, root_data, prefix

torch.backends.cudnn.benchmark = True
torch.autograd.set_detect_anomaly(False)
torch.autograd.profiler.profile(False)
torch.autograd.profiler.emit_nvtx(False)

best_prec1 = 0

EPOCHS = 120
LR = 0.001
MOMENTUM = 0.9
WEIGHT_DECAY = 1E-4
NUM_SEGMENTS = 8
BATCH_SIZE = 32
CLIP_GRADIENT = 20
ITER_SIZE = 2
PRINT_FREQ = 200

import argparse as arg

parser = arg.ArgumentParser(description="Choose the dataset to use to generate videos")
parser.add_argument("--dataset", type=str, help="Name of the dataset")

args = parser.parse_args()
DATASET = args.dataset

def main():
    global best_prec1, store_name
    store_name = '_'.join(
        [DATASET, 'resnet18', 'segment%d' % NUM_SEGMENTS])

    if DATASET == 'mnist':
        num_class = 10
        rgb_prefix = ''
        rgb_read_format = "frames_{:06d}.png"


    elif DATASET == 'fashion-mnist':
        num_class = 10
        rgb_prefix = ''
        rgb_read_format = "frames_{:06d}.png"

    elif DATASET == 'cifar-10':
        num_class = 4
        rgb_prefix = ''
        rgb_read_format = "frames_{:06d}.png"

    elif DATASET == 'emnist':
        num_class = 26
        rgb_prefix = ''
        rgb_read_format = "frames_{:06d}.png"

    else:
        raise ValueError('Unknown dataset ' + DATASET)

    global model_dir

    model_dir = os.path.join('experiments', DATASET, 'resnet18SSL', datetime.now().strftime('%b%d_%H-%M-%S'))
    os.makedirs(model_dir)
    #os.makedirs(os.path.join(model_dir, args.root_log))

    train_list, val_list, root_path, prefix = return_dataset(args.dataset)

    model = ResNet18(num_classes=num_class, num_segments=NUM_SEGMENTS)

    train_transform = transforms.Compose([
        Stack(roll=False),
        ToTorchFormatTensor(),
        GroupNormalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) #it requires this format
    ])

   # optimizer = torch.optim.SGD(model.parameters(),
                       #         LR,
                        #        momentum=MOMENTUM,
                          #      weight_decay=WEIGHT_DECAY)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler = amp.GradScaler(enabled=True)

    model = model.cuda()
    model = torch.nn.DataParallel(model).cuda()

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
        if (epoch + 1) % 4 == 0:
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
        prec1, prec5 = accuracy(output.data, target, topk=(1, 3))
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
            prec1, prec5 = accuracy(output.data, target, topk=(1, 3))
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
