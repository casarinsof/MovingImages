import os

def return_mnist(aux_dir):
    root_data = f'./{aux_dir}moving_mnist/'
    filename_imglist_train = f'./{aux_dir}moving_mnist/train_videofolder.txt'
    filename_imglist_val = f'./{aux_dir}moving_mnist/val_videofolder.txt'
    prefix = '{:05d}.png'

    return filename_imglist_train, filename_imglist_val, root_data, prefix

def return_emnist(aux_dir):
    root_data = f'./{aux_dir}moving_emnist/'
    filename_imglist_train = f'./moving_emnist/train_videofolder.txt'
    filename_imglist_val = f'./moving_emnist/val_videofolder.txt'
    prefix = '{:05d}.png'

    return filename_imglist_train, filename_imglist_val, root_data, prefix



def return_fashion_mnist(aux_dir):
    root_data = f'./{aux_dir}moving_fashion-mnist/'
    filename_imglist_train = f'./{aux_dir}moving_fashion-mnist/train_videofolder.txt'
    filename_imglist_val = f'./{aux_dir}moving_fashion-mnist/val_videofolder.txt'
    prefix = '{:05d}.png'

    return filename_imglist_train, filename_imglist_val, root_data, prefix


def return_cifar10(aux_dir):
    root_data = f'./{aux_dir}moving_cifar-10/'
    filename_imglist_train = f'./{aux_dir}moving_cifar-10/train_videofolder.txt'
    filename_imglist_val = f'./{aux_dir}moving_cifar-10/val_videofolder.txt'
    prefix = '{:05d}.png'

    return filename_imglist_train, filename_imglist_val, root_data, prefix

def return_cifar100(aux_dir):
    root_data = f'./{aux_dir}moving_cifar-100/'
    filename_imglist_train = f'{aux_dir}moving_cifar-100/train_videofolder.txt'
    filename_imglist_val = f'{aux_dir}moving_cifar-100/val_videofolder.txt'
    prefix = '{:05d}.png'

    return filename_imglist_train, filename_imglist_val, root_data, prefix

def return_tiny(aux_dir):
    root_data = f'./{aux_dir}moving_tiny/'
    filename_imglist_train = f'{aux_dir}moving_tiny/train_videofolder.txt'
    filename_imglist_val = f'{aux_dir}moving_tiny/val_videofolder.txt'
    prefix = '{:05d}.png'

    return filename_imglist_train, filename_imglist_val, root_data, prefix



def return_dataset(dataset, aux_dir, ROOT_DATASET):
    dict_single = {'mnist': return_mnist(aux_dir), 'fashion-mnist' : return_fashion_mnist(aux_dir),
                   'cifar-10': return_cifar10(aux_dir),
                   'emnist': return_emnist(aux_dir), 'cifar-100': return_cifar100(aux_dir), 'tiny': return_tiny(aux_dir)}
    if dataset in dict_single:
            file_imglist_train, file_imglist_val, root_data, prefix = dict_single[dataset]
    else:
        raise ValueError('Unknown dataset '+dataset)
    file_imglist_train = os.path.join(ROOT_DATASET, file_imglist_train)
    file_imglist_val = os.path.join(ROOT_DATASET, file_imglist_val)
    root_data = os.path.join(ROOT_DATASET, root_data)

    return file_imglist_train, file_imglist_val, root_data, prefix


