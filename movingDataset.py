import os
import numpy as np
import random
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from utils.tiny import TinyImageNetDataset

np.random.seed(np.random.randint(1 << 30))
num_frames = 12
seq_length = 12
batch_size = 1
num_digits = 1
step_length = 0.1


def create_reverse_dictionary(dictionary):
    dictionary_reverse = {}
    for word in dictionary:
        index = dictionary[word]
        dictionary_reverse[index] = word
    return dictionary_reverse


def create_dataset(classes):
    numbers = np.random.permutation(classes)
    dataset = np.zeros((2, classes), dtype=int)
    dataset[0, :] = numbers
    dataset[1, :] = 10 + numbers
    train = []
    val = []
    count = 0
    for i in range(classes):
        dummy = count % 2
        val.append(dataset[dummy, i])
        train.append(dataset[1 - dummy, i])
        count = count + 1
    return np.array(train), np.array(val)  # ,np.array(test)


def sent2matrix(sentence, dictionary):
    words = sentence.split()
    m = np.int32(np.zeros((1, len(words))))

    for i in range(len(words)):
        m[0, i] = dictionary[words[i]]
    return m


def matrix2sent(matrix, reverse_dictionary):
    text = ""
    for i in range(matrix.shape[0]):
        text = text + " " + reverse_dictionary[matrix[i]]
    return text


def zoom_in_out(image, zoom_factor, image_size2):
    height, width = image.shape[:2]

    # Calculate new dimensions based on the zoom factor
    new_height = int(height * zoom_factor)
    new_width = int(width * zoom_factor)

    # Resize the image using the new dimensions
    resized_image = cv2.resize(image, (new_width, new_height))

    # Calculate the padding required to restore the original shape
    pad_top = (image_size2 - new_height) // 2
    pad_bottom = image_size2 - new_height - pad_top  # height - new_height - pad_top
    pad_left = (image_size2 - new_width) // 2  # (width - new_width) // 2
    pad_right = image_size2 - new_width - pad_top

    # Pad the resized image to match the original shape
    zoomed_image = cv2.copyMakeBorder(resized_image, pad_top, pad_bottom, pad_left, pad_right,
                                      cv2.BORDER_CONSTANT, value=0)

    return zoomed_image


def rotate_image(image, angle, image_size2):
    # transform to PIL Image
    height, width = image.shape[:2]
    center = (width / 2, height / 2)

    # Define the rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Perform the rotation
    rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))

    # Calculate the padding required to restore the original shape
    pad_top = (image_size2 - height) // 2  # (height - new_height) // 2
    pad_bottom = image_size2 - height - pad_top  # height - new_height - pad_top
    pad_left = (image_size2 - width) // 2  # (width - new_width) // 2
    pad_right = image_size2 - width - pad_top  # width - new_width - pad_left

    # Pad the resized image to match the original shape
    rotated_image = cv2.copyMakeBorder(rotated_image, pad_top, pad_bottom, pad_left, pad_right,
                                       cv2.BORDER_CONSTANT, value=0)

    # Return the rotated image
    return rotated_image


def Overlap(a, b):
    return np.maximum(a, b)


def GetRandomTrajectory(batch_size, motion, digit_size, image_size1):
    length = seq_length
    canvas_size = image_size1 - digit_size

    y = np.random.rand(batch_size)  # the starting point of the two numbers
    x = np.random.rand(batch_size)

    start_y = np.zeros((length, batch_size))
    start_x = np.zeros((length, batch_size))

    if int(motion) == 0:
        theta = np.ones(batch_size) * 0.1 * np.pi
    else:
        theta = np.ones(batch_size) * 0.1 * np.pi

    direction = random.choice([0, 1, 2, 3])  # 1 is moving up and down, 0 is moving left and right
    zoom_factors = []
    rotations_angles = []
    initial_angle = 0
    max_rotation_angle = random.uniform(15, 45)
    step_size = max_rotation_angle / length
    # v_y = 2 * np.sin(theta)
    # v_x = 2 * np.cos(theta)
    v_y = 2 * np.sin(theta)
    v_x = 2 * np.sin(theta)

    zoom_factor = 1.0  # Initial zoom factor
    zoom_step = 0.005  # Step size for zooming #todo forse meglio zoom out ?? Cioe che si rimpicciolisca?

    for i in range(length):
        if direction == 1:
            y += v_y * step_length
            x += 0 * step_length
        elif direction == 0:
            y -= 0 * step_length
            x -= v_x * step_length
        elif direction == 2:
            if zoom_factor > 1.7 or zoom_factor < 0.8:
                zoom_step = -zoom_step
            zoom_factor += zoom_step
            zoom_factors.append(zoom_factor)
        elif direction == 3:
            # Generate smooth rotations
            current_angle = initial_angle + i * step_size
            rotations_angles.append(current_angle)

        for j in range(batch_size):
            # check of boundaries conditions
            if x[j] <= 0:
                x[j] = 0
                v_x[j] = -v_x[j]
            if x[j] >= 1.0:
                x[j] = 1.0
                v_x[j] = -v_x[j]
            if y[j] <= 0:
                y[j] = 0
                v_y[j] = -v_y[j]
            if y[j] >= 1.0:
                y[j] = 1.0
                v_y[j] = -v_y[j]
        # print x, y
        start_y[i, :] = y
        start_x[i, :] = x

    # scale to the size of the canvas.
    start_y = (canvas_size * start_y).astype(np.int32)
    start_x = (canvas_size * start_x).astype(np.int32)

    return start_y, start_x, direction, zoom_factors, rotations_angles


# function to render the final gif
def create_gif(digit_imgs, motion, digit_size, image_size1):
    # get an array of random numbers for indices
    start_y, start_x, direction, zoom_factors, rotation_angles = GetRandomTrajectory(batch_size * num_digits, motion,
                                                                                     digit_size, image_size1)
    if digit_size == 28:
        num_channels = 1
    else:
        num_channels = 3
    gifs = np.zeros((seq_length, batch_size, image_size1, image_size1, num_channels), dtype=np.float32)
    for j in range(batch_size):
        for n in range(num_digits):
            digit_image = digit_imgs[n, :, :]  # it was with n for multiple digits in one frame digit_imgs[n, :, :]
            for i in range(num_frames):
                if direction == 0 or direction == 1:
                    top = start_y[i, j * num_digits + n]
                    left = start_x[i, j * num_digits + n]
                    bottom = top + digit_size
                    right = left + digit_size
                    if digit_size == 28:
                        gifs[i, j, top:bottom, left:right, 0] = Overlap(gifs[i, j, top:bottom, left:right, 0],
                                                                        digit_image)
                    else:
                        gifs[i, j, top:bottom, left:right] = Overlap(gifs[i, j, top:bottom, left:right], digit_image)
                elif direction == 2:
                    if digit_size == 28:
                        gifs[i, j, :, :, 0] = zoom_in_out(digit_image, zoom_factors[i], image_size1)
                    else:
                        gifs[i, j] = zoom_in_out(digit_image, zoom_factors[i], image_size1)
                elif direction == 3:
                    if digit_size == 28:
                        gifs[i, j, :, :, 0] = rotate_image(digit_image, rotation_angles[i], image_size1)
                    else:
                        gifs[i, j] = rotate_image(digit_image, rotation_angles[i], image_size1)
    return gifs, direction


def create_gifs(dataset, data, labels, num, digit_size, motion_strings, classes, image_size1):
    final_gif_data = []
    outer_index = 0
    digits = dataset % classes
    motions_values = dataset / classes
    while outer_index < num:
        idxs = np.random.randint(data.shape[0], size=num_digits)
        print(outer_index)
        if labels[idxs[0]] in digits:

            motion_list = np.where(digits == labels[idxs[0]])[0]
            # print motion_list,digits,motions_values,labels[idxs[0]]
            random.shuffle(motion_list)
            motion_idx = motion_list[0]
            motion = motions_values[motion_idx]

            digit = data[idxs]
            dummy, direction = create_gif(digit, motion, digit_size, image_size1)

            sentence = 'the digit %d is moving %s .' % (
                labels[idxs[0]], motion_strings[int(direction)])
            instance = {'video': dummy, 'caption': sentence}
            final_gif_data.append(instance)

            # show_gif(dummy, sentence)
        else:
            outer_index -= 1
        outer_index += 1
    return final_gif_data


def GetNoSmoothRandomTrajectory(batch_size, motion, digit_size, image_size1):
    length = seq_length
    canvas_size = image_size1 - digit_size

    y1, y0 = np.random.rand(batch_size), np.random.rand(batch_size)  # the starting point of the two numbers
    x1, x0 = np.random.rand(batch_size), np.random.rand(batch_size)

    start_y0, start_y1 = np.zeros((length, batch_size)), np.zeros((length, batch_size))
    start_x1, start_x0 = np.zeros((length, batch_size)), np.zeros((length, batch_size))

    if int(motion) == 0:
        theta = np.ones(batch_size) * 0.8 * np.pi
    else:
        theta = np.ones(batch_size) * 0.8 * np.pi

    zoom_factors = []
    rotations_angles = []
    initial_angle = 0
    max_rotation_angle = random.uniform(40, 45)
    step_size = max_rotation_angle / length

    v_y = 2 * np.sin(theta)
    v_x = 2 * np.sin(theta)

    zoom_factor = 1.0
    zoom_step = 0.8

    for i in range(length):
        y1 += v_y * step_length
        x1 += 0 * step_length
        y0 -= 0 * step_length
        x0 -= v_x * step_length

        if zoom_factor > 1.7 or zoom_factor < 0.8:
            zoom_step = -zoom_step
        zoom_factor += zoom_step
        zoom_factors.append(zoom_factor)

        current_angle = initial_angle + i * step_size
        rotations_angles.append(current_angle)

        for j in range(batch_size):
            if x1[j] <= 0:
                x1[j] = 0
                v_x[j] = -v_x[j]
            if x1[j] >= 1.0:
                x1[j] = 1.0
                v_x[j] = -v_x[j]
            if y1[j] <= 0:
                y1[j] = 0
                v_y[j] = -v_y[j]
            if y1[j] >= 1.0:
                y1[j] = 1.0
                v_y[j] = -v_y[j]

            if x0[j] <= 0:
                x0[j] = 0
                v_x[j] = -v_x[j]
            if x0[j] >= 1.0:
                x0[j] = 1.0
                v_x[j] = -v_x[j]
            if y0[j] <= 0:
                y0[j] = 0
                v_y[j] = -v_y[j]
            if y0[j] >= 1.0:
                y0[j] = 1.0
                v_y[j] = -v_y[j]
        # print x, y
        start_y0[i, :] = y0
        start_x0[i, :] = x0

        start_y1[i, :] = y1
        start_x1[i, :] = x1

    start_y1 = (canvas_size * start_y1).astype(np.int32)
    start_x1 = (canvas_size * start_x1).astype(np.int32)
    start_y0 = (canvas_size * start_y0).astype(np.int32)
    start_x0 = (canvas_size * start_x0).astype(np.int32)

    return start_y0, start_x0, start_y1, start_x1, zoom_factors, rotations_angles


def create_NoSmooth_gif(digit_imgs, motion, digit_size, image_size1):
    # get an array of random numbers for indices
    start_y0, start_x0, start_y1, start_x1, zoom_factors, rotation_angles = GetNoSmoothRandomTrajectory(
        batch_size * num_digits, motion,
        digit_size, image_size1)
    if digit_size == 28:
        num_channels = 1
    else:
        num_channels = 3
    gifs = np.zeros((seq_length, batch_size, image_size1, image_size1, num_channels), dtype=np.float32)
    for j in range(batch_size):
        for n in range(num_digits):
            digit_image = digit_imgs[n, :, :]  # it was with n for multiple digits in one frame digit_imgs[n, :, :]
            for i in range(num_frames):
                direction = random.choice([0, 1, 2, 3])  # 1 is moving up and down, 0 is moving left and right
                if direction == 0:
                    top = start_y0[i, j * num_digits + n]
                    left = start_x0[i, j * num_digits + n]
                    bottom = top + digit_size
                    right = left + digit_size
                    if digit_size == 28:
                        gifs[i, j, top:bottom, left:right, 0] = Overlap(gifs[i, j, top:bottom, left:right, 0],
                                                                        digit_image)
                    else:
                        gifs[i, j, top:bottom, left:right] = Overlap(gifs[i, j, top:bottom, left:right], digit_image)

                elif direction == 1:
                    top = start_y1[i, j * num_digits + n]
                    left = start_x1[i, j * num_digits + n]
                    bottom = top + digit_size
                    right = left + digit_size
                    if digit_size == 28:
                        gifs[i, j, top:bottom, left:right, 0] = Overlap(gifs[i, j, top:bottom, left:right, 0],
                                                                        digit_image)
                    else:
                        gifs[i, j, top:bottom, left:right] = Overlap(gifs[i, j, top:bottom, left:right], digit_image)
                elif direction == 2:
                    if digit_size == 28:
                        gifs[i, j, :, :, 0] = zoom_in_out(digit_image, zoom_factors[i], image_size1)
                    else:
                        gifs[i, j] = zoom_in_out(digit_image, zoom_factors[i], image_size1)
                elif direction == 3:
                    if digit_size == 28:
                        gifs[i, j, :, :, 0] = rotate_image(digit_image, rotation_angles[i], image_size1)
                    else:
                        gifs[i, j] = rotate_image(digit_image, rotation_angles[i], image_size1)
    return gifs, direction


def create_NoSmooth_gifs(dataset, data, labels, num, digit_size, motion_strings, classes, image_size1):
    final_gif_data = []
    outer_index = 0
    digits = dataset % classes
    motions_values = dataset / classes

    while outer_index < num:
        idxs = np.random.randint(data.shape[0], size=num_digits)
        print(outer_index)
        if labels[idxs[0]] in digits:

            motion_list = np.where(digits == labels[idxs[0]])[0]
            # print motion_list,digits,motions_values,labels[idxs[0]]
            random.shuffle(motion_list)
            motion_idx = motion_list[0]
            motion = motions_values[motion_idx]

            digit = data[idxs]
            dummy, direction = create_NoSmooth_gif(digit, motion, digit_size, image_size1)

            sentence = 'the digit %d is moving %s .' % (
                labels[idxs[0]], motion_strings[int(direction)])
            instance = {'video': dummy, 'caption': sentence}
            final_gif_data.append(instance)

            # show_gif(dummy, sentence)
        else:
            outer_index -= 1
        outer_index += 1
    return final_gif_data


def GetComplexRandomTrajectory(batch_size, motion, digit_size, image_size1):
    length = seq_length
    canvas_size = image_size1 - digit_size

    y = np.random.rand(batch_size)  # the starting point of the two numbers
    x = np.random.rand(batch_size)

    start_y = np.zeros((length, batch_size))
    start_x = np.zeros((length, batch_size))

    if int(motion) == 0:
        theta = np.ones(batch_size) * 0.02 * np.pi
    else:
        theta = np.ones(batch_size) * 0.01 * np.pi

    zoom_factors = []
    rotations_angles = []
    initial_angle = 0
    max_rotation_angle = random.uniform(15, 50)
    step_size = max_rotation_angle / length
    v_y = 2 * np.sin(theta)
    v_x = 2 * np.cos(theta)

    zoom_factor = 1.0  # Initial zoom factor
    zoom_step = 0.1  # Step size for zooming #todo forse meglio zoom out ?? Cioe che si rimpicciolisca?

    for i in range(length):

        y += v_y * step_length
        x += v_x * step_length

        zoom_factor += zoom_step
        if zoom_factor > 1.7 or zoom_factor < 0.7:
            zoom_step = -zoom_step
            zoom_factor += zoom_step
        zoom_factors.append(zoom_factor)

        # Generate smooth rotations
        current_angle = initial_angle + i * step_size
        rotations_angles.append(current_angle)

        for j in range(batch_size):
            # check of boundaries conditions
            if x[j] <= 0:
                x[j] = 0
                v_x[j] = -v_x[j]
            if x[j] >= 1.0:
                x[j] = 1.0
                v_x[j] = -v_x[j]
            if y[j] <= 0:
                y[j] = 0
                v_y[j] = -v_y[j]
            if y[j] >= 1.0:
                y[j] = 1.0
                v_y[j] = -v_y[j]
        # print x, y
        start_y[i, :] = y
        start_x[i, :] = x

    # scale to the size of the canvas.
    start_y = (canvas_size * start_y).astype(np.int32)
    start_x = (canvas_size * start_x).astype(np.int32)

    return start_y, start_x, zoom_factors, rotations_angles


# function to render the final gif
def create_complex_gif(digit_imgs, motion, digit_size, image_size1, image_size2):
    # get an array of random numbers for indices
    start_y, start_x, zoom_factors, rotation_angles = GetComplexRandomTrajectory(batch_size * num_digits, motion,
                                                                                 digit_size, image_size1)

    num_channels = 1 if digit_size == 28 else 3

    gifs = np.zeros((seq_length, batch_size, image_size1, image_size1, num_channels), dtype=np.float32)
    gifs2 = np.zeros((seq_length, batch_size, image_size2, image_size2, num_channels), dtype=np.float32)
    for j in range(batch_size):
        for n in range(num_digits):
            digit_image = digit_imgs[n, :, :]  # it was with n for multiple digits in one frame digit_imgs[n, :, :]
            for i in range(num_frames):

                top = start_y[i, j * num_digits + n]
                left = start_x[i, j * num_digits + n]
                bottom = top + digit_size
                right = left + digit_size

                if digit_image.shape[0] == 28:
                    gifs[i, j, top:bottom, left:right, 0] = Overlap(gifs[i, j, top:bottom, left:right, 0],
                                                                    digit_image)
                    gifs2[i, j, :, :, 0] = zoom_in_out(gifs[i, j, :, :, 0], zoom_factors[i], image_size2)
                    gifs2[i, j, :, :, 0] = rotate_image(gifs2[i, j, :, :, 0], rotation_angles[i], image_size2)
                else:
                    gifs[i, j, top:bottom, left:right] = Overlap(gifs[i, j, top:bottom, left:right], digit_image)
                    gifs2[i, j] = zoom_in_out(gifs[i, j], zoom_factors[i], image_size2)
                    gifs2[i, j] = rotate_image(gifs2[i, j], rotation_angles[i], image_size2)

    return gifs2


def create_complex_gifs(dataset, data, labels, num, digit_size, classes, image_size1, image_size2):
    final_gif_data = []
    outer_index = 0
    digits = dataset % classes
    motions_values = dataset / classes

    while outer_index < num:
        idxs = np.random.randint(data.shape[0], size=num_digits)
        print(outer_index)
        if labels[idxs[0]] in digits:

            motion_list = np.where(digits == labels[idxs[0]])[0]
            # print motion_list,digits,motions_values,labels[idxs[0]]
            random.shuffle(motion_list)
            motion_idx = motion_list[0]
            motion = motions_values[motion_idx]

            digit = data[idxs]
            dummy = create_complex_gif(digit, motion, digit_size, image_size1, image_size2)

            sentence = 'the digit %d is moving complex.' % (
                labels[idxs[0]])
            instance = {'video': dummy, 'caption': sentence}
            final_gif_data.append(instance)

            # show_gif(dummy, sentence)
        else:
            outer_index -= 1
        outer_index += 1
    return final_gif_data


def show_gif(digit_imgs, caption):
    for j in range(num_frames):
        plt.subplot(1, num_frames, j + 1)
        plt.imshow(digit_imgs[j, 0, :, :].astype(np.uint8), cmap='gray')
        plt.axis('off')
    plt.suptitle(caption)
    plt.show()


def create_moving(data, flag, dest='', classes=10):
    videos = [[] for i in range(classes)]
    output = []
    for i, dict in enumerate(data):
        caption = dict['caption']
        label = caption[10]
        if caption[11] != '':
            label = caption[10:12]
        video = dict['video']
        videos[int(label)].append(video)

    for label, sample in enumerate(videos):
        class_folder = os.path.join(dest, f"{flag}/frames/{flag}_video_class{label}_")
        sample = (videos[label])
        for index, image_moving in enumerate(sample):
            video_dir = class_folder + f'{index:05d}'
            os.makedirs(video_dir, exist_ok=True)
            for index_frame, frame in enumerate(image_moving):
                frame = np.array(frame)
                frame = (frame).astype(np.uint8)
                image_path = f"{video_dir}/frames_{index_frame + 1:06d}.png"
                Image.fromarray(frame.squeeze()).save(image_path)
            output.append(
                '%s %d %d' % ('{}/frames/{}_video_class{}_{:05d}'.format(flag, flag, label, index),
                              int(num_frames), label))
    with open(os.path.join(dest, '{}_videofolder.txt'.format(flag)), 'w') as f:
        f.write('\n'.join(output))


if __name__ == "__main__":

    from torchvision import datasets
    import argparse as arg

    parser = arg.ArgumentParser(description="Choose the dataset to use to generate videos")
    parser.add_argument("--dataset", type=str, help="Name of the dataset")
    parser.add_argument("--digit_size", type=int, help="Original dimension of the images")
    parser.add_argument("--combined_motion", action='store_true',
                        help="True if images have all transformation together")
    parser.add_argument("--smooth", action='store_true', help="Generate smooth or not smooth motions")

    args = parser.parse_args()
    global train_dataset, test_dataset

    dataset = args.dataset
    digit_size = args.digit_size
    combined_motion = args.combined_motion
    smooth = args.smooth

    if dataset == 'mnist':
        dir_path = 'data/raw-data/mnist'
        file_name = os.path.join(dir_path)
        train_dataset = datasets.MNIST(file_name, train=True, download=True, )
        test_dataset = datasets.MNIST(file_name, train=False, download=True, )
        classes = 10


    elif dataset == 'fashion-mnist':
        dir_path = f'data/raw-data/{dataset}'
        train_dataset = datasets.FashionMNIST(dir_path, train=True, download=True)
        test_dataset = datasets.FashionMNIST(dir_path, train=False, download=True)
        classes = 10

    elif dataset == 'cifar-10':
        dir_path = f'data/raw-data/{dataset}'
        train_dataset = datasets.CIFAR10(dir_path, train=True, download=True)
        test_dataset = datasets.CIFAR10(dir_path, train=False, download=True)
        classes = 10

    elif dataset == 'cifar-100':
        dir_path = f'data/raw-data/{dataset}'
        train_dataset = datasets.CIFAR100(dir_path, train=True, download=True)
        test_dataset = datasets.CIFAR100(dir_path, train=False, download=True)
        classes = 100

    elif dataset == 'emnist':
        dir_path = f'data/raw-data/{dataset}'
        train_dataset = datasets.EMNIST(dir_path, split='letters', train=True, download=True)
        test_dataset = datasets.EMNIST(dir_path, split='letters', train=False, download=True)
        classes = 26

    elif dataset == 'flowers-102':
        dir_path = f'data/raw-data/{dataset}'
        train_dataset = datasets.Flowers102(dir_path, split='train', download=True)
        val_dataset = datasets.Flowers102(dir_path, split='val', train=False, download=True)
        test_dataset = datasets.Flowers102(dir_path, split='test', train=False, download=True)
        classes = 102

    elif dataset == 'tiny':
        dir_path = f'data/raw-data/{dataset}'
        train_dataset = TinyImageNetDataset(root_dir=dir_path, mode='train', preload=True,
                                            load_transform=None,
                                            transform=None, download=True, max_samples=None)
        test_dataset = TinyImageNetDataset(root_dir=dir_path, mode='val', preload=True,
                                           load_transform=None,
                                           transform=None, download=True, max_samples=None)
        classes = 200

    if dataset == 'mnist' or dataset == 'fashion-mnist' or dataset == 'cifar-10':
        dictionary = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, 'the': 10,
                      'digit': 11, 'is': 12, 'moving': 13, 'vertically': 14, 'horizontally': 15, 'zooming': 16,
                      'rotating': 17}
    elif dataset == 'emnist':
        dictionary = {str(i): i for i in range(26)}
        dict2 = {'the': 26, 'digit': 27, 'is': 28, 'moving': 29, 'vertically': 30, 'horizontally': 31,
                 'zooming': 32, 'rotating': 33, }
        dictionary.update(dict2)

    elif dataset == 'cifar-100':
        dictionary = {str(i): i for i in range(100)}
        dict2 = {'the': 100, 'digit': 101, 'is': 102, 'moving': 103, 'vertically': 104, 'horizontally': 105,
                 'zooming': 106, 'rotating': 107}
        dictionary.update(dict2)

    elif dataset == 'flowers-102':
        dictionary = {str(i): i for i in range(103)}
        dict2 = {'the': 103, 'digit': 104, 'is': 105, 'moving': 106, 'vertically': 107, 'horizontally': 108,
                 'zooming': 109, 'rotating': 110}
        dictionary.update(dict2)

    elif dataset == 'tiny':
        dictionary = {str(i): i for i in range(200)}
        dict2 = {'the': 201, 'digit': 202, 'is': 203, 'moving': 204, 'vertically': 205, 'horizontally': 206,
                 'zooming': 207, 'rotating': 208}
        dictionary.update(dict2)

    # motion_strings = ['up then down', 'left then right', 'down then up', 'right then left', 'big then small',
    #                   'small then big', 'rotate to right', 'rotate to left']
    motion_strings = ['horizontally', 'vertically', 'zooming', 'rotating']

    # Get the training data and labels
    train_x = train_dataset.data
    train_y = train_dataset.targets

    # Get the test data and labels
    test_x = test_dataset.data
    test_y = test_dataset.targets

    train_data = train_x
    train_labels = train_y
    val_data = test_x
    val_labels = test_y

    data = np.concatenate((train_data, val_data), axis=0)
    labels = np.concatenate((train_labels, val_labels), axis=0)
    if np.min(labels) == 1:
        labels = [label - 1 for label in labels]

    if combined_motion is True:
        if dataset != 'tiny':
            image_size1 = 64
            image_size2 = 128
        else:
            image_size1 = 128
            image_size2 = 224
    else:
        if dataset != 'tiny':
            image_size1 = image_size2 = 64
        else:
            image_size1 = image_size2 = 128

    if combined_motion is True:
        name = 'complex_moving'
        dir = 'data'
    else:
        name = 'moving'
        if smooth:
            dir = 'data'
        else:
            dir = 'NoSmoothData'

    movDATA = f'./{dir}/{name}_{dataset}'
    if not os.path.exists(movDATA):
        os.makedirs(movDATA)

    train, val = create_dataset(classes)
    if combined_motion is True:
        print('\n\n Smoothness parameter in combined motion is always True\n\n')
        data_train = create_complex_gifs(train, data, labels, len(train_dataset), digit_size, classes, image_size1,
                                         image_size2)
        data_val = create_complex_gifs(val, data, labels, len(test_dataset), digit_size, classes, image_size1,
                                       image_size2)
    else:
        if smooth:
            data_train = create_gifs(train, data, labels, len(train_dataset), digit_size, motion_strings, classes,
                                     image_size1)
            data_val = create_gifs(val, data, labels, len(test_dataset), digit_size, motion_strings, classes,
                                   image_size1)
        else:
            data_train = create_NoSmooth_gifs(train, data, labels, len(train_dataset), digit_size, motion_strings,
                                              classes, image_size1)
            data_val = create_NoSmooth_gifs(val, data, labels, len(test_dataset), digit_size, motion_strings, classes,
                                            image_size1)

    create_moving(data_train, flag='train', dest=movDATA, classes=classes)
    create_moving(data_val, flag='val', dest=movDATA, classes=classes)

    print('Done')
