from conv import Conv3x3
from maxpool import MaxPool2
from softmax import Softmax

import numpy as np
import os
from PIL import Image

def load_images_from_folder(folder, size=(128, 128)):
    images = []
    for filename in os.listdir(folder):
        with Image.open(os.path.join(folder, filename)) as img:
            img = img.resize(size).convert('RGB')  # Convert to RGB
            images.append(np.array(img))
    return np.array(images)


folder_path = '../Dataset/'
folder_types = ['Mild_Demented/', 'Moderate_Demented/', 'Non_Demented/', 'Very_Mild_Demented/']

label_mapping = {
    'Mild_Demented': 0,
    'Moderate_Demented': 1,
    'Non_Demented': 2,
    'Very_Mild_Demented': 3
}


# NumPy 배열 초기화
train_images = np.empty((0, 128, 128, 3))
test_images = np.empty((0, 128, 128, 3))
train_labels = np.array([])
test_labels = np.array([])


for folder_type in folder_types:
    path = folder_path + folder_type
    all_images = load_images_from_folder(path)

    data_num = len(all_images)
    train_num = int(data_num * 0.8)

    # 이미지 배열 분할
    train_img = all_images[:train_num]
    test_img = all_images[train_num:]

    # 이미지 배열을 기존 배열에 추가
    train_images = np.concatenate((train_images, train_img), axis=0)
    test_images = np.concatenate((test_images, test_img), axis=0)

    # 레이블 배열 생성 및 추가
    label = folder_type[:-1]
    train_labels = np.concatenate((train_labels, np.full(train_num, label)))
    test_labels = np.concatenate((test_labels, np.full(len(test_img), label)))

def calculate_flattened_output_size(input_shape, num_filters):
    # Unpack the input shape
    height, width, _ = input_shape

    # Output shape after Conv3x3 (reduces height and width by 2)
    conv_output_height = height - 2
    conv_output_width = width - 2

    # Output shape after MaxPool2 (halves the height and width)
    pool_output_height = conv_output_height // 2
    pool_output_width = conv_output_width // 2

    # Flattened output size
    flattened_size = pool_output_height * pool_output_width * num_filters
    return flattened_size

# Example usage with an input image of size 128x128x3 and 8 filters
input_shape = (128, 128, 3)
num_filters = 8
flattened_output_size = calculate_flattened_output_size(input_shape, num_filters)


conv = Conv3x3(8)
pool = MaxPool2()
softmax = Softmax(flattened_output_size, 4)  # Adjusted to match the flattened output size of the previous layer

def forward(image, label):
    # Convert the string label to an integer using the label mapping
    label_index = label_mapping[label.rstrip('/')]  # Remove trailing slash if present

    out = conv.forward((image / 255) - 0.5)
    out = pool.forward(out)
    out = softmax.forward(out)

    # Cross entropy loss와 accuracy 계산
    loss = -np.log(out[label_index])
    acc = 1 if np.argmax(out) == label_index else 0

    return out, loss, acc

print("Begin CNN")

loss = 0
num_correct = 0


for i, (im, label) in enumerate(zip(test_images, test_labels)):
    # Forward pass
    _, l, acc = forward(im, label)
    loss += l
    num_correct += acc

    # for문이 100번 돌 때마다 실행
    if i % 100 == 99:
        print(
            '[Step %d] Past 100 steps: Average Loss %.3f | Accuracy: %d%%' %
            (i + 1, loss / 100, num_correct)
        )
        loss = 0
        num_correct = 0


def train(im, label, lr=.005):
    # Forward
    out, loss, acc = forward(im, label)

    # Calculate initial gradient
    gradient = np.zeros(10)
    gradient[label] = -1 / out[label]

    # Backprop
    gradient = softmax.backprop(gradient, lr)

    return loss, acc


print("MNIST CNN initialized!")
loss = 0
num_correct = 0
for i, (im, label_index) in enumerate(zip(train_images, train_labels)):
    if i % 100 == 99:
        print(
            '[Step %d] Past 100 steps: Average Loss %.3f | Accuracy: %d%%' %
            (i + 1, loss / 100, num_correct)
        )
        loss = 0
        num_correct = 0
        pass
    l, acc = train(im, label_index)
    loss += l
    num_correct += acc
