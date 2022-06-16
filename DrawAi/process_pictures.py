import math
import random

import cv2
import os
import numpy as np
from math import ceil
import time

import multiprocessing as mp


def find_all_files_in(rootdir, name, files):
    for file in os.listdir(rootdir):
        fullPath = os.path.join(rootdir, file)
        if os.path.isdir(fullPath):
            find_all_files_in(fullPath, name, files)
        elif name in rootdir and os.path.isfile(fullPath) and ".meta" not in fullPath:
            files.append(fullPath)


path = "C:\\Users\\User\Downloads\Angle Laser Pictures-20220523T141231Z-001\Angle Laser Pictures"


def resize_without_borders(AIColors, Size):
    # start = time.time()
    max = np.zeros(2)
    min = np.zeros(2)

    max[0] = float('-inf')
    max[1] = float('-inf')

    min[0] = float('inf')
    min[1] = float('inf')

    for i in range(len(AIColors)):
        y = int(i / Size)
        x = i - Size * y
        if AIColors[i] == 1:
            if x > max[0]:
                max[0] = x

            if x < min[0]:
                min[0] = x

            if y > max[1]:
                max[1] = y

            if y < min[1]:
                min[1] = y

    max = max + 1

    side1 = np.array([min[0], max[1]]) - min
    side2 = np.array([max[0], min[1]]) - min

    length = 0

    lenSide1 = np.sqrt(np.dot(side1, side1))
    lenSide2 = np.sqrt(np.dot(side2, side2))
    if lenSide1 > lenSide2:
        length = lenSide1 / 2
    else:
        length = lenSide2 / 2

    center = np.array([(max[0] + min[0]) / 2, (max[1] + min[1]) / 2])

    sqrPoint = center + np.array([0, -1]) * length + np.array([-1, 0]) * length

    inputSize = 26

    input = np.zeros(((inputSize + 2) * (inputSize + 2)))

    factor = length * 2 / inputSize

    for i in range(inputSize):
        for j in range(inputSize):
            y = round(sqrPoint[1] + i * factor)
            x = round(sqrPoint[0] + j * factor)

            resValue = 0
            counter = 0
            for k in range(ceil(factor)):
                for l in range(ceil(factor)):
                    y1 = y + k
                    if y1 >= Size or y1 < 0:
                        continue
                    x1 = x + l

                    if x1 >= Size or x1 < 0:
                        continue

                    resValue += AIColors[y1 * Size + x1]

                    counter += 1

            if counter != 0:
                resValue /= counter

            input[(i + 1) * (inputSize + 2) + (j + 1)] = resValue

    # end = time.time()
    # print(end - start)
    return input


def show_image(image):
    cv2.imshow('sample image ' + str(random.random()), image)
    cv2.waitKey(0)


def show_array_image(image):
    show = cv2.resize(np.reshape(image, (28, 28)) * 255, (1000, 1000))
    show_image(show)


def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result


imageToProcessSize = 256

borderSizeCoef = math.sqrt(2) - 1  # get this by rotating square by 45 degrees
borderSize = ceil(imageToProcessSize * borderSizeCoef)

angleToRotate = 45
angleStep = 2


# angleStep = 100


def resize_image(image):
    return cv2.resize(image, (imageToProcessSize, imageToProcessSize))


def add_border(image):
    return cv2.copyMakeBorder(
        image,
        top=borderSize,
        bottom=borderSize,
        left=borderSize,
        right=borderSize,
        borderType=cv2.BORDER_CONSTANT,
        value=[0, 0, 0]
    )


def multiply_by_rotation(images):
    res = []
    for image in images.copy():
        for angle in range(-angleToRotate, angleToRotate, angleStep):
            rotatedImage = rotate_image(image, angle)
            res.append(rotatedImage)
    return res


kernel = np.ones((2, 2), 'uint8')
dilateIterationsAmount = 3


# dilateIterationsAmount = 1


def multiply_by_dilate_and_erode(images):
    for image in images.copy():
        for iterations in range(0, dilateIterationsAmount):
            erodedImage = cv2.erode(image, kernel, iterations=iterations)
            dilatedImage = cv2.dilate(image, kernel, iterations=iterations)
            images.append(dilatedImage)
            images.append(erodedImage)
            # show_image(erodedImage)
    return images


def process_image(image):
    image = image / 255
    size = image.shape[0]
    image = np.reshape(image, (size * size))
    res = resize_without_borders(image, size)
    res = res.astype("float32")
    return res


cpuCount = round(mp.cpu_count() / 2)


def process_all(files):
    result = []
    for file in files:
        start = time.time()
        print(file)

        image = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        image = resize_image(image)
        image = add_border(image)

        images = [image]
        images = multiply_by_rotation(images)
        images = multiply_by_dilate_and_erode(images)
        print(len(images))

        pool = mp.Pool(cpuCount)

        t = images[0]
        k = cv2.resize(t, (28, 28)) / 255
        # show_image(k)
        images = [images[0]]

        res = pool.map(process_image, images)
        pool.close()
        a = res[0].reshape((28, 28))
        result.extend(res)

        end = time.time()
        print(end - start)
    return result


def write_features(label):
    files = []
    find_all_files_in(path, label, files)
    # files = files[:5]
    result = process_all(files)

    # np.save(f"processed_data/{label.lower()}_features", result)


if __name__ == '__main__':
    # labels = []
    # labels = ["Axe", "Blaster", "Sword", "Knifes", "Hummer", "Spear", "Nunchucks", "Fork"]
    labels = ["Axe"]
    for label in labels:
        write_features(label)

# print("dsdds")
