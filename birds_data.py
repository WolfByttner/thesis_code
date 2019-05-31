import os
from imageio import imread
import tarfile
from urllib.request import urlretrieve


data_path = os.path.join(os.getcwd(), 'Data', 'CUB_200_2011')


def setup_data_folder():
    data_folder = os.path.join(os.getcwd(), 'Data')
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)


def download_birds_data():
    url = 'http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz'
    tarball = os.path.join(os.getcwd(), 'Data/CUB_200_2011.tgz')
    target_directory =  os.path.join(os.getcwd(), 'Data')

    setup_data_folder()
    if os.path.exists(data_path):
        return

    filename, headers = urlretrieve(url, tarball)

    tar = tarfile.open(tarball)
    tar.extractall(path=target_directory)
    tar.close()


def load_images_list(path):
    images_path = os.path.join(path, 'images.txt')
    images = {}
    f = open(images_path, 'r')
    for line in f.read().split("\n"):
        if len(line) < 1:
            continue
        image_id, image_path = line.split(" ")
        images[image_id] = image_path
    f.close()
    return images


def load_bounding_boxes(path):
    bounding_boxes_path = os.path.join(path, 'bounding_boxes.txt')
    boxes = {}
    f = open(bounding_boxes_path, 'r')
    for line in f.read().split("\n"):
        if len(line) < 1:
            continue
        image_id, x, y, width, height = line.split(" ")
        boxes[image_id] = [x, y, width, height]
    f.close()
    return boxes


def load_class_labels(path):
    class_labels_path = os.path.join(path, 'image_class_labels.txt')
    labels = {}
    f = open(class_labels_path, 'r')
    for line in f.read().split("\n"):
        if len(line) < 1:
            continue
        image_id, image_label = line.split(" ")
        labels[image_id] = int(image_label)
    f.close()
    return labels


def load_categories(labels):
    categories = list(set(labels.values()))
    return categories


def load_training_data_labels(path):
    training_class_labels_path = os.path.join(path, 'train_test_split.txt')
    training_labels = {}
    f = open(training_class_labels_path, 'r')
    for line in f.read().split("\n"):
        if len(line) < 1:
            continue
        image_id, image_label = line.split(" ")
        training_labels[image_id] = bool(int(image_label))
    f.close()
    return training_labels


download_birds_data()
images = load_images_list(data_path)
boxes = load_bounding_boxes(data_path)
labels = load_class_labels(data_path)
categories = load_categories(labels)
training_data_labels = load_training_data_labels(data_path)


def load_bounding_box_image(image_id):
    image_path = os.path.join(data_path, 'images', images[image_id])
    x, y, width, height = map(int,map(float,boxes[image_id]))
    image = imread(image_path)
    try:
        bounded_image = image[x:x+width,y:y+height,:]
    except IndexError:
        bounded_image = image[x:x+width,y:y+height]
    return bounded_image
