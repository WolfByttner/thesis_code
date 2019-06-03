# Copyright (c) 2019 Wolf Byttner
#
# This file is part of the code implementing the thesis
# "Classifying RGB Images with multi-colour Persistent Homology".
#
#     This file is free software: you can redistribute it and/or modify
#     it under the terms of the GNU Lesser General Public License as published
#     by the Free Software Foundation, either version 3 of the License, or
#     (at your option) any later version.
#
#     This file is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU Lesser General Public License for more details.
#
#     You should have received a copy of the GNU Lesser General Public License
#     along with this file.  If not, see <https://www.gnu.org/licenses/>.


from diamorse.python import persistence
from imageio import imread
import datetime
import netCDF4 as nc4
import sys
import os
import scipy
import scipy.misc
import numpy as np
from matplotlib import pyplot as plt
from collections import defaultdict
from math import pi, e, atan
from functools import reduce, partial
from sklearn import svm, linear_model, neural_network
import pickle
from persim.persim import images as persim
import multiprocessing
import types
from operator import itemgetter
from birds_data import images, labels, categories, training_data_labels, \
                       load_bounding_box_image


def create_netcdf_from_image_id(image_id, path, images, boxes, out_path):
    image_id = str(image_id)
    time = [datetime.datetime(2000, 10, 1, 1, val) for val in range(60)]
    if not os.path.exists(out_path + images[image_id]):
        os.makedirs(out_path + images[image_id])
    print(images[image_id])
    image_orig = load_bounding_box_image(image_id, path, images, boxes)
    image_interp = scipy.misc.imresize(image_orig,
                                       (interp_width, interp_height, 3),
                                       interp='bilinear')
    try:
        image = (np.sum(image, axis=2) // 3)
    except Exception as e:
        print(e)
        image = image_interp

    image_histogram, bins = np.histogram(image.flatten(), 256, density=True)
    cdf = image_histogram.cumsum()  # cumulative distribution function
    cdf = 255 * cdf / cdf[-1]  # normalize

    # use linear interpolation of cdf to find new pixel values
    image_equalized = np.interp(image.flatten(), bins[:-1], cdf)

    image = image_equalized.reshape(image.shape)

    nc_path = os.path.join(out_path, images[image_id], interp_str + 'img.nc')

    nc = nc4.Dataset(nc_path, 'w', format='NETCDF3_CLASSIC')
    x_dim = 'x_dim'
    y_dim = 'y_dim'
    z_dim = 'z_dim'
    dims = image.shape
    nc.createDimension(x_dim, dims[0])
    nc.createDimension(y_dim, dims[1])
    nc.createDimension(z_dim, 1)
    pic = nc.createVariable('IMP', 'f4', (x_dim, y_dim, z_dim))
    print(image_orig.shape)
    print(image.shape)
    pic[:, :, 0] = (np.sum(image, axis=2) // 3)
    nc.close()
    return nc_path


class diamorse_options:
    def __init__(self, infile):
        self.betti = True
        self.threshold = 1
        self.infile = infile
        self.field = bytes('', 'utf-8')


def persistent_homology_from_image_id(image_id, path, images,
                                      boxes, out_path,
                                      save_pickle, load_pickle):
    pickle_path = os.path.join(out_path, images[image_id],
                               interp_str + 'dump.pickle')
    if load_pickle and os.path.exists(pickle_path):
        with open(pickle_path, 'rb') as f:
            pairs = pickle.load(f)
    else:
        nc_path = create_netcdf_from_image_id(image_id, path, images,
                                              boxes, out_path)
        nc_path = bytes(nc_path, 'utf-8')
        opts = diamorse_options(nc_path)

        pairs = persistence.fromVolumeFile(nc_path, opts)
        if save_pickle:
            with open(str(pickle_path), 'wb') as f:
                pickle.dump(pairs, f)
    return pairs


def compute_persim_from_image_id(image_id, settings, transformer):
    persistence_pairs = persistent_homology_from_image_id(image_id,
                                                          settings.data_path,
                                                          settings.images,
                                                          settings.boxes,
                                                          settings.out_path,
                                                          settings.save_pickle,
                                                          settings.load_pickle)
    diagrams = [[], []]
    for pair in persistence_pairs:
        if pair[1] == float('inf'):
            continue
        diagrams[pair[2]].append((pair[0], pair[1]))
    persistence_images = transformer.transform(diagrams)
    return persistence_images[0].flatten() + persistence_images[1].flatten()


def get_training_data_for_labels(label_list, labels, training_data_labels):
    training_labels = defaultdict(list)
    test_labels = defaultdict(list)
    for image_label in labels:
        if labels[image_label] in label_list:
            if training_data_labels[image_label]:
                training_labels[labels[image_label]].append(image_label)
            else:
                test_labels[labels[image_label]].append(image_label)
    return training_labels, test_labels


def create_persistence_image_and_label(payload):
    image_id, settings, transformer = payload
    try:
        sample = (compute_persim_from_image_id(image_id, settings,
                                               transformer),
                  settings.labels[str(image_id)], image_id)
    except Exception as e:
        print(e)
        sample = (None, None, image_id)
    return sample


def weighting(self, landscape=None, print_self=False, print_compact=False):
    C = 0.5
    p = 2
    if print_self:
        if print_compact:
            return "C({})_p({})".format(C, p)
        else:
            print("C: {}, p: {}".format(C, p))

    def arctan(interval):
        return atan(C * (interval[1] - interval[0])**p)

    return arctan


def create_persistence_images_and_labels(image_labels_dict, settings,
                                         is_training=False):
    transformer = persim.PersImage(settings.persistence_image_size,
                                   spread=settings.gaussian_stddev)
    transformer.weighting = weighting
    data = []
    label_list = []
    image_id_list = []

    if is_training:
        dataset_str = "training/"
    else:
        dataset_str = "evaluation/"
    cache_dir = os.path.join(settings.out_path,
                             "persistence_images_" + interp_str[:-1],
                             (settings.cache_str() +
                              transformer.weighting(None, None, True, True)),
                             dataset_str)
    if settings.load_pickle and os.path.exists(cache_dir):
        with open(cache_dir + "data.pickle", 'rb') as f:
            data = pickle.load(f)
        with open(cache_dir + "labels.pickle", 'rb') as f:
            label_list = pickle.load(f)
        with open(cache_dir + "image_ids.pickle", 'rb') as f:
            image_id_list = pickle.load(f)
        print("Loaded {} samples from cache".format(len(image_id_list)))

    image_ids = []
    for key in image_labels_dict:
        image_label_list = image_labels_dict[key]
        for image_id in image_label_list:
            if image_id in image_id_list:
                continue
            image_ids.append((image_id, settings, transformer))

    print("Running computations on {} of {} images".format(len(image_ids),
                                                           len(image_ids) +
                                                           len(image_id_list)))
    transformer.weighting(None, None, print_self=True)
    print("Starting thread pool")
    with multiprocessing.Pool() as pool:
        persistence_images_with_labels = \
            pool.map(create_persistence_image_and_label, image_ids)
    for image, label, image_id in persistence_images_with_labels:
        if label is None:
            continue
        data.append(image)
        label_list.append(label)
        image_id_list.append(image_id)

    if settings.save_pickle:
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        with open(cache_dir + "data.pickle", 'wb') as f:
            pickle.dump(data, f)
        with open(cache_dir + "labels.pickle", 'wb') as f:
            pickle.dump(label_list, f)
        with open(cache_dir + "image_ids.pickle", 'wb') as f:
            pickle.dump(image_id_list, f)

    return data, label_list


class prediction_evaluator:
    def __init__(self):
        self.correct = 0
        self.existing = 0

    def add_prediction(self, prediction, truth):
        if prediction == truth:
            self.correct += 1
        self.existing += 1

    def correctness(self):
        return self.correct / self.existing

    def __repr__(self):
        return "({}, {})".format(str(self.correct), str(self.existing))


def evaluate_data_with_model(payload):
    correctness = defaultdict(prediction_evaluator)
    training_data, training_labels, test_data, test_labels, clf = payload
    clf.fit(training_data, training_labels)
    predicted_labels = clf.predict(test_data)
    for i in range(len(predicted_labels)):
        correctness[test_labels[i]].add_prediction(predicted_labels[i],
                                                   test_labels[i])
    agg_true = 0
    agg_correct = 0
    for label in correctness:
        agg_correct += correctness[label].correct
        agg_true += correctness[label].existing
    return (agg_correct/agg_true, clf)


def handle_model_timeout(payload):
    try:
        return evaluate_data_with_model(payload)
    except multiprocessing.TimeoutError:
        clf = payload[-1]
        return (0, clf)


def evaluate_data(training_data, training_labels, test_data, test_labels):
    max_iter = 3000
    classifiers = []

    # It is easier to just test everything.
    classifiers.append(
            linear_model.LogisticRegression(random_state=0, solver='newton-cg',
                                            multi_class='multinomial',
                                            max_iter=max_iter))
    classifiers.append(
            linear_model.LogisticRegression(random_state=0, solver='lbfgs',
                                            multi_class='multinomial',
                                            max_iter=max_iter))
    classifiers.append(
            linear_model.LogisticRegression(random_state=0, solver='liblinear',
                                            multi_class='ovr'))
    classifiers.append(
            linear_model.LogisticRegression(random_state=0, solver='liblinear',
                                            multi_class='ovr', penalty='l1',
                                            max_iter=max_iter))
    classifiers.append(
            linear_model.LogisticRegression(random_state=0, solver='sag',
                                            multi_class='multinomial',
                                            max_iter=max_iter))
    classifiers.append(
            linear_model.LogisticRegression(random_state=0, solver='saga',
                                            multi_class='multinomial',
                                            max_iter=max_iter))

    classifiers.append(
            linear_model.LogisticRegressionCV(random_state=0,
                                              solver='newton-cg',
                                              multi_class='multinomial',
                                              max_iter=max_iter))
    classifiers.append(
            linear_model.LogisticRegressionCV(random_state=0, solver='lbfgs',
                                              multi_class='multinomial',
                                              max_iter=max_iter))
    classifiers.append(
            linear_model.LogisticRegressionCV(random_state=0,
                                              solver='liblinear',
                                              multi_class='ovr',
                                              max_iter=max_iter))
    classifiers.append(
            linear_model.LogisticRegressionCV(random_state=0,
                                              solver='liblinear',
                                              multi_class='ovr', penalty='l1',
                                              max_iter=max_iter))
    classifiers.append(
            linear_model.LogisticRegressionCV(random_state=0, solver='sag',
                                              multi_class='multinomial',
                                              max_iter=max_iter))
    classifiers.append(
            linear_model.LogisticRegressionCV(random_state=0, solver='saga',
                                              multi_class='multinomial',
                                              max_iter=max_iter))

    classifiers.append(neural_network.MLPClassifier(solver='adam',
                                                    random_state=0,
                                                    activation='identity'))
    classifiers.append(neural_network.MLPClassifier(solver='adam',
                                                    random_state=0,
                                                    activation='logistic'))
    classifiers.append(neural_network.MLPClassifier(solver='adam',
                                                    random_state=0,
                                                    activation='tanh'))
    classifiers.append(neural_network.MLPClassifier(solver='adam',
                                                    random_state=0,
                                                    activation='relu'))

    classifiers.append(neural_network.MLPClassifier(solver='lbfgs',
                                                    random_state=0,
                                                    activation='identity'))
    classifiers.append(neural_network.MLPClassifier(solver='lbfgs',
                                                    random_state=0,
                                                    activation='logistic'))
    classifiers.append(neural_network.MLPClassifier(solver='lbfgs',
                                                    random_state=0,
                                                    activation='tanh'))
    classifiers.append(neural_network.MLPClassifier(solver='lbfgs',
                                                    random_state=0,
                                                    activation='relu'))

    classifiers.append(svm.LinearSVC(multi_class="crammer_singer",
                                     random_state=0, penalty='l1',
                                     dual=False))
    classifiers.append(svm.LinearSVC(multi_class="crammer_singer",
                                     random_state=0, penalty='l2'))
    classifiers.append(svm.LinearSVC(multi_class="ovr", random_state=0,
                                     penalty='l1', dual=False))
    classifiers.append(svm.LinearSVC(multi_class="ovr", random_state=0,
                                     penalty='l2'))

    classifiers.append(svm.SVC(kernel='linear', random_state=0))
    classifiers.append(svm.SVC(gamma='scale', random_state=0))

    payloads = []
    for classifier in classifiers:
        payloads.append((training_data, training_labels, test_data,
                         test_labels, classifier))
    print("Starting Model Evaluation")
    classification_rates = []

    async_jobs = []
    with multiprocessing.Pool() as pool:
        for payload in payloads:
            async_jobs.append(pool.apply_async(handle_model_timeout,
                                               (payload,)))
        for async_job in async_jobs:
            try:
                classification_rates.append(async_job.get(timeout=400))
            except multiprocessing.TimeoutError:
                pass
            except Exception as e:
                pass
    best_pair = sorted(classification_rates, key=itemgetter(0),
                       reverse=True)[0]
    print("Best classifiers are ({:.2%}):".format(best_pair[0]))
    print(best_pair[1])


def evaluate_labels(label_list, settings):
    training_ids, test_ids = \
        get_training_data_for_labels(label_list, settings.labels,
                                     settings.training_data_labels)

    training_data, training_labels = \
        create_persistence_images_and_labels(training_ids, settings,
                                             is_training=True)
    test_data, test_labels = create_persistence_images_and_labels(test_ids,
                                                                  settings)
    evaluate_data(training_data, training_labels, test_data, test_labels)


class persistent_image_settings:
    def __init__(self, images, boxes, out_path, data_path,
                 labels, training_data_labels,
                 persistence_image_size, gaussian_stddev):
        self.images = images
        self.boxes = boxes
        self.out_path = out_path
        self.data_path = data_path
        self.labels = labels
        self.training_data_labels = training_data_labels
        self.persistence_image_size = persistence_image_size
        self.gaussian_stddev = gaussian_stddev

    def weight_function(self, z):
        return min(z[1]/200, 200)
    gaussian_var = 0.2
    grid_step = 10
    grid_width = 20
    save_pickle = True
    load_pickle = False

    def cache_str(self):
        pi_string = "pi({}_{})_".format(*self.persistence_image_size)
        return pi_string + "stddev({})_".format(self.gaussian_stddev)


interp_width = 200
interp_height = 200
interp_str = "{}x{}_".format(interp_width, interp_height) if \
        (interp_width or interp_height) else ""


print("Interpolation:", interp_str[:-1])
for stddev in [0.5, 1, 1.5, 2]:
    out_p = os.path.join(os.getcwd(), "tmp")
    pi_settings = persistent_image_settings(images, boxes, out_p, data_path,
                                            labels, training_data_labels,
                                            persistence_image_size=(128, 128),
                                            gaussian_stddev=stddev)

    evaluate_labels(list(range(1, 201)), pi_settings)
