import torch
import torch.nn as nn
import numpy as np

import sys
import os

import multiprocessing

sys.path.append(os.path.join(os.path.dirname(__file__),"chofer_nips2017/chofer_torchex"))
from sklearn.preprocessing.label import LabelEncoder
from torch import optim
from torch.utils.data import SubsetRandomSampler, DataLoader
from rotated_persistence_diagrams_rgb import colours, get_folder_string
from birds_data import images, labels, categories, training_data_labels
from chofer_nips2017.src.sharedCode.provider import Provider
from chofer_nips2017.src.sharedCode.experiments import \
    UpperDiagonalThresholdedLogTransform, \
    pers_dgm_center_init, SLayerPHT, \
    PersistenceDiagramProviderCollate
from sklearn.model_selection import StratifiedShuffleSplit

import chofer_torchex.utils.trainer as tr
from chofer_torchex.utils.trainer.plugins import *
from rotated_persistence_diagrams_rgb import rotate_all_persistence_diagrams


def _parameters():
    return \
    {
        'colour_mode': 'grayscale',
        'data_path': None,
        'epochs': 30,
        'momentum': 0.2,
        'lr_start': 0.1,
        'lr_ep_step': 20,
        'lr_adaption': 0.5,
        'test_ratio': 0.1,
        'batch_size': 8,
        'directions': 32,
        'resampled_size': (800,800),
        'cuda': False
    }


def serialise_params(params):
    serial = ""
    for key, val in params.items():
        serial += "_{}_{}".format(key, val)
    return serial

class SLayerRgbNN(torch.nn.Module):
    def __init__(self, subscripted_views, directions):
        super(SLayerRgbNN, self).__init__()
        self.subscripted_views = subscripted_views

        print(subscripted_views)
        n_elements = 75
        n_filters = directions
        stage_2_out = 25
        n_neighbor_directions = 1

        self.transform = UpperDiagonalThresholdedLogTransform(0.1)

        self.pht_sl = SLayerPHT(len(subscripted_views),
                                n_elements,
                                2,
                                n_neighbor_directions=n_neighbor_directions,
                                center_init=self.transform(pers_dgm_center_init(n_elements)),
                                sharpness_init=torch.ones(n_elements, 2) * 4)

        self.stage_1 = []
        for i in range(len(subscripted_views)):
            seq = nn.Sequential()
            seq.add_module('conv_1', nn.Conv1d(1 + 2 * n_neighbor_directions, n_filters, 1, bias=False))
            seq.add_module('conv_2', nn.Conv1d(n_filters, 8, 1, bias=False))
            self.stage_1.append(seq)
            self.add_module('stage_1_{}'.format(i), seq)

        self.stage_2 = []
        for i in range(len(subscripted_views)):
            seq = nn.Sequential()
            seq.add_module('linear_1', nn.Linear(n_elements, stage_2_out))
            seq.add_module('batch_norm', nn.BatchNorm1d(stage_2_out))
            seq.add_module('linear_2'
                           , nn.Linear(stage_2_out, stage_2_out))
            seq.add_module('relu', nn.ReLU())
            seq.add_module('Dropout', nn.Dropout(0.4))

            self.stage_2.append(seq)
            self.add_module('stage_2_{}'.format(i), seq)

        linear_1 = nn.Sequential()
        linear_1.add_module('linear', nn.Linear(len(subscripted_views) * stage_2_out, 500))
        linear_1.add_module('batchnorm', torch.nn.BatchNorm1d(500))
        linear_1.add_module('drop_out', torch.nn.Dropout(0.3))
        self.linear_1 = linear_1

        linear_2 = nn.Sequential()
        linear_2.add_module('linear', nn.Linear(500, 200))

        self.linear_2 = linear_2

    def forward(self, batch):
        x = [batch[n] for n in self.subscripted_views]
        x = [[self.transform(dgm) for dgm in view_batch] for view_batch in x]

        x = self.pht_sl(x)

        x = [l(xx) for l, xx in zip(self.stage_1, x)]

        x = [torch.squeeze(torch.max(xx, 1)[0]) for xx in x]

        x = [l(xx) for l, xx in zip(self.stage_2, x)]

        x = torch.cat(x, 1)
        x = self.linear_1(x)
        x = self.linear_2(x)
        return x


def train_test_from_dataset(dataset,
                            batch_size):
    sample_labels = list(dataset.sample_labels)
    label_encoder = LabelEncoder().fit(sample_labels)
    sample_labels = label_encoder.transform(sample_labels)

    label_map = lambda l: int(label_encoder.transform([l])[0])

    collate_fn = PersistenceDiagramProviderCollate(dataset, label_map=label_map)

    train_ids = np.array([label_map(image_id) for image_id in dataset.sample_labels if training_data_labels[image_id]])
    test_ids = np.array([label_map(image_id) for image_id in dataset.sample_labels if not training_data_labels[image_id]])
    #sp = StratifiedShuffleSplit(n_splits=1, test_size=0.5)
    #train_ids, test_ids = list(sp.split([0]*len(sample_labels), sample_labels))[0]

    data_train = DataLoader(dataset,
                            batch_size=batch_size,
                            collate_fn=collate_fn,
                            shuffle=False,
                            sampler=SubsetRandomSampler(train_ids.tolist()))

    data_test = DataLoader(dataset,
                           batch_size=batch_size,
                           collate_fn=collate_fn,
                           shuffle=False,
                           sampler=SubsetRandomSampler(test_ids.tolist()))

    return data_train, data_test


def read_provider(data_path):
    dataset = Provider(dict(), None, dict())
    dataset.read_from_h5(data_path)
    return dataset


def load_data(params):
    rgb = params['colour_mode'] == 'rgb'
    grayscale_wide = params['colour_mode'] == 'grayscale_wide'

    view_name_template = 'dim_0_dir_{}'
    subscripted_views = sorted([view_name_template.format(i) for i in range(params['directions'])])
    assert (str(len(subscripted_views)) in params['data_path'])

    subscripted_views_colour = []
    if rgb or grayscale_wide:
        for colour in colours:
            for view in subscripted_views:
                subscripted_views_colour.append(view + '_{}'.format(colour))
    else:
        subscripted_views_colour = subscripted_views

    print('Loading providers')
    data_paths = []
    datasets = []
    if rgb or grayscale_wide:
        if grayscale_wide:
            print("Accuracy: Modified grayscale")
        for colour in colours:
            if rgb:
                datasets.append(read_provider(os.path.join(params['data_path'], colour + '.h5')))
            else:
                datasets.append(read_provider(os.path.join(params['data_path'], 'gray' + '.h5')))

        print("Merging providers")
        merged_dataset = Provider(dict(), None, dict())
        for i, colour in enumerate(colours):
            for view in datasets[i].data_views:
                key = view + "_{}".format(colour)
                merged_dataset.add_view(key, datasets[i].data_views[view])

    else:
        merged_dataset = read_provider(os.path.join(params['data_path'], 'gray.h5'))

    print('Create data loader...')
    data_train, data_test = train_test_from_dataset(merged_dataset,
                                                    batch_size=params['batch_size'])

    return data_train, data_test, subscripted_views_colour


def setup_trainer(model, params, data_train, data_test):
    optimizer = optim.SGD(model.parameters(),
                          lr=params['lr_start'],
                          momentum=params['momentum'])

    loss = nn.CrossEntropyLoss()

    trainer = tr.Trainer(model=model,
                         optimizer=optimizer,
                         loss=loss,
                         train_data=data_train,
                         n_epochs=params['epochs'],
                         cuda=params['cuda'],
                         variable_created_by_model=True)

    def determine_lr(self, **kwargs):
        epoch = kwargs['epoch_count']
        if epoch % params['lr_ep_step'] == 0:
            return params['lr_start'] / 2 ** (epoch / params['lr_ep_step'])

    lr_scheduler = LearningRateScheduler(determine_lr, verbose=True)
    lr_scheduler.register(trainer)

    progress = ConsoleBatchProgress()
    progress.register(trainer)

    prediction_monitor_test = PredictionMonitor(data_test,
                                                verbose=True,
                                                eval_every_n_epochs=1,
                                                variable_created_by_model=True)
    prediction_monitor_test.register(trainer)
    trainer.prediction_monitor = prediction_monitor_test

    return trainer


def train_network(parameters):
    if torch.cuda.is_available():
        parameters['cuda'] = True
    else:
        # Let the CPU cluster chew large pieces
        parameters['batch_size'] = 128

    print(params)

    print('Data setup...')
    data_train, data_test, subscripted_views = load_data(parameters)

    print("Creating network")
    model = SLayerRgbNN(subscripted_views, parameters['directions'])

    print("Creating trainer")
    trainer = setup_trainer(model, params, data_train, data_test)

    print("Running training")

    trainer.run()

    print("Saving model")
    model_path = os.path.join(parameters['data_path'],
                              "model" + serialise_params(parameters) + ".torch")
    torch.save(model.state_dict(), model_path)

    last_10_accuracies = list(trainer.prediction_monitor.accuracies.values())[-10:]
    mean = np.mean(last_10_accuracies)

    return mean



if __name__ == '__main__':
    params = _parameters()
    histogram_normalised=True
    outpath = os.path.join(os.path.dirname(__file__), 'h5images')
    params['data_path'] = get_folder_string(32, params['resampled_size'],
                                            outpath,
                                            histogram_normalised=histogram_normalised)
    if params['colour_mode'] == 'rgb':
        if not os.path.exists(os.path.join(params['data_path'],'red.h5')):
            do_stuff(params['directions'], params['resampled_size'],
                     outpath,histogram_normalised, rgb=True)
    elif params['colour_mode'] == 'grayscale' or params['colour_mode'] == 'grayscale_wide':
        if not os.path.exists(os.path.join(params['data_path'],'gray.h5')):
            rotate_all_persistence_diagrams(params['directions'],
                                            params['resampled_size'],outpath,
                                            histogram_normalised, rgb=False)
    else:
        raise RuntimeError('Parameter colour_mode = {} not recognised!' \
                           .format(params['colour_mode']))
    mean = train_network(params)
    print("Mean is {}".format(mean))

