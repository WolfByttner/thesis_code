# Original work Copyright (c) 2019 Christoph Hofer
# Modified work Copyright (c) 2019 Wolf Byttner
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

import scipy
import skimage.morphology

import multiprocessing
import os
import sys
import numpy as np

from birds_data import images, labels, categories, training_data_labels, \
                       load_bounding_box_image
from chofer_nips2017.src.sharedCode.provider import Provider

sys.path.append(os.path.join(os.getcwd(), "chofer_nips2017", "tda-toolkit"))

from pershombox import calculate_discrete_NPHT_2d


colours = ['red', 'green', 'blue']


def threshold_diagram(persistence_diagram):
    t = 0.01
    return [pdgm for pdgm in persistence_diagram if pdgm[1] - pdgm[0] > t]


def calculate_npht_2d(monochrome_image, directions):
    label_map, count = skimage.morphology.label(monochrome_image,
                                                neighbors=4,
                                                background=0,
                                                return_num=True)
    volumes = [np.count_nonzero(label_map == (i + 1)) for i in range(count)]
    arg_max = np.argmax(volumes)
    label_image = (label_map == (arg_max + 1))
    label_image = np.ndarray.astype(label_image, bool)
    return calculate_discrete_NPHT_2d(label_image, directions)


def calculate_rotated_diagrams(monochrome_image, directions):
    diagrams = {}
    error = None
    npht = calculate_npht_2d(monochrome_image, directions)
    diagrams_dim_0 = [threshold_diagram(dgm[0]) for dgm in npht]
    diagrams_dim_1 = [threshold_diagram(dgm[1]) for dgm in npht]

    for direction, dgm_0, dgm_1 in zip(range(directions),
                                       diagrams_dim_0, diagrams_dim_1):
        if len(dgm_0) == 0:
            error = 'Diagram is degenerage'
            break

        diagrams['dim_0_dir_{}'.format(direction)] = dgm_0
        diagrams['dim_1_dir_{}'.format(direction)] = dgm_1
    return diagrams, error


def generate_diagram_job(arguments):
    image_id = arguments['image_id']
    directions = arguments['directions']
    resampled_size = arguments['resampled_size']
    histogram_normalised = arguments['histogram_normalised']
    rgb = arguments['rgb']

    result = {'label': labels[image_id], 'image_id': image_id, 'diagrams': {}}
    print("Processing {}".format(image_id))
    try:
        image_orig = load_bounding_box_image(image_id)
        if resampled_size is None:
            image_resampled = image_orig
        else:
            image_resampled = scipy.misc.imresize(image_orig,
                                                  (*resampled_size, 3),
                                                  interp='bilinear')

        if histogram_normalised:
            image_histogram, bins = np.histogram(image_resampled.flatten(),
                                                 256, density=True)
            cdf = image_histogram.cumsum()  # cumulative distribution function
            cdf = 255 * cdf / cdf[-1]  # normalize

            # use linear interpolation of cdf to find new pixel values
            image_equalized = np.interp(image_resampled.flatten(),
                                        bins[:-1], cdf)

            image = image_equalized.reshape(image_resampled.shape)
        else:
            image = image_resampled

        if rgb:
            image_red = image[:, :, 0]
            image_green = image[:, :, 1]
            image_blue = image[:, :, 2]

            for colour, monochrome_image in [('red', image_red),
                                             ('green', image_green),
                                             ('blue', image_blue)]:
                diagrams, error = calculate_rotated_diagrams(monochrome_image,
                                                             directions)
                if error is None:
                    result['diagrams'][colour] = diagrams
                else:
                    raise RuntimeError(error)
        else:
            try:
                image_gray = (np.sum(image, axis=2) // 3)
            except Exception:  # Grayscale image to begin with
                image_gray = image
            monochrome_image = image_gray
            diagrams, error = calculate_rotated_diagrams(monochrome_image,
                                                         directions)
            if error is None:
                result['diagrams']['gray'] = diagrams
            else:
                raise RuntimeError(error)

    except Exception as e:
        result['exception'] = e
        print(e)

    print("Returning {}".format(image_id))
    return result


def get_folder_string(directions, resampled_size, output_path,
                      histogram_normalised):
    if resampled_size is None:
        folder_string = "resampled_raw_"
    else:
        folder_string = "resampled_{}x{}_".format(*resampled_size)

    folder_string += "directions_{}".format(directions)
    if histogram_normalised:
        folder_string += "_histnorm"

    return os.path.join(output_path, folder_string)


def rotate_all_persistence_diagrams(directions, resampled_size, output_path,
                                    histogram_normalised=False, rgb=True):
    if rgb:
        colours = globals()['colours']
    else:
        colours = ['gray']
    output_path = get_folder_string(directions, resampled_size,
                                    output_path, histogram_normalised)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    if rgb:
        views = {'red': {}, 'green': {}, 'blue': {}}
    else:
        views = {'gray': {}}
    for i in range(0, directions):
        for colour in colours:
            views[colour]['dim_0_dir_{}'.format(i)] = {}
            views[colour]['dim_1_dir_{}'.format(i)] = {}

    job_arguments = []

    for category in categories:
        for colour in colours:
            for view in views[colour].values():
                view[str(category)] = {}
    for image_id in images:
        arguments = {'image_id': image_id,
                     'directions': directions,
                     'resampled_size': resampled_size,
                     'rgb': rgb,
                     'histogram_normalised': histogram_normalised}
        job_arguments.append(arguments)

    with multiprocessing.Pool() as pool:
        errors = {}
        for i, result in enumerate(pool.imap_unordered(generate_diagram_job,
                                                       job_arguments)):
            label = str(result['label'])
            image_id = str(result['image_id'])

            if 'exception' not in result:
                for colour in colours:
                    for view_id, diagram in result['diagrams'][colour].items():
                        views[colour][view_id][label][image_id] = diagram
            else:
                errors[image_id] = result['exception']

    for image_id, error in errors.items():
        print("{} had error {}".format(image_id, error))
    print('Writing {} of {} samples to disk'.format(len(views[colours[0]]),
                                                    len(job_arguments)))

    for colour in colours:
        print("Saving {}".format(colour))
        provider = Provider({}, None, {})
        for view_id, view_data in views[colour].items():
            print(view_id)
            provider.add_view(view_id, view_data)
        meta_data = {'number_of_directions': directions}
        provider.add_meta_data(meta_data)

        provider.dump_as_h5(os.path.join(output_path, colour + '.h5'))

    print("Data saved, Exiting.")


if __name__ == '__main__':
    outpath = os.path.join(os.path.dirname(__file__), 'h5images/')

    rotate_all_persistence_diagrams(32, (64, 64), outpath)
