#
#
#      0==================================0
#      |    Kernel Point Convolutions     |
#      0==================================0
#
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Class handling HumanBodySegmentation dataset
#
# ----------------------------------------------------------------------------------------------------------------------

import tensorflow as tf
import numpy as np
import time
import pickle
import os

from os import listdir, makedirs
from os.path import exists, join


# Dataset parent class
from datasets.common import Dataset

import cpp_wrappers.cpp_subsampling.grid_subsampling as cpp_subsampling


def grid_subsampling(points, features=None, labels=None, sampleDl=0.1, verbose=0):
    """
    CPP wrapper for a grid subsampling (method = barycenter for points and features
    """
    if (features is None) and (labels is None):
        return cpp_subsampling.compute(points, sampleDl=sampleDl, verbose=verbose)
    elif (labels is None):
        return cpp_subsampling.compute(points, features=features, sampleDl=sampleDl, verbose=verbose)
    elif (features is None):
        return cpp_subsampling.compute(points, classes=labels, sampleDl=sampleDl, verbose=verbose)
    else:
        return cpp_subsampling.compute(points, features=features, classes=labels, sampleDl=sampleDl, verbose=verbose)


class HumanBodySegmentationDataset(Dataset):
    """
    HumanBodySegmentation dataset for segmentation task.
     Human Body with 14 parts (1~14) and background (0)
    """

    def __init__(self, input_threads=8):
        Dataset.__init__(self, 'HumanBodySegmentation')

        ###########################
        # Label to names
        ###########################
        self.label_to_names = {
            0: 'background',
            1: 'head',
            2: 'torso',
            3: 'right_upper_arm',
            4: 'left_upper_arm',
            5: 'right_forearm',
            6: 'left_forearm',
            7: 'right_hand',
            8: 'left_hand',
            9: 'right_thigh',
            10:'left_thigh',
            11:'right_shank',
            12:'left_shank',
            13:'right_foot',
            14:'left_foot'
        }

        self.init_labels()
        self.ignored_labels = np.array([])

        self.network_model = 'segmentation'
        self.path = 'Data/HumanBodySegmentationDataset'
        self.num_threads = input_threads

        return

    def load_subsampled_clouds(self, subsampling_parameter):
        """
        Load and subsample point clouds directly from .npy files
        """

        if 0 < subsampling_parameter <= 0.01:
            raise ValueError('subsampling_parameter too low (should be over 1 cm)')

        self.input_points = {'training': [], 'validation': [], 'test': []}
        self.input_labels = {'training': [], 'validation': [], 'test': []}
        self.input_point_labels = {'training': [], 'validation': [], 'test': []}

        # Helper function to load a split
        def load_split(split):
            t0 = time.time()
            print(f'\nLoading {split} points')

            filename = join(self.path, f'{split}_{subsampling_parameter:.3f}_record.pkl')
            if exists(filename):
                with open(filename, 'rb') as file:
                    self.input_labels[split], self.input_points[split], self.input_point_labels[split] = pickle.load(file)
            else:
                split_path = join(self.path, split)
                # Find all _features.npy files
                feature_files = [f for f in listdir(split_path) if f.endswith('_features.npy')]
                feature_files.sort()

                temp_points = []
                temp_labels = []
                input_labels_split = []

                for ff in feature_files:
                    prefix = ff.replace('_features.npy', '')
                    feat_path = join(split_path, ff)
                    lbl_path = join(split_path, prefix+'_labels.npy')

                    points = np.load(feat_path).astype(np.float32)  # Nx3
                    point_labels = np.load(lbl_path).astype(np.int32) # N

                    # Subsampling
                    if subsampling_parameter > 0:
                        sub_points, sub_labels = grid_subsampling(points, labels=point_labels, sampleDl=subsampling_parameter)
                        temp_points.append(sub_points)
                        temp_labels.append(sub_labels)
                    else:
                        temp_points.append(points)
                        temp_labels.append(point_labels)

                    input_labels_split.append(0)

                self.input_points[split] = temp_points
                self.input_point_labels[split] = temp_labels
                self.input_labels[split] = np.array(input_labels_split, dtype=np.int32)

                with open(filename, 'wb') as file:
                    pickle.dump((self.input_labels[split],
                                 self.input_points[split],
                                 self.input_point_labels[split]), file)

            lengths = [p.shape[0] for p in self.input_points[split]]
            sizes = [l * 4 * 3 for l in lengths]
            print('{:.1f} MB loaded in {:.1f}s'.format(np.sum(sizes) * 1e-6, time.time() - t0))

        # Load training/validation/test
        load_split('train')
        load_split('val')
        load_split('test')

        # Setup num_train and num_test
        self.num_train = len(self.input_points['train'])
        self.num_test = len(self.input_points['test'])

        # For consistency with original code:
        # rename 'val' to 'validation'
        self.input_points['validation'] = self.input_points['val']
        self.input_labels['validation'] = self.input_labels['val']
        self.input_point_labels['validation'] = self.input_point_labels['val']
        del self.input_points['val']
        del self.input_labels['val']
        del self.input_point_labels['val']

        return

    def get_batch_gen(self, split, config):

        def variable_batch_gen_segment():
            tp_list = []
            tpl_list = []
            ti_list = []
            batch_n = 0

            if split == 'training':
                gen_indices = np.random.permutation(self.num_train)
            elif split == 'validation':
                gen_indices = np.random.permutation(self.num_test)
            elif split == 'test':
                gen_indices = np.arange(self.num_test)
            else:
                raise ValueError('Split argument should be "training", "validation" or "test"')

            for i, rand_i in enumerate(gen_indices):
                new_points = self.input_points[split][rand_i].astype(np.float32)
                n = new_points.shape[0]

                # In case batch is full
                if batch_n + n > self.batch_limit:
                    yield (np.concatenate(tp_list, axis=0),
                           np.concatenate(tpl_list, axis=0),
                           np.array(ti_list, dtype=np.int32),
                           np.array([tp.shape[0] for tp in tp_list]))
                    tp_list = []
                    tpl_list = []
                    ti_list = []
                    batch_n = 0

                tp_list.append(new_points)
                tpl_list.append(self.input_point_labels[split][rand_i])
                ti_list.append(rand_i)
                batch_n += n

            if tp_list:
                yield (np.concatenate(tp_list, axis=0),
                       np.concatenate(tpl_list, axis=0),
                       np.array(ti_list, dtype=np.int32),
                       np.array([tp.shape[0] for tp in tp_list]))

        gen_types = (tf.float32, tf.int32, tf.int32, tf.int32)
        gen_shapes = ([None, 3], [None], [None], [None])
        return variable_batch_gen_segment, gen_types, gen_shapes

    def get_tf_mapping(self, config):

        def tf_map_segment(stacked_points, point_labels, obj_inds, stack_lengths):
            # Get batch indices for each point
            batch_inds = self.tf_get_batch_inds(stack_lengths)

            stacked_points, scales, rots = self.tf_augment_input(stacked_points,
                                                                 batch_inds,
                                                                 config)

            # Features
            stacked_features = tf.ones((tf.shape(stacked_points)[0], 1), dtype=tf.float32)
            if config.in_features_dim == 4:
                stacked_features = tf.concat((stacked_features, stacked_points), axis=1)
            elif config.in_features_dim == 7:
                stacked_features = tf.concat((stacked_features, stacked_points, tf.square(stacked_points)), axis=1)
            elif config.in_features_dim != 1:
                raise ValueError('Only accepted input dimensions are 1,4,7')

            input_list = self.tf_segmentation_inputs(config,
                                                     stacked_points,
                                                     stacked_features,
                                                     point_labels,
                                                     stack_lengths,
                                                     batch_inds)
            input_list += [scales, rots, obj_inds]

            return input_list

        return tf_map_segment
