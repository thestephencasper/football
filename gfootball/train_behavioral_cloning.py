# coding=utf-8
# Copyright 2019 Google LLC
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Script allowing to play the game by multiple players."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
from baselines.bench import monitor
from baselines import logger
import numpy as np
import pickle
import tensorflow as tf
from tensorflow import Tensor
from tensorflow.keras.layers import Input, Conv2D, ReLU, BatchNormalization, Add, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Model
import sonnet as snt
from absl import app
from absl import flags
from absl import logging
from gfootball.env import config
from gfootball.env import football_env
from gfootball.env import wrappers
from gfootball.env import _apply_output_wrappers
from gfootball.env import _process_representation_wrappers
from gfootball.env import observation_preprocessing
from gfootball.examples.models import gfootball_impala_cnn
from gfootball.env.players.ppo2_cnn import Player, ObservationStacker
from gfootball.env.football_action_set import full_action_set
# tf.enable_eager_execution()

# https://www.tensorflow.org/tutorials/quickstart/advanced
# https://towardsdatascience.com/building-a-resnet-in-keras-e8f1322a49ba
# TODO: will unbalanced loading cause problems???


def relu_bn(inputs):
    relu = ReLU()(inputs)
    bn = BatchNormalization()(relu)
    return bn


def residual_block(x, filters, kernel_size=3):
    y = Conv2D(kernel_size=kernel_size, strides=1, filters=filters, padding="same")(x)
    y = relu_bn(y)
    y = Conv2D(kernel_size=kernel_size, strides=1, filters=filters, padding="same")(y)
    out = Add()([x, y])
    out = relu_bn(out)
    return out


def make_gfootball_resnet():

    filters_and_blocks = [(16, 2), (32, 2), (32, 2), (32, 2)]

    inputs = Input(shape=(observation_preprocessing.SMM_HEIGHT, observation_preprocessing.SMM_WIDTH, 16))
    x = inputs

    for n_filters, n_blocks in filters_and_blocks:
        x = Conv2D(kernel_size=3, strides=1, filters=n_filters, padding="same")(x)
        x = MaxPooling2D(pool_size=[3,3], strides=[2,2], padding='same')(x)
        x = relu_bn(x)
        for _ in range(n_blocks):
            x = residual_block(x, n_filters)

    x = ReLU()(x)
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    outputs = Dense(len(full_action_set), activation='softmax')(x)

    model = Model(inputs, outputs)

    return model


def train_on_games(model, start_game, end_game, n_epochs):

    observations = []
    actions = []
    for game_i in range(start_game, end_game):
        with open(f'/Users/stephen/Documents/football/data/observations{game_i}.pkl', 'rb') as f:
            game_obs = pickle.load(f)
            observations.append(game_obs)
        with open(f'/Users/stephen/Documents/football/data/actions{game_i}.pkl', 'rb') as f:
            game_actions = pickle.load(f)
            actions.append(game_actions)

    observations = np.concatenate(observations)
    actions = np.concatenate(actions)

    n_total = observations.shape[0]
    n_train = int(0.9 * n_total)
    n_test = n_total - n_train
    train_idxs = np.random.permutation(np.concatenate([np.ones((n_train,), dtype=int),
                                                       np.zeros((n_test,), dtype=int)]))
    train_idxs = train_idxs.astype(bool)

    train_X = observations[train_idxs]
    train_y = actions[train_idxs]
    test_X = observations[1-train_idxs]
    test_y = actions[1-train_idxs]

    loss_weights = 1 / np.bincount(train_y)
    inf_idxs = np.isinf(loss_weights)
    loss_weights[inf_idxs] = 1
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_X, train_y, batch_size=64, epochs=n_epochs, shuffle=True, validation_data=(test_X, test_y),
              class_weight=loss_weights)
        

def main(_):

    model = make_gfootball_resnet()
    train_on_games(model, 0, 5, 3)
    train_on_games(model, 5, 10, 3)

    model.save('/Users/stephen/Documents/football/models/behavioral_cloning.h5')


if __name__ == '__main__':
    app.run(main)
