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


def main(_):

    left_player = 'ppo2_cnn:left_players=1,policy=gfootball_impala_cnn,checkpoint=/Users/stephen/Documents/football/checkpoints/11_vs_11_easy_stochastic_v2'
    right_player = 'ppo2_cnn:right_players=1,policy=gfootball_impala_cnn,checkpoint=/Users/stephen/Documents/football/checkpoints/11_vs_11_easy_stochastic_v2'
    players = [left_player, right_player]

    env_config_values = {'dump_full_episodes': False,
                     'dump_scores': False,
                     'players': players,
                     'level': '11_vs_11_easy_stochastic',
                     'tracesdir': '/Users/stephen/Documents/football/logs',  # logdir
                     'write_video': False}

    env_config = config.Config(env_config_values)
    env = football_env.FootballEnv(env_config)
    env.reset()

    player_config = {'index': 2}
    name, definition = config.parse_player_definition(left_player)
    config_name = 'player_{}'.format(name)
    if config_name in player_config:
        player_config[config_name] += 1
    else:
        player_config[config_name] = 0
    player_config.update(definition)
    player_config['stacked'] = True
    player = Player(player_config, env_config)
    stacker = ObservationStacker(4)

    n_timesteps = 30000  # 10 games
    game_i = 0
    observations = []
    actions = []

    for i in range(n_timesteps):
        obs, _, done, _ = env.step([])
        obs_processed = observation_preprocessing.generate_smm([obs])
        obs_processed = stacker.get(obs_processed)
        observations.append(obs_processed)
        act = player.take_action([obs])[0]
        actions.append(full_action_set.index(act))
        if done:
            env.reset()
            stacker.reset()
            observations = np.squeeze(np.vstack(observations))  # should not be shape (3000, 72, 96, 16)
            actions = np.array(actions)  # should be shape (n_samples,)
            with open(f'/Users/stephen/Documents/football/data/observations{game_i}.pkl', 'wb') as f:
                pickle.dump(observations, f)
            with open(f'/Users/stephen/Documents/football/data/actions{game_i}.pkl', 'wb') as f:
                pickle.dump(actions, f)
            game_i += 1
            observations = []
            actions = []

    print('Done :)')


if __name__ == '__main__':
    app.run(main)