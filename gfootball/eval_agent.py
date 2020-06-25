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
from absl import app
from absl import flags
from absl import logging
from gfootball.env import config
from gfootball.env import football_env
from gfootball.env import wrappers
from gfootball.examples.models import gfootball_impala_cnn


def main(_):

    left_player = 'ppo2_cnn:left_players=1,policy=gfootball_impala_cnn,checkpoint=/Users/stephen/Documents/football/checkpoints/11_vs_11_easy_stochastic_v2'
    right_player = 'ppo2_cnn:right_players=1,policy=gfootball_impala_cnn,checkpoint=/Users/stephen/Documents/football/checkpoints/11_vs_11_easy_stochastic_v2'
    players = [left_player, right_player]

    config_values = {'dump_full_episodes': False,
                     'dump_scores': False,
                     'players': players,
                     'level': '11_vs_11_easy_stochastic',
                     'tracesdir': '/Users/stephen/Documents/football/logs',  # logdir
                     'write_video': False}

    cfg = config.Config(config_values)
    env = football_env.FootballEnv(cfg)

    render = False
    if render:
        env.render()

    env.reset()

    dump_frequency = 3
    env = wrappers.PeriodicDumpWriter(env, dump_frequency)

    n_timesteps = int(2 * 3e3 + 1)  # 3k per episode

    right_agent_ep_scores = []
    ep_right_scores = 0.0

    for _ in range(n_timesteps):
        _, reward, done, _ = env.step([])
        ep_right_scores -= reward
        if done:
            right_agent_ep_scores.append(ep_right_scores)
            ep_right_scores = 0.0
            env.reset()

    mean_score = sum(right_agent_ep_scores) / len(right_agent_ep_scores)
    print(f'\n***\nRight agent episode scores: {right_agent_ep_scores}\n' +
          f'Right agent episode mean score: {mean_score}\n***\n')


if __name__ == '__main__':
    app.run(main)
