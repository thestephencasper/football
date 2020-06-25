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

"""Runs football_env on OpenAI's ppo2."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import multiprocessing
import os

from absl import app
from absl import flags
from baselines import logger
from baselines.bench import monitor
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.ppo2 import ppo2
import tensorflow.compat.v1 as tf
from gfootball.env import config
from gfootball.env import football_env
from gfootball.env import wrappers
from gfootball.env import _apply_output_wrappers
from gfootball.env import observation_preprocessing
from gfootball.examples.models import gfootball_impala_cnn


def create_multiagent_env(iprocess):

    left_player = 'ppo2_cnn:left_players=1,policy=gfootball_impala_cnn,checkpoint=/Users/stephen/Documents/football/checkpoints/11_vs_11_easy_stochastic_v2'
    right_player = 'agent:right_players=1,policy=gfootball_impala_cnn,checkpoint=/Users/stephen/Documents/football/checkpoints/11_vs_11_easy_stochastic_v2'
    players = [left_player, right_player]

    write_full_episode_dumps = False and (iprocess == 0)
    write_goal_dumps = False and (iprocess == 0)
    config_values = {'dump_full_episodes': write_full_episode_dumps,
                     'dump_scores': write_goal_dumps,
                     'players': players,
                     'level': '11_vs_11_easy_stochastic',
                     'tracesdir': '',  # logdir
                     'write_video': False}

    cfg = config.Config(config_values)
    env = football_env.FootballEnv(cfg)

    render = False and (iprocess == 0)
    if render:
        env.render()

    dump_frequency = 10 if render and iprocess == 0 else 0
    env = wrappers.PeriodicDumpWriter(env, dump_frequency)

    rewards = 'scoring,checkpoints'  # what to base rewards on
    representation = 'extracted'  # ['simple115v2'] what observations model gets
    channel_dimensions = (observation_preprocessing.SMM_WIDTH, observation_preprocessing.SMM_HEIGHT)
    apply_single_agent_wrappers = True
    stacked = True  # whether to get last 4 observations stacked or just last 1
    env = _apply_output_wrappers(env, rewards, representation, channel_dimensions,
                                 apply_single_agent_wrappers, stacked)

    env = monitor.Monitor(env, logger.get_dir() and os.path.join(logger.get_dir(), str(iprocess)))

    return env


def train(_):
    """Trains a PPO2 policy."""

    num_envs = 8  # number to run in parallel

    vec_env = SubprocVecEnv([
        (lambda _i=i: create_multiagent_env(_i))
        for i in range(num_envs)
    ], context=None)

    # Import tensorflow after we create environments. TF is not fork sake, and
    # we could be using TF as part of environment if one of the players is
    # controled by an already trained model.
    ncpu = multiprocessing.cpu_count()
    config = tf.ConfigProto(allow_soft_placement=True,
                            intra_op_parallelism_threads=ncpu,
                            inter_op_parallelism_threads=ncpu)
    config.gpu_options.allow_growth = True
    tf.Session(config=config).__enter__()

    ppo2.learn(network='gfootball_impala_cnn',
               total_timesteps=1e6 + 1,
               env=vec_env,
               seed=0,
               nsteps=128,
               nminibatches=8,
               noptepochs=2,
               max_grad_norm=0.64,
               gamma=0.993,
               ent_coef=0.003,
               lr=0.000343,
               log_interval=10,
               save_interval=10,
               cliprange=0.8,
               load_path='/Users/stephen/Documents/football/checkpoints/11_vs_11_easy_stochastic_v2')


if __name__ == '__main__':
    app.run(train)


# TODO: train right against left to get a baseline.
#  At what point should you stop training and consider the avdersary generated?

