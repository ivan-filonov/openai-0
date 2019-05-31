#!/usr/bin/env python3
import argparse
import collections
import gym
import gzip
import matplotlib.pyplot as plot
import numpy as np
import os
import random
import signal
import sys
import torch as t
import tqdm
from torch.multiprocessing import Process, Pipe


sys.path.append(sys.path[0] + "/..")
import common

ENV_NAME = 'Tennis'
ENV_SUFFIX = 'Deterministic-v4'
APP_NAME = ENV_NAME.lower() + '-cnn-a2c-pytorch'


class Models(t.nn.Module):
    def __init__(self, input_shape, input_frames, n_out):
        super().__init__()
        self.cnn = t.nn.Sequential(
            t.nn.Conv2d(3 * input_frames, 32, kernel_size=8, stride=4),
            t.nn.PReLU(),
            t.nn.Conv2d(32, 64, kernel_size=4, stride=2),
            t.nn.PReLU(),
            t.nn.Conv2d(64, 64, kernel_size=3, stride=1),
            t.nn.PReLU(),
        )
        cnn_fc = self.feature_size(self.cnn, input_shape)
        # state -> expected value
        self.critic = t.nn.Sequential(
            t.nn.Linear(cnn_fc, 512),
            t.nn.PReLU(),
            t.nn.Linear(512, 1)
        )
        # state -> action weights
        self.actor = t.nn.Sequential(
            t.nn.Linear(cnn_fc, 512),
            t.nn.PReLU(),
            t.nn.Linear(512, n_out),
            t.nn.Softmax(dim=1)
        )

    def feature_size(self, cnn, shape):
        return cnn(t.zeros(1, *shape)).view(1, -1).size(1)

    def forward(self, x):
        if len(x.shape) < 4:
            x = x.unsqueeze(0)
        p = self.cnn(x)
        p = p.view(p.size(0), -1)
        value = self.critic(p)
        probs = self.actor(p)
        dist = t.distributions.Categorical(probs)
        return dist, value


def build_model(env, input_frames):
    return Models(
        env.observation_space.shape,
        input_frames,
        env.action_space.n
    )


def make_env(name, vid_dir='.'):
    def f():
        env = gym.make(ENV_NAME + ENV_SUFFIX)
        env = gym.wrappers.Monitor(env, vid_dir, force=True)
        env = common.wrap_env(env, resize=True, pytorch_layout=True, stack_frames=frames_per_state)
        return env
    return f


def eval_test_env(env):
    state = reset_env(env)
    done = False
    total_reward = 0
    frame = 0
    while not done:
        frame += 1
        state = t.FloatTensor(np.array(state)).unsqueeze(0).to(device)
        dist, _ = models(state)
        next_state, reward, done, _ = env.step(dist.sample().cpu().numpy()[0])
        state = next_state
        total_reward += reward
        print(f'\r{frame}', end='')
    print()
    return total_reward


def reset_env(env):
    state = env.reset()
    if 'FIRE' in env.unwrapped.get_action_meanings():
        state, _, _, _ = env.step(env.unwrapped.get_action_meanings().index('FIRE'))
    return state


if '__main__' == __name__:
    USE_CUDA = t.cuda.is_available()
    device = t.device('cuda' if USE_CUDA else 'cpu')
    if USE_CUDA:
        print(f'Using CUDA device: {t.cuda.get_device_name(device)}')

    frames_per_state = 4
    env = make_env(ENV_NAME + ENV_SUFFIX)()
    models = build_model(env, frames_per_state).to(device)

    snapshot_path = f'.model-snapshot-{APP_NAME}.pkl.gz'
    try:
        with gzip.open(snapshot_path, 'rb') as f:
            s = t.load(f)
            models.load_state_dict(s['models'])
            del s
        print(f'loaded snapshot from {snapshot_path}')
    except Exception as e:
        print(f'didn\'t load snapshot from {snapshot_path}')
        exit(1)

    with t.no_grad():
        eval_test_env(env)
        eval_test_env(env)
    env.close()

    print('done.')
