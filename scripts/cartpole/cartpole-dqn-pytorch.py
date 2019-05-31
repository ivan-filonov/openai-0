#!/usr/bin/env python3
import argparse
import collections
import gym
import matplotlib.pyplot as plot
import numpy as np
import os
import random
import signal
import sys
import torch as t
import tqdm


sys.path.append(sys.path[0] + "/..")
import common

ENV_NAME = 'CartPole'
APP_NAME = ENV_NAME.lower() + '-dqn-pytorch'
received_sighup = False


def handle_sighup(signum, frame):
    print(f'signal {signum} received')
    global received_sighup
    received_sighup = True


def parse_args():
    parser = argparse.ArgumentParser(description=APP_NAME)
    parser.add_argument('--gpu', dest='gpu', nargs='+', default=[0],
                        help='GPU to use')
    parser.add_argument('--cpu', dest='gpu', action='store_const', const=[],
                        help='use CPU')

    return parser.parse_args()


def plot_state(msg):
    plot.figure(figsize=(15, 5))

    msg += f', mean last 100 reward={rewards_mean}'
    if msg:
        plot.gcf().suptitle(msg)
    messages = [
        f'episode={episode}',
        f'frame={frame}',
    ]
    plot.subplot(121)
    plot.title(f'{ENV_NAME} rewards ({", ".join(messages)})')
    plot.plot(all_rewards)

    plot.subplot(122)
    plot.title('losses')
    plot.plot(losses)

    plot.savefig(f'state-{APP_NAME}.png', format='png')
    plot.close()


class Replay(object):
    def __init__(self, maxlen):
        self.memory = collections.deque(maxlen=maxlen)

    def add(self, state, action, next_state, reward, done):
        state = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)
        self.memory.append((state, action, next_state, reward, done))

    def sample(self, n):
        state, action, next_state, reward, masks = zip(*random.sample(self.memory, n))
        state = np.concatenate(state, axis=0)
        next_state = np.concatenate(next_state, axis=0)
        reward = np.array(reward)
        masks = np.array(masks)

        state = t.FloatTensor(state).to(device)
        action = t.LongTensor(action).to(device)
        next_state = t.FloatTensor(next_state).to(device)
        reward = t.FloatTensor(reward).to(device)
        masks = t.FloatTensor(masks).to(device)

        return state, action, next_state, reward, masks

    def __len__(self):
        return len(self.memory)


class Model(t.nn.Module):
    def __init__(self, n_in, n_out, hidden):
        super().__init__()
        self.seq = t.nn.Sequential(
            t.nn.Linear(n_in, hidden),
            t.nn.ReLU(),
            t.nn.Linear(hidden, n_out)
        )

    def forward(self, x):
        return self.seq(x)


def update(model, target_model, opt, loss_fn, replay, batch_size):
    states, actions, next_states, rewards, masks = replay.sample(batch_size)

    q_value = model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
    q_next_value = target_model(next_states).max(1).values

    target = rewards + gamma * (1 - masks) * q_next_value
    loss = loss_fn(q_value, target.detach())

    opt.zero_grad()
    loss.backward()
    opt.step()

    return loss.item()


if '__main__' == __name__:
    signal.signal(signal.SIGHUP, handle_sighup)

    args = parse_args()
    print(args)
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(x) for x in args.gpu)

    USE_CUDA = t.cuda.is_available()
    device = t.device('cuda' if USE_CUDA else 'cpu')
    if USE_CUDA:
        print(f'Using CUDA device: {t.cuda.get_device_name(device)}')

    env = gym.make(ENV_NAME + '-v1')
    model = Model(
        env.observation_space.shape[0],
        env.action_space.n,
        128
    ).to(device)
    opt = t.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = t.nn.MSELoss()

    eps, eps_final = 1, 0.02
    eps_steps = 1000
    gamma = 0.99
    replay = Replay(10000)
    batch_size = 32

    all_rewards = collections.deque(maxlen=250)
    losses = collections.deque(maxlen=10000)
    rewards_mean = 0
    frame = 0
    eps_decay = np.exp(np.log(eps_final / eps) / eps_steps)

    episodes_tqdm = tqdm.trange(10000)
    for episode in episodes_tqdm:
        state, done = env.reset(), False
        all_rewards.append(0)

        while not done:
            frame += 1
            eps = max(eps_final, eps * eps_decay)
            if random.random() > eps:
                action = model(t.FloatTensor(state).to(device)).argmax().item()
            else:
                action = env.action_space.sample()
            next_state, reward, done, info = env.step(action)
            replay.add(state, action, next_state, reward, done)
            all_rewards[-1] += reward
            state = next_state

            if len(replay) > 1000:
                loss = update(model, model, opt, loss_fn, replay, batch_size)
                losses.append(loss)

            if received_sighup:
                plot_state('in progress')
                received_sighup = False

        rewards_mean = np.array(all_rewards)[-100:].mean()
        episodes_tqdm.set_description_str('mean rewards = %g' % rewards_mean)
        if len(all_rewards) > 100 and rewards_mean > 495:
            print('solved.')
            t.save(model.state_dict(), f'model-{APP_NAME}.pkl', pickle_protocol=4)
            break

    plot_state('finished.')
    print('done.')
