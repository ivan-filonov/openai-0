#!/usr/bin/env python3
import argparse
import collections
import gym
import matplotlib.pyplot as plot
import numpy as np
import os
import signal
import sys
import torch as t
from tqdm import tqdm


# pylint: disable=wrong-import-position
sys.path.append(sys.path[0] + "/..")
import common
# pylint: enable=wrong-import-position

ENV_NAME = 'CartPole'
APP_NAME = ENV_NAME.lower() + '-a2c-pytorch'
received_sighup = False


def handle_sighup(signum, frame):
    print(f'signal {signum} received')
    global received_sighup
    received_sighup = True


def parse_args():
    parser = argparse.ArgumentParser(description='simple a2c')
    parser.add_argument('--gpu', dest='gpu', nargs='+', default=[0],
                        help='GPU to use')
    parser.add_argument('--cpu', dest='gpu', action='store_const', const=[],
                        help='use CPU')
    parser.add_argument('--hidden-size', default=256, type=int,
                        help='hidden layer size')
    parser.add_argument('--num-steps', default=5, type=int,
                        help='number of steps per update')
    parser.add_argument('--lrate', default=3e-4, type=float,
                        help='learning rate')

    return parser.parse_args()


def plot_state():
    plot.figure(figsize=(15, 5))

    messages = [
        f'episode={episode}',
        f'frame={frame}',
        f'mean last 100 reward={rewards_mean}'
    ]
    plot.title(f'{ENV_NAME} rewards ({", ".join(messages)})')
    plot.plot(all_rewards)

    plot.savefig(f'state-{APP_NAME}.png', format='png')
    plot.close()


class Models(t.nn.Module):
    def __init__(self, n_in, n_out, hidden):
        super().__init__()
        # state -> expected value
        self.critic = t.nn.Sequential(
            t.nn.Linear(n_in, hidden),
            t.nn.ReLU(),
            t.nn.Linear(hidden, 1)
        )
        # state -> action weights
        self.actor = t.nn.Sequential(
            t.nn.Linear(n_in, hidden),
            t.nn.ReLU(),
            t.nn.Linear(hidden, n_out),
            t.nn.Softmax(dim=1)
        )

    def forward(self, x):
        value = self.critic(x)
        probs = self.actor(x)
        dist = t.distributions.Categorical(probs)
        return dist, value


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
    models = Models(
        env.observation_space.shape[0],
        env.action_space.n,
        args.hidden_size
    )
    models = models.to(device)
    opt = t.optim.Adam(models.parameters(), lr=args.lrate)

    gamma = 0.99
    all_rewards = collections.deque(maxlen=250)
    rewards_mean = 0
    frame = 0
    for episode in tqdm(range(10000)):
        state, done = env.reset(), False
        all_rewards.append(0)

        while not done:
            frame += 1
            entropy = 0
            log_probs, values, rewards, masks = [], [], [], []
            for _ in range(args.num_steps):
                state = t.FloatTensor(state).unsqueeze(0).to(device)
                dist, value = models(state)
                action = dist.sample()
                state, reward, done, info = env.step(action.item())

                entropy += dist.entropy().mean()
                log_probs.append(dist.log_prob(action))
                values.append(value)
                rewards.append(t.FloatTensor([reward]).to(device))
                masks.append(t.FloatTensor([1 - done]).to(device))

                all_rewards[-1] += reward

            next_state = t.FloatTensor(state).unsqueeze(0).to(device)
            _, next_value = models(next_state)
            returns, R = [], next_value
            for idx in reversed(range(len(rewards))):
                R = rewards[idx] + gamma * R * masks[idx]
                returns.append(R)
            returns = list(reversed(returns))

            log_probs = t.cat(log_probs)
            returns = t.cat(returns).detach()
            values = t.cat(values)

            advantage = returns - values

            actor_loss = -(log_probs * advantage.detach()).mean()
            critic_loss = advantage.pow(2).mean()

            loss = actor_loss + 0.5 * critic_loss - 0.001 * entropy

            opt.zero_grad()
            loss.backward()
            opt.step()

            if received_sighup:
                plot_state()
                received_sighup = False

        rewards_mean = np.array(all_rewards)[-100:].mean()
        if len(all_rewards) > 100 and rewards_mean > 495:
            print('solved.')
            t.save(models.state_dict(), f'model-{APP_NAME}.pkl', pickle_protocol=4)
            break

    plot_state()
    print('done.')
