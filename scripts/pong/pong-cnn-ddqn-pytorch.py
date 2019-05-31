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


sys.path.append(sys.path[0] + "/..")
import common

ENV_NAME = 'Pong'
ENV_SUFFIX = 'Deterministic-v4'
APP_NAME = ENV_NAME.lower() + '-cnn-ddqn-pytorch'

received_sighup = False
received_sigint = False


def handle_sighup(signum, frame):
    print(f'signal {signum} received')
    global received_sighup
    received_sighup = True


def handle_sigint(signum, frame):
    '''Save model snapshot on Ctrl-C before exiting'''
    print(f'signal {signum} received')
    global received_sigint
    received_sigint = True
    signal.alarm(3)


def parse_args():
    parser = argparse.ArgumentParser(description=APP_NAME)
    parser.add_argument('--gpu', dest='gpu', nargs='+', default=[0],
                        help='GPU to use')
    parser.add_argument('--cpu', dest='gpu', action='store_const', const=[],
                        help='use CPU')
    parser.add_argument('--use-snapshot', action='store_true',
                        help='use model snapshot if available')
    parser.add_argument('--steps', default=4, type=int,
                        help='train every n steps')

    return parser.parse_args()


def plot_state(msg):
    plot.figure(figsize=(15, 5))

    msg = f'{APP_NAME}, {msg}, mean last 100 reward={rewards_mean}'
    if msg:
        plot.gcf().suptitle(msg)

    messages = [
        f'episode={episode}',
        f'frame={frame}',
    ]
    plot.subplot(131)
    plot.title(f'{ENV_NAME} rewards ({", ".join(messages)})')
    plot.plot(np.array(all_rewards)[:-1])

    plot.subplot(132)
    plot.title('losses')
    plot.plot(losses)

    plot.subplot(133)
    plot.title('random screen')
    state, _, _, _, _ = replay.sample(1, device)
    state = state.squeeze(0)[-3:].permute(1, 2, 0)
    plot.imshow(state.cpu().numpy() / 255)

    plot.savefig(f'state-{APP_NAME}.png', format='png')
    plot.close()


class Model(t.nn.Module):
    def __init__(self, input_shape, input_frames, n_out):
        super().__init__()
        self.cnn = t.nn.Sequential(
            t.nn.Conv2d(3 * input_frames, 32, kernel_size=8, stride=4),
            t.nn.ReLU(),
            t.nn.Conv2d(32, 64, kernel_size=4, stride=2),
            t.nn.ReLU(),
            t.nn.Conv2d(64, 64, kernel_size=3, stride=1),
            t.nn.ReLU(),
        )
        cnn_fc = self.feature_size(self.cnn, input_shape)
        self.fc = t.nn.Sequential(
            t.nn.Linear(cnn_fc, 512),
            t.nn.ReLU(),
            t.nn.Linear(512, n_out)
        )

    def feature_size(self, cnn, shape):
        return cnn(t.zeros(1, *shape)).view(1, -1).size(1)

    def forward(self, x):
        if len(x.shape) < 4:
            x = x.unsqueeze(0)
        p = self.cnn(x)
        p = p.view(p.size(0), -1)
        return self.fc(p)


def build_model(frames_per_state):
    return Model(
        env.observation_space.shape,
        frames_per_state,
        env.action_space.n
    )


def update(model, target_model, opt, loss_fn, replay, batch_size):
    states, actions, next_states, rewards, masks = replay.sample(batch_size, device)

    q_values = model(states)
    q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

    q_next_values = target_model(next_states)
    if False:  # original paper use this
        q_next_actions = model(next_states).max(1).indices
        q_next_value = q_next_values.gather(1, q_next_actions.unsqueeze(1)).squeeze(1)
    else:
        q_next_value = q_next_values.max(1).values

    target = rewards + gamma * (1 - masks) * q_next_value
    loss = loss_fn(q_value, target.detach())

    opt.zero_grad()
    loss.backward()
    opt.step()

    return loss.item()


if '__main__' == __name__:
    signal.signal(signal.SIGHUP, handle_sighup)
    signal.signal(signal.SIGINT, handle_sigint)

    args = parse_args()
    print(args)
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(x) for x in args.gpu)

    USE_CUDA = t.cuda.is_available()
    device = t.device('cuda' if USE_CUDA else 'cpu')
    if USE_CUDA:
        print(f'Using CUDA device: {t.cuda.get_device_name(device)}')

    frames_per_state = 4
    env = gym.make(ENV_NAME + ENV_SUFFIX)
    env = common.wrap_env(env, resize=True, pytorch_layout=True, stack_frames=frames_per_state)
    model = build_model(frames_per_state).to(device)
    target_model = build_model(frames_per_state).to(device)
    opt = t.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = t.nn.SmoothL1Loss()

    eps, eps_final = 1, 0.01
    eps_steps = 50000
    gamma = 0.99
    replay = common.ReplayCnnBuffer(100000)
    batch_size = 32

    all_rewards = collections.deque(maxlen=250)
    losses = collections.deque(maxlen=10000)
    rewards_mean = 0
    frame = 0
    eps_decay = np.exp(np.log(eps_final / eps) / eps_steps)

    snapshot_path = f'.model-snapshot-{APP_NAME}.pkl.gz'
    if args.use_snapshot:
        try:
            with gzip.open(snapshot_path, 'rb') as f:
                s = t.load(f)
                model.load_state_dict(s['model'])
                target_model.load_state_dict(s['model'])
                opt.load_state_dict(s['opt'])
                del s
            print(f'loaded snapshot from {snapshot_path}')
        except Exception as e:
            print(f'didn\'t load snapshot from {snapshot_path}')

    episodes_tqdm = tqdm.trange(10000)
    for episode in episodes_tqdm:
        state, done = env.reset(), False
        all_rewards.append(0)

        while not done:
            frame += 1
            eps = max(eps_final, eps * eps_decay)
            if random.random() > eps:
                model.eval()
                s = t.FloatTensor(np.array(state)).to(device)
                action = model(s).argmax().item()
            else:
                action = env.action_space.sample()
            next_state, reward, done, info = env.step(action)
            replay.add(state, action, next_state, reward, done)
            all_rewards[-1] += reward
            state = next_state

            if len(replay) > batch_size * args.steps and frame % args.steps == 0:
                model.train()
                loss = update(model, target_model, opt, loss_fn, replay, batch_size * args.steps)
                losses.append(loss)

                if frame % 1000 == 0:
                    target_model.load_state_dict(model.state_dict())

            if received_sighup:
                plot_state('in progress')
                received_sighup = False

            if received_sigint or (frame % 100000 == 0):
                plot_state('saving snapshot')
                with gzip.open(snapshot_path, 'wb') as f:
                    t.save({
                        'model': model.state_dict(),
                        'opt': opt.state_dict()
                    }, f, pickle_protocol=4)
                episodes_tqdm.write(f'saved snapshot {snapshot_path}')
                if received_sigint:
                    print('now exiting')
                    exit(1)

        rewards_mean = np.array(all_rewards)[-100:].mean()
        episodes_tqdm.set_description_str('%s: mean rewards = %g' % (APP_NAME, rewards_mean))
        if len(all_rewards) > 100 and rewards_mean > 18:
            print('solved')
            with gzip.open(f'model-{APP_NAME}.pkl.gz', 'wb') as f:
                t.save(model.state_dict(), f, pickle_protocol=4)
            break

    plot_state('finished.')
    if os.path.exists(snapshot_path):
        os.remove(snapshot_path)
    print('done.')
