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

received_sighup = False
received_sigint = False


def handle_sighup(signum, frame):
    print(f'signal {signum} received')
    global received_sighup
    received_sighup = True


def handle_sigint(signum, frame):
    '''Save model snapshot on Ctrl-C before exiting'''
    alarm = 3
    print(f'signal {signum} received, setting alarm to {alarm} seconds')
    global received_sigint
    if not received_sigint:
        signal.alarm(alarm)
    received_sigint = True


def worker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()  # Pipe's Connection, unused for duplex=True Pipe()
    env = env_fn_wrapper.x()
    frame = 0
    last_nz_reward = 0
    while True:
        cmd, data = remote.recv()
        if cmd == 'step':
            ob, reward, done, info = env.step(data)
            if done:
                ob = reset_env(env)
            if reward != 0:
                last_nz_reward = frame
            elif frame - last_nz_reward > 2000:
                reward = -0.001
            remote.send((ob, reward, done, info))
            frame += 1
        elif cmd == 'reset':
            ob = reset_env(env)
            remote.send(ob)
        elif cmd == 'reset_task':
            ob = env.reset_task()
            remote.send(ob)
        elif cmd == 'close':
            remote.close()
            break
        elif cmd == 'get_spaces':
            remote.send((env.observation_space, env.action_space))
        else:
            raise NotImplementedError


class VecEnv(object):
    """
    An abstract asynchronous, vectorized environment.
    """
    def __init__(self, num_envs, observation_space, action_space):
        self.num_envs = num_envs
        self.observation_space = observation_space
        self.action_space = action_space

    def reset(self):
        """
        Reset all the environments and return an array of
        observations, or a tuple of observation arrays.
        If step_async is still doing work, that work will
        be cancelled and step_wait() should not be called
        until step_async() is invoked again.
        """
        pass

    def step_async(self, actions):
        """
        Tell all the environments to start taking a step
        with the given actions.
        Call step_wait() to get the results of the step.
        You should not call this if a step_async run is
        already pending.
        """
        pass

    def step_wait(self):
        """
        Wait for the step taken with step_async().
        Returns (obs, rews, dones, infos):
         - obs: an array of observations, or a tuple of
                arrays of observations.
         - rews: an array of rewards
         - dones: an array of "episode done" booleans
         - infos: a sequence of info objects
        """
        pass

    def close(self):
        """
        Clean up the environments' resources.
        """
        pass

    def step(self, actions):
        self.step_async(actions)
        return self.step_wait()


class CloudpickleWrapper(object):
    """
    Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)
    """
    def __init__(self, x):
        self.x = x

    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.x)

    def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)


class SubprocVecEnv(VecEnv):
    def __init__(self, env_fns, spaces=None):
        """
        envs: list of gym environments to run in subprocesses
        """
        self.waiting = False
        self.closed = False
        nenvs = len(env_fns)
        self.nenvs = nenvs
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])
        self.ps = [
            Process(target=worker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
            for (work_remote, remote, env_fn)
            in zip(self.work_remotes, self.remotes, env_fns)
        ]
        for p in self.ps:
            p.daemon = True  # if the main process crashes, we should not cause things to hang
            p.start()
        for remote in self.work_remotes:
            remote.close()

        self.remotes[0].send(('get_spaces', None))
        observation_space, action_space = self.remotes[0].recv()
        VecEnv.__init__(self, len(env_fns), observation_space, action_space)

    def step_async(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        return np.stack(obs), np.stack(rews), np.stack(dones), infos

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def reset_task(self):
        for remote in self.remotes:
            remote.send(('reset_task', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()
            self.closed = True

    def __len__(self):
        return self.nenvs


def parse_args():
    parser = argparse.ArgumentParser(description=APP_NAME)
    parser.add_argument('--gpu', dest='gpu', nargs='+', default=[0],
                        help='GPU to use')
    parser.add_argument('--cpu', dest='gpu', action='store_const', const=[],
                        help='use CPU')
    parser.add_argument('--use-snapshot', action='store_true',
                        help='use model snapshot if available')
    parser.add_argument('--num-steps', default=16, type=int,
                        help='steps per update')
    parser.add_argument('--num-envs', default=16, type=int,
                        help='number of concurrent environments')

    return parser.parse_args()


def plot_state(msg):
    plot.figure(figsize=(15, 5))

    msg = f'{APP_NAME}, {msg}, mean last 100 reward={rewards_mean}'
    if msg:
        plot.gcf().suptitle(msg)

    messages = [
        f'frame={frame_idx}',
    ]
    plot.subplot(131)
    plot.title(f'{ENV_NAME} rewards ({", ".join(messages)})')
    plot.plot(np.array(all_rewards)[:-1])

    plot.subplot(132)
    plot.title('losses')
    plot.plot(losses)

    plot.subplot(133)
    plot.title('random screen')
    screen = np.array(random_screen, dtype=np.float32)
    screen = screen[-3:].transpose(1, 2, 0)
    plot.imshow(screen / 255)

    plot.savefig(f'state-{APP_NAME}.png', format='png')
    plot.close()


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


def make_env(name):
    def f():
        env = gym.make(ENV_NAME + ENV_SUFFIX)
        env = common.wrap_env(env, resize=True, pytorch_layout=True, stack_frames=frames_per_state)
        return env
    return f


def compute_returns(next_value, rewards, masks, gamma=0.99):
    R = next_value
    returns = []
    for step in reversed(range(len(rewards))):
        R = rewards[step] + gamma * R * masks[step]
        returns.insert(0, R)
    return returns


def eval_test_env(env):
    state = reset_env(env)
    done = False
    total_reward = 0
    while not done:
        state = t.FloatTensor(np.array(state)).unsqueeze(0).to(device)
        dist, _ = models(state)
        next_state, reward, done, _ = env.step(dist.sample().cpu().numpy()[0])
        state = next_state
        total_reward += reward
    return total_reward


def reset_env(env):
    state = env.reset()
    if 'FIRE' in env.unwrapped.get_action_meanings():
        state, _, _, _ = env.step(env.unwrapped.get_action_meanings().index('FIRE'))
    return state


if '__main__' == __name__:
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    signal.signal(signal.SIGINT, handle_sigint)

    args = parse_args()
    print(args)
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(x) for x in args.gpu)
    # t.multiprocessing.set_start_method('spawn')

    USE_CUDA = t.cuda.is_available()
    device = t.device('cuda' if USE_CUDA else 'cpu')
    if USE_CUDA:
        print(f'Using CUDA device: {t.cuda.get_device_name(device)}')

    frames_per_state = 4
    envs = [make_env(ENV_NAME + ENV_SUFFIX) for _ in range(args.num_envs)]
    envs = SubprocVecEnv(envs)
    test_env = make_env(ENV_NAME + ENV_SUFFIX)()
    models = build_model(envs, frames_per_state).to(device)
    opt = t.optim.Adam(models.parameters(), lr=3e-4)
    loss_fn = t.nn.SmoothL1Loss()

    signal.signal(signal.SIGHUP, handle_sighup)

    gamma = 0.99

    all_rewards = collections.deque(maxlen=1000)
    losses = collections.deque(maxlen=10000)
    rewards_mean = 0
    frame_idx = 0
    max_frames = int(1e8)

    snapshot_path = f'.model-snapshot-{APP_NAME}.pkl.gz'
    if args.use_snapshot:
        try:
            with gzip.open(snapshot_path, 'rb') as f:
                s = t.load(f)
                models.load_state_dict(s['models'])
                opt.load_state_dict(s['opt'])
                frame_idx = s['frame_idx']
                del s
            print(f'loaded snapshot from {snapshot_path}')
        except Exception as e:
            print(f'didn\'t load snapshot from {snapshot_path}')

    state = envs.reset()
    random_screen = state[-1]

    frames_tqdm = tqdm.trange(frame_idx, max_frames)
    while frame_idx < max_frames:
        entropy = 0
        log_probs = []
        values = []
        rewards = []
        masks = []

        for idx in range(args.num_steps):
            state = t.FloatTensor(state).to(device)
            dist, value = models(state)

            action = dist.sample()
            next_state, reward, done, _ = envs.step(action.cpu().numpy())

            log_prob = dist.log_prob(action)
            entropy += dist.entropy().mean()

            log_probs.append(log_prob.unsqueeze(1))
            values.append(value)
            rewards.append(t.FloatTensor(reward).unsqueeze(1).to(device))
            masks.append(t.FloatTensor(1 - done).unsqueeze(1).to(device))

            state = next_state
            frame_idx += 1

            if random.random() < 0.01:
                random_screen = state[-1]

            if received_sigint:
                break

        if received_sigint or (len(all_rewards) + 1) % 255 == 0:
            with gzip.open(snapshot_path, 'wb') as f:
                t.save({
                    'models': models.state_dict(),
                    'opt': opt.state_dict(),
                    'frame_idx': frame_idx,
                }, f, pickle_protocol=4)
            print(f'saved snapshot from {snapshot_path}')
            if received_sigint:
                exit(1)

        next_state = t.FloatTensor(next_state).to(device)
        _, next_value = models(next_state)
        returns = compute_returns(next_value, rewards, masks)

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

        losses.append(loss.item())

        frames_tqdm.update(args.num_steps)

        if received_sighup:
            plot_state('in progress')
            received_sighup = False

        if (frame_idx // args.num_steps) % 100 == 0:
            all_rewards.append(eval_test_env(test_env))
            rewards_mean = np.array(all_rewards)[-100:].mean()
            frames_tqdm.set_description_str('%s: mean rewards = %g' % (APP_NAME, rewards_mean))

        if len(all_rewards) > 100 and rewards_mean > 18:
            print('solved')
            with gzip.open(f'model-{APP_NAME}.pkl.gz', 'wb') as f:
                t.save(models.state_dict(), f, pickle_protocol=4)
            break

    plot_state('finished.')
    if os.path.exists(snapshot_path):
        os.remove(snapshot_path)
    print('done.')
