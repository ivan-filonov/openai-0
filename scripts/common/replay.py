import numpy as np
import torch as t


class ReplayCnnBuffer(object):
    def __init__(self, maxlen):
        self.memory = np.empty(maxlen, dtype=np.object)
        self.index = 0
        self.count = 0

    def __len__(self):
        return self.count

    def add(self, state, action, next_state, reward, done):
        self.memory[self.index] = (state, action, next_state, reward, done)
        self.index = (self.index + 1) % len(self.memory)
        if self.count < len(self.memory):
            self.count += 1

    def sample(self, n, device):
        with t.no_grad():
            indices = np.random.randint(low=0, high=self.count, size=n)
            samples = zip(*list(self.memory[indices]))
            states, actions, next_states, rewards, masks = samples

            actions = t.LongTensor(actions).to(device)
            rewards = t.FloatTensor(rewards).to(device)
            masks = t.FloatTensor(masks).to(device)
            states = ReplayCnnBuffer.stack_states(states, device)
            next_states = ReplayCnnBuffer.stack_states(next_states, device)
            return states, actions, next_states, rewards, masks

    @staticmethod
    def stack_states(states, device):
        s = np.concatenate([np.expand_dims(x, 0) for x in states])
        return t.ByteTensor(s).to(device).float()
