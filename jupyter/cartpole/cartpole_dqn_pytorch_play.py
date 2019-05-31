import numpy as np
import random
import torch as t

def play(env, model, n_frames, eps_final, eps_steps, q_out):
    #print('started')
    eps, eps_decay = 1, np.exp(np.log(eps_final) / eps_steps)
    state, done = env.reset(), False
    test_state = t.FloatTensor(state).cuda()
    try:
        for frame in range(n_frames):
            if random.random() < eps:
                action = env.action_space.sample()
            else:
                action = model(t.FloatTensor(state).cuda()).argmax().item()
            #print('frame %d action is %d' % (frame, action))
            eps = max(eps_final, eps * eps_decay)
            next_state, reward, done, _ = env.step(action)
            test_value = model(test_state).max().item()
            q_out.put((state, action, next_state, reward, done, test_value), timeout=5)
            if done:
                state, done = env.reset(), False
            else:
                state = next_state
    finally:
        q_out.put(None)

class Model(t.nn.Module):
    def __init__(self, n_in, n_out, n_hid=16):
        super().__init__()
        self.seq = t.nn.Sequential(
            t.nn.Linear(n_in, n_hid),
            t.nn.PReLU(),
        )
        self.adv = t.nn.Linear(n_hid, n_out)
        self.val = t.nn.Linear(n_hid, 1)
    def forward(self, x):
        p = self.seq(x)
        adv = self.adv(p)
        val = self.val(p)
        return val + adv - adv.mean()

def build_model(env, device, n_hid):
    return Model(np.prod(env.observation_space.shape), env.action_space.n, n_hid).to(device)