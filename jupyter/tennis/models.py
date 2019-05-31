import torch as t

class DuelingCnnDqn(t.nn.Module):
    def __init__(self, input_shape, n_out):
        super().__init__()
        self.input_shape = input_shape
        self.cnn = t.nn.Sequential(
            t.nn.Conv2d(12, 64, kernel_size=8, stride=4),            t.nn.PReLU(),
            t.nn.Conv2d(64, 96, kernel_size=4, stride=2),            t.nn.PReLU(),
            t.nn.Conv2d(96, 128, kernel_size=3, stride=1),            t.nn.PReLU(),
        ) # -> 64 12 6
        self.adv = t.nn.Sequential(
            t.nn.Linear(self.feature_size(), 512),                   t.nn.PReLU(),
            t.nn.Linear(512, n_out)
        )
        self.val = t.nn.Sequential(
            t.nn.Linear(self.feature_size(), 256),                   t.nn.PReLU(),
            t.nn.Linear(256, 1)
        )

    def feature_size(self):
        return self.cnn(t.zeros(1, *self.input_shape)).view(1, -1).size(1)

    def forward(self, x):
        if len(x.shape) < 4:
            x = x.unsqueeze(0)
        p = self.cnn(x)
        p = p.view(p.size(0), -1)
        adv = self.adv(p)
        val = self.val(p)
        return val + adv - adv.mean()

    def model_type(self):
        return 'dueling-cnn-dqn'