"""Choose velocity commands without averaging them into the walker's dead zone."""

import torch
from torch.distributions import Categorical, Distribution, constraints


class VelocityDistribution(Distribution):
    """Three independent categorical choices, returned as normalized XY/yaw commands.

    The policy learns the probabilities. Evaluation selects the most likely
    command for each axis; it does not average opposite commands into a stop.
    No state-dependent controller or teacher action is included here.
    """

    arg_constraints = {"logits": constraints.real}
    support = constraints.independent(constraints.interval(-1, 1), 1)
    has_rsample = False

    def __init__(self, logits, grid, validate_args=False):
        self.grid = torch.as_tensor(grid, device=logits.device, dtype=logits.dtype)
        self.logits = logits
        self.choices = Categorical(
            logits=logits.reshape(*logits.shape[:-1], 3, len(grid)),
            validate_args=False,
        )
        super().__init__(logits.shape[:-1], torch.Size([3]), validate_args)

    def sample(self, sample_shape=()):
        return self.grid[self.choices.sample(sample_shape)]

    def log_prob(self, value):
        matches = value[..., None] == self.grid
        ids = matches.to(torch.int64).argmax(-1)
        log_prob = self.choices.log_prob(ids).sum(-1)
        return torch.where(matches.any(-1).all(-1), log_prob, -torch.inf)

    def entropy(self):
        return self.choices.entropy().sum(-1)

    @property
    def mean(self):
        return (self.choices.probs * self.grid).sum(-1)

    @property
    def mode(self):
        return self.grid[self.choices.logits.argmax(-1)]

    @property
    def deterministic_sample(self):
        return self.mode
