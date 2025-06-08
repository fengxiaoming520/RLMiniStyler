import random
import numpy as np


class ReplayMemory:
    def __init__(self, capacity, seed):
        random.seed(seed)
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state_moving, style, action, reward, next_state_moving, mask):
        if len(self.buffer) < self.capacity:  # First reset the target position with 'None',
            self.buffer.append(None)  # and then put in the record
        self.buffer[self.position] = (state_moving, style, action, reward, next_state_moving, mask)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state_moving, style, action, reward, next_state_moving, mask = map(np.stack, zip(*batch))
        return state_moving, style, action, reward, next_state_moving, mask

    def __len__(self):
        return len(self.buffer)
