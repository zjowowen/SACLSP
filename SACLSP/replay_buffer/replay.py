import random
from collections import deque

class ReplayBuffer:
    def __init__(self, cfg):
        self.capacity = int(cfg.capacity)
        self.buffer = deque(maxlen=int(cfg.capacity))

    def add_experience(self, experience):
        self.buffer.append(experience)

    def add_experiences(self, experiences):
        self.buffer.extend(experiences)

    def sample_batch(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, dones, next_states = zip(*batch)
        return states, actions, rewards, dones, next_states

    def get_all_data(self):
        if not self.empty():
            states, actions, rewards, dones, next_states = zip(*self.buffer)
            return states, actions, rewards, dones, next_states
        else:
            return None, None, None, None, None

    def is_full(self):
        return len(self.buffer) == self.capacity

    def size(self):
        return len(self.buffer)

    def clear(self):
        self.buffer.clear()

    def empty(self):
        return len(self.buffer) == 0

if __name__ == "__main__":
    cfg=dict(
        capacity=100,
    )
    from easydict import EasyDict
    cfg = EasyDict(cfg)
    replay_pool = ReplayBuffer(cfg)
    for i in range(100):
        replay_pool.add_experience((i, i, i, i, i))
    print(replay_pool.sample_batch(10))
    print(replay_pool.is_full())
    print(replay_pool.size())
    print(replay_pool.get_all_data())
    replay_pool.clear()
    print(replay_pool.is_full())
    print(replay_pool.size())
    print(replay_pool.get_all_data())

