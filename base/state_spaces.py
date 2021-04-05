from abc import abstractmethod
import random


class StateSpaceBase:

    def __init__(self) -> None:
        super().__init__()
        self.keys = []
        for state in self.generate_states():
            self.keys.append(state)
        random.seed(0)
        random.shuffle(self.keys)

    def __iter__(self):
        return self.keys.__iter__()

    def __getitem__(self, item):
        return self.keys[item]

    @abstractmethod
    def generate_states(self):
        pass

