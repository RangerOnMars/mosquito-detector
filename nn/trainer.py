import yaml
from .shufflenetv2 import ShuffleNetModified


class Trainer:
    def __init__(self, file):
        args = []
        with open(file, "r"):
            args = yaml
        self.args = args
        self.act = args.act
        self.model = ShuffleNetModified(act=self.act)