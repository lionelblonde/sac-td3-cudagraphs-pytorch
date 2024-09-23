from beartype import beartype
import torch


class ActionNoise(object):

    @beartype
    def reset(self):  # exists even if useless for non-temporally correlated noise
        pass


