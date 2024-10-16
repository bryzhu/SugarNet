import os
import torch
from utils.constants import mode

class Exp_Basic(object):
    def __init__(self, args, model_name):
        self.args = args
        self.device = self._acquire_device()
        self.name = model_name
        self.model_name = model_name.split('_')[0]
        self.model = self._build_model().to(self.device)
        if self.args.no_dim_extension:
          self.ext = 'noext'
        else:
          self.ext = args.embed_size
        if args.delta_forecast:
          self.delta = "delta"
        else:
          self.delta = "nodelta"
        self.mode = mode[args.mode]

    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
