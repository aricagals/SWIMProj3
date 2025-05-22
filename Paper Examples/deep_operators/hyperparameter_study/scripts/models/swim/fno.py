from swimnetworks import FNO1D
from .base import BaseSwim


class FNO1DSwim(BaseSwim):
   def __init__(self, model_params, seed):
        self.predictor = FNO1D(n_blocks=model_params["n_layers"], 
            n_hidden_channels=model_params["n_hidden_channels"],
            n_modes=model_params["n_modes"],
            layer_width=model_params["layer_width"],
            activation="tanh",
            parameter_sampler="tanh",
            random_seed=seed,
            regularization_scale=model_params["regularization_scale"])