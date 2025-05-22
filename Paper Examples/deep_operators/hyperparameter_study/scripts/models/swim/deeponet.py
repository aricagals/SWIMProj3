import numpy as np
from sklearn.pipeline import Pipeline
from swimnetworks import Dense, Linear, DeepONetPOD


from .base import BaseSwim

class PODDeepONetSwim(BaseSwim):
    def __init__(self, model_params, seed):
        rng = np.random.default_rng(seed)
        fcn = [
            (
                f"fcn{i+1}",
                Dense(
                    layer_width=model_params["layer_width"],
                    activation=np.tanh,
                    parameter_sampler="tanh",
                    random_seed=rng.integers(np.iinfo(np.int64).max),
                ),
            )
            for i in range(model_params["n_layers"])
        ]
        linear = (
            "lin",
            Linear(regularization_scale=model_params["regularization_scale"]),
        )
        steps = fcn + [linear]
        pipeline = Pipeline(steps)
        self.predictor = DeepONetPOD(pipeline, model_params["n_modes"])