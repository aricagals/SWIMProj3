from .base import BaseModel, mean_rel_l2, mse
import torch
import torch.nn as nn


class FCNPredictor(nn.Module):
    def __init__(self, n_layers, layer_width, input_dim):
        super().__init__()
        widths = [input_dim] + [layer_width] * n_layers + [input_dim]
        layers = [nn.Linear(widths[0], widths[1], dtype=torch.float64)]
        for width_in, width_out in zip(widths[1:-1], widths[2:]):
            layers.append(nn.Tanh())
            layers.append(nn.Linear(width_in, width_out, dtype=torch.float64))
        self.fcn = nn.Sequential(*layers)

    def forward(self, u0_batch):
        return self.fcn(u0_batch)
    
    def __call__(self, batch):
        return self.forward(batch)

    def restore_data(self, output):
        return output


class FCN(BaseModel):
    def build_predictor(self, model_params):
        self.predictor = FCNPredictor(
            model_params["n_layers"],
            model_params["layer_width"],
            model_params["input_dim"],
        )

    def metric_fn(self, pred, true):
        return mean_rel_l2(pred, true)
    
    def loss_fn(self, pred, true):
        return mse(pred, true)
    
    def restore_data(self, output):
        return self.predictor.restore_data(output)
