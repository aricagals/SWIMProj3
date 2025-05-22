from typing import Any
from .base import BaseModel, mean_rel_l2, mse
import torch
import torch.nn as nn


def crop_rfft_modes(x, ks):
    # Works only for 1d!
    assert len(ks) == 1
    return x[..., : ks[0]]


def pad_rfft_modes(x, target_lengths):
    # Works only for 1d!
    assert len(target_lengths) == 1
    delta = target_lengths[0] - x.shape[1]
    padding = torch.zeros((x.shape[0], delta), dtype=x.dtype)
    padding = padding.to(x.device)
    return torch.concat([x, padding], dim=-1)


def rfft(signal, n_modes, norm="backward"):
    """Comptes RFFT along the axes."""
    shift = signal.ndim - len(n_modes)
    transformed = torch.fft.rfftn(signal, s=signal.shape[shift:], norm=norm)
    cropped = crop_rfft_modes(transformed, n_modes)
    return cropped


def irfft(modes, target_lengths, norm="backward"):
    """Comptes inverse RFFT along the axes."""
    padded = pad_rfft_modes(modes, target_lengths)
    return torch.fft.irfftn(padded, s=target_lengths, norm=norm)


class InFourierPredictor(nn.Module):
    # Works only for 1d!
    def __init__(self, n_layers, layer_width, n_modes, input_dim):
        super().__init__()
        # We split complex coefficients into the real and the imaginary parts
        widths = [2 * n_modes] + [layer_width] * n_layers + [2 * n_modes]
        layers = [nn.Linear(widths[0], widths[1], dtype=torch.float64)]
        for width_in, width_out in zip(widths[1:-1], widths[2:]):
            layers.append(nn.Tanh())
            layers.append(nn.Linear(width_in, width_out, dtype=torch.float64))
        self.fcn = nn.Sequential(*layers)
        self.n_modes = n_modes
        self.input_dim = input_dim

    def forward(self, u0_batch):
        real_fft_u0 = self.transform_data(u0_batch)
        fcn_output = self.fcn(real_fft_u0)
        return fcn_output
    
    def __call__(self, batch):
        return self.forward(batch)


    def transform_data(self, output):
        fft_out = rfft(output, [self.n_modes])
        real_fft_out = torch.view_as_real(fft_out)
        real_fft_out = real_fft_out.view(*fft_out.shape[:-1], -1)
        return real_fft_out

    def restore_data(self, output):
        output = output.view(*output.shape[:-1], -1, 2)
        complex_output = torch.view_as_complex(output)
        return irfft(complex_output, [self.input_dim])


class InFourier(BaseModel):
    def build_predictor(self, model_params):
        self.predictor = InFourierPredictor(
            model_params["n_layers"],
            model_params["layer_width"],
            model_params["n_modes"],
            model_params["input_dim"]
        )

    def metric_fn(self, pred, true):
        # Metric in signal space
        pred = self.restore_data(pred)
        return mean_rel_l2(pred, true)
    
    def loss_fn(self, pred, true):
        # Loss in Fourier space
        true = self.predictor.transform_data(true)
        return mse(pred, true)
    
    def restore_data(self, output):
        return self.predictor.restore_data(output)
