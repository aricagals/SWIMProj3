from typing import Any
import torch
import numpy as np
import yaml
import sys
from neuralop import get_model
# from neuralop import Trainer
from ._fno_trainer import Trainer # use the modified Trainer with early stopping
from configmypy import ConfigPipeline, YamlConfig
from neuralop.datasets.tensor_dataset import TensorDataset
from neuralop.utils import count_params
from neuralop import LpLoss, H1Loss

from torch.nn.parallel import DistributedDataParallel as DDP

FOLDER = "scripts/models/adam"
DEFAULT_CONFIG_FILENAME = "_default_fno_config.yaml"


class FNO1D:
    def __init__(self, device, model_params, seed):
        reader = ConfigPipeline([YamlConfig(DEFAULT_CONFIG_FILENAME, config_name='default', config_folder=FOLDER), YamlConfig(config_folder=FOLDER)])
        self.config = reader.read_conf()
        self._update_model_config(model_params, seed)
        self.device = device

        self.trainer = None

        # Make sure the defualt device is 'cpu'
        # if code before set it up to a different value.
        torch.set_default_device("cpu")


    def _update_model_config(self, model_params, seed):
        model_config = self.config.tfno1d
        model_config.n_modes_height = model_params["n_modes"]
        model_config.hidden_channels = model_params["n_hidden_channels"]
        # Use the same number of channels inside the projection
        model_config.projection_channels = model_params["n_hidden_channels"]
        model_config.n_layers = model_params["n_layers"]
        self.config.distributed.seed = seed
    
    def _update_train_config(self, training_params):
        train_config = self.config.opt
        train_config.n_epochs = training_params["n_epochs"]
        train_config.patience = training_params["patience"]
        train_config.learning_rate = training_params["learning_rate"]
        train_config.weight_decay = training_params["weight_decay"]  
        self.config.data.batch_size = training_params["batch_size"]
        self.config.data.test_batch_sizes = training_params["batch_size"]


    def _get_dataloder(self, data, batch_size, grid=[0,1]):
        u0, u1 = data.astype(np.float32)
        u0, u1 = torch.from_numpy(u0), torch.from_numpy(u1).unsqueeze(1)
        s = u0.size(-1)
        grid = torch.linspace(grid[0], grid[1], s + 1)[0:-1].view(1,-1)
        grid = grid.repeat(len(u0), 1)
        u0 = torch.cat((u0.unsqueeze(1), grid.unsqueeze(1)), 1)
        loader = torch.utils.data.DataLoader(TensorDataset(u0, u1), batch_size=batch_size, shuffle=False)
        return loader 


    def train(self, train_data, val_data, training_params, _):
        self._update_train_config(training_params)
        config = self.config
        device, is_logger = self.device, True

        # The following code is taken from https://github.com/neuraloperator/neuraloperator

        # Make sure we only print information when needed
        config.verbose = config.verbose and is_logger

        # Loading the Burgers' dataset in 256 resolution
        train_loader = self._get_dataloder(train_data, config.data.batch_size)
        val_loader = self._get_dataloder(val_data, config.data.test_batch_sizes)

        model = get_model(config)
        model = model.to(device)

        #Use distributed data parallel 
        if config.distributed.use_distributed:
            model = DDP(model,
                        device_ids=[device.index],
                        output_device=device.index,
                        static_graph=True)

        #Log parameter count
        if is_logger:
            n_params = count_params(model)

            if config.verbose:
                print(f'\nn_params: {n_params}')
                sys.stdout.flush()


        #Create the optimizer
        optimizer = torch.optim.Adam(model.parameters(), 
                                        lr=config.opt.learning_rate, 
                                        weight_decay=config.opt.weight_decay)

        if config.opt.scheduler == 'ReduceLROnPlateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=config.opt.gamma, patience=config.opt.scheduler_patience, mode='min')
        elif config.opt.scheduler == 'CosineAnnealingLR':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.opt.scheduler_T_max)
        elif config.opt.scheduler == 'StepLR':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 
                                                        step_size=config.opt.step_size,
                                                        gamma=config.opt.gamma)
        else:
            raise ValueError(f'Got {config.opt.scheduler=}')


        # Creating the losses
        l2loss = LpLoss(d=2, p=2)
        h1loss = H1Loss(d=2)
        if config.opt.training_loss == 'l2':
            train_loss = l2loss
        elif config.opt.training_loss == 'h1':
            train_loss = h1loss
        else:
            raise ValueError(f'Got training_loss={config.opt.training_loss} but expected one of ["l2", "h1"]')
        eval_losses={'h1': h1loss, 'l2': l2loss}

        if config.verbose and is_logger:
            print('\n### MODEL ###\n', model)
            print('\n### OPTIMIZER ###\n', optimizer)
            print('\n### SCHEDULER ###\n', scheduler)
            print('\n### LOSSES ###')
            print(f'\n * Train: {train_loss}')
            print(f'\n * Test: {eval_losses}')
            print(f'\n### Beginning Training...\n')
            sys.stdout.flush()

        trainer = Trainer(model, n_epochs=config.opt.n_epochs,
                          patience=config.opt.patience,
                        device=device,
                        mg_patching_levels=config.patching.levels,
                        mg_patching_padding=config.patching.padding,
                        mg_patching_stitching=config.patching.stitching,
                        use_distributed=config.distributed.use_distributed,
                        verbose=config.verbose and is_logger,
                        wandb_log=False)


        trainer.train(train_loader, val_loader,
                    None,
                    model, 
                    optimizer,
                    scheduler, 
                    regularizer=False, 
                    training_loss=train_loss,
                    eval_losses=eval_losses)

        self.model = model
        self.trainer = trainer
        self.eval_loss = l2loss
        return {}, []

    def evaluate(self, data):
        loader = self._get_dataloder(data, self.config.data.test_batch_sizes)
        loss_dict = {"eval_loss": self.eval_loss}
        predictions, error_dict = self.trainer.evaluate(self.model, loss_dict,
                                                        loader, return_predictions=True)
        error = error_dict["_eval_loss"]
        return predictions.squeeze(), error
    