import abc
import copy
import numpy as np
from timeit import default_timer
import torch
import torch.utils.data as torch_data


def mean_rel_l2(pred, true):
    pred = torch.flatten(pred, start_dim=1)
    true = torch.flatten(true, start_dim=1)
    diff_norm = torch.linalg.norm(pred - true, axis=-1)
    true_norm = torch.linalg.norm(true, axis=-1)
    return torch.mean(diff_norm / true_norm)


def mse(pred, true):
    pred = torch.flatten(pred, start_dim=1)
    true = torch.flatten(true, start_dim=1)
    return torch.sum(torch.abs(pred - true) ** 2, axis=-1).mean()


def get_loss_fn(loss_name):
    if loss_name == "mean_rel_l2":
        return mean_rel_l2
    elif loss_name == "mse":
        return mse
    else:
        return None


class BaseModel(abc.ABC):
    def __init__(self, device, model_params):
        # Make sure the defualt device is 'cpu'
        # if code before set it up to a different value.
        torch.set_default_device("cpu")
        self.device = device
        self.build_predictor(model_params)
        self.predictor.to(device)
        self.batch_size = None

    @abc.abstractmethod
    def build_predictor(self, model_params):
        pass

    @abc.abstractmethod
    def loss_fn(self, pred, true):
        pass

    @abc.abstractmethod
    def metric_fn(self, pred, true):
        pass

    def train(self, train_data, val_data, params, logger):
        batch_size = params["batch_size"]
        self.batch_size = batch_size
        train_loader = self.get_dataloader(
            train_data, batch_size, shuffle=True
        )
        val_loader = self.get_dataloader(
            val_data, batch_size, shuffle=False
        )

        optimizer = torch.optim.Adam(
            self.predictor.parameters(), lr=params["learning_rate"],
            weight_decay=params["weight_decay"]
        )

        n_epochs = params["n_epochs"]
        patience = params["patience"]
        print_every = params["print_every"]

        best_val_epoch = 0
        best_val_metric = np.inf
        best_val_state_dict = None

        losses = {"train_losses": [], "val_metrics": []}
        steps = []

        for epoch in range(n_epochs):
            self.predictor.train()
            t_start = default_timer()

            train_loss = 0
            for u0, u1 in train_loader:
                u0, u1 = u0.to(self.device), u1.to(self.device)
        
                optimizer.zero_grad()
                out = self.predictor(u0)
                loss = self.loss_fn(out, u1)
                loss.backward()
        
                optimizer.step()
                train_loss += loss.item()

            train_loss /= len(train_loader)
            self.predictor.eval()
            _, val_metric = self.evaluate(val_loader)

            t_end = default_timer()

            if (epoch + 1) % print_every == 0:
                logger.info(f"Epoch: {epoch + 1}/{n_epochs}, time for the last epoch: {t_end - t_start:.2f}s")
                logger.info(f"train loss: {train_loss:.4f}, val metric: {val_metric:.4f}")

                losses["train_losses"].append(train_loss)
                losses["val_metrics"].append(val_metric)
                steps.append(epoch+1)

            # Early stopping
            if val_metric < best_val_metric:
                best_val_metric = val_metric
                best_val_epoch = epoch 
                best_val_state_dict = copy.deepcopy(self.predictor.state_dict())

            if patience is not None and (epoch - best_val_epoch) > patience:
                logger.info(f"Stopped early after {epoch + 1}/{n_epochs} epochs.")
                break
        
        self.predictor.eval()
        if best_val_state_dict is not None:
            self.predictor.load_state_dict(best_val_state_dict)
        else:
            logger.warning("WARNING: couldn't find the best state dict.")

        return losses, steps
    
    def evaluate(self, data):
        if not isinstance(data, torch_data.DataLoader):
            data = self.get_dataloader(data, self.batch_size, shuffle=False)
        
        self.predictor.eval()
        metric = 0
        predictions = []
        with torch.no_grad():
            for u0, u1 in data:
                u0, u1 = u0.to(self.device), u1.to(self.device)
                pred = self.predictor(u0)
                metric += self.metric_fn(pred, u1).item()
                predictions.append(self.restore_data(pred))
        metric /= len(data)
        return torch.cat(predictions).cpu().numpy(), metric

    def get_dataloader(self, data, batch_size, shuffle=True):
        u0, u1 = data
        u0, u1 = torch.from_numpy(u0), torch.from_numpy(u1)
        dataset = torch_data.TensorDataset(u0, u1)
        if batch_size is None:
            batch_size = len(u0)
        return torch_data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    def restore_data(self, output):
        return output