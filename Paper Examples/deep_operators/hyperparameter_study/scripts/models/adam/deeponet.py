import os
os.environ["DDE_BACKEND"] = "pytorch"

import torch
import deepxde as dde
import numpy as np

def pod(y):
    n = len(y)
    y_mean = np.mean(y, axis=0)
    y = y - y_mean
    C = 1 / (n - 1) * y.T @ y
    w, v = np.linalg.eigh(C)
    w = np.flip(w)
    v = np.fliplr(v)
    v *= len(y_mean) ** 0.5
    return y_mean, v.copy()



class PODDeepONet:
    def __init__(self, device, model_params, artifact_folder):
        widths = [model_params["input_dim"]]
        widths += [model_params["layer_width"]] * model_params["n_layers"]
        widths += [model_params["n_modes"]]
        self.widths = widths
        self.device = device
        self.n_modes = model_params["n_modes"]
        self.train_mean_tensor = None
        self.grid = np.linspace(0, 1, model_params["input_dim"])[:, None].astype(np.float32)
        self.predictor = None
        self.model_filename = os.path.join(artifact_folder, "model")

        torch.set_default_device(device)

    def _clean_data(self, u0, u1):
        return u0.astype(np.float32), u1.astype(np.float32)

    def _prepare_data(self, train_data, val_data):
        train_u0, train_u1 = self._clean_data(*train_data)
        val_u0, val_u1 = self._clean_data(*val_data)
        
        data = dde.data.TripleCartesianProd(X_train=(train_u0, self.grid),
                                    y_train=train_u1,
                                    X_test=(val_u0, self.grid),
                                    y_test=val_u1)
        



        train_mean, train_pod_modes = pod(train_u1)
        train_mean_tensor = torch.tensor(train_mean)
        return data, train_mean_tensor, train_pod_modes

    def _build_predictor(self, train_pod_modes, train_mean_tensor, data, weight_decay):
        net = dde.nn.PODDeepONet(
            train_pod_modes[:, :self.n_modes],
            self.widths,
            "tanh",
            "Glorot normal"
        )
        net.regularizer = ("l2", weight_decay)

        def output_transform(_, outputs):
            return outputs / self.n_modes + train_mean_tensor

        net.apply_output_transform(output_transform)
        model = dde.Model(data, net)
        return model



    def train(self, train_data, val_data, training_params, _log):
        data, train_mean_tensor, train_pod_modes = self._prepare_data(train_data, val_data)
        model = self._build_predictor(train_pod_modes, train_mean_tensor, data, training_params["weight_decay"])

        early_stopping = dde.callbacks.EarlyStopping(patience=training_params["patience"])
        checkpoint = dde.callbacks.ModelCheckpoint(self.model_filename, save_better_only=True)
        model.compile("adam", lr=training_params["learning_rate"], metrics=["mean l2 relative error"])

        train_losses, train_state = model.train(iterations=training_params["n_epochs"],
                                                batch_size=training_params["batch_size"],
                                                display_every=100, # approx. len(train_size)/batch_size
                                                callbacks=[early_stopping, checkpoint])
    

        state_filepath = f"{self.model_filename}-{train_state.best_step}.pt"
        model.restore(state_filepath)
        self.predictor = model

        train_loss = [loss[0] for loss in train_losses.loss_train]
        val_metric = [metric[0] for metric in train_losses.metrics_test]

        return {"train_losses": train_loss, "val_metrics": val_metric}, train_losses.steps


    def evaluate(self, data):
        u0, u1 = self._clean_data(*data)
        predictions = self.predictor.predict((u0, self.grid))
        metric = float(self.predictor.metrics[0](u1, predictions))
        return predictions, metric