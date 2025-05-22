import argparse
import numpy as np
import os
from timeit import default_timer
import utils

from sacred import Experiment
from sacred.observers import FileStorageObserver

def run_experiment(config_file, tag, artifact_folder):

    ex = Experiment("adam-training")
    ex.observers.append(FileStorageObserver.create(tag))
    ex.add_config(config_file)


    @ex.capture(prefix="dataset_params")
    def load_data(data_folder, _log):
        train_data = np.load(os.path.join(data_folder, "train.npy"))
        val_data = np.load(os.path.join(data_folder, "val.npy"))
        test_data = np.load(os.path.join(data_folder, "test.npy"))

        for data, prefix in zip(
            [train_data, val_data, test_data], ["train", "val", "test"]
        ):
            u0, u1 = data
            _log.info(f"{prefix}: {u0.shape=}, {u1.shape=}")
        return train_data, val_data, test_data


    @ex.capture
    def build_model(device, model_params, seed):
        return utils.build_model(device, model_params, artifact_folder, seed)


    @ex.capture
    def train(model, train_data, val_data, _log, training_params=dict()):
        t_start = default_timer()
        losses, steps = model.train(train_data, val_data, training_params, _log)
        t_end = default_timer()
        for loss_name, loss_values in losses.items():
            for step, value in zip(steps, loss_values):
                ex.log_scalar(loss_name, value, step)
        ex.log_scalar("training_time", t_end - t_start)


    @ex.capture
    def evaluate(model, train_data, val_data, test_data, _log):
        splits = {"train": train_data, "val": val_data, "test": test_data}
        final_metrics = {}
        for split, data in splits.items():
            predictions, metric = model.evaluate(data)
            metric_name = f"eval_{split}_metric"
            final_metrics[metric_name] = metric
            # Log metric value
            ex.log_scalar(metric_name, metric)
            # Save predictions
            predictions_filename = os.path.join(artifact_folder,
                                                f"predictions_{split}.npy")
            np.save(predictions_filename, predictions)
            ex.add_artifact(predictions_filename)
        _log.info(f"Final metrics: {final_metrics}")

    @ex.main
    def run_experiment():
        train_data, val_data, test_data = load_data()
        model = build_model()
        train(model, train_data, val_data)
        evaluate(model, train_data, val_data, test_data)

    ex.run()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--config_filename")
    parser.add_argument("-t", "--tag")
    parser.add_argument("-a", "--artifacts_folder")
    args = parser.parse_args()
    run_experiment(args.config_filename, args.tag, args.artifacts_folder)
