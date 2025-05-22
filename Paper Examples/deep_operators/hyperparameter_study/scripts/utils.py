import logging 
import models

def get_logger(log_filename, logger_name):
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    handler = logging.FileHandler(log_filename)
    handler.setFormatter(formatter)
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    return logger

def build_model(device, model_params, artifact_folder, seed):
    model = model_params["architecture"]
    if model == "in_fourier":
        return models.InFourier(device, model_params)
    if model == "fcn":
        return models.FCN(device, model_params)
    if model == "deeponet":
        return models.PODDeepONet(device, model_params, artifact_folder)
    if model == "fno":
        return models.FNO1D(device, model_params, seed)
    if model == "in_fourier_swim":
        return models.InFourierSwim(model_params, seed)
    if model == "fcn_swim":
        return models.DenseSwim(model_params, seed)
    if model == "deeponet_swim":
        return models.PODDeepONetSwim(model_params, seed)
    if model == "fno_swim":
        return models.FNO1DSwim(model_params, seed)
    
    raise ValueError(f"Unknown architecture: {model}")
