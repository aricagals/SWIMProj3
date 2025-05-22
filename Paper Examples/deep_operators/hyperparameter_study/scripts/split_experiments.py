import argparse
import logging
import os
import itertools as it
import json
import utils


def flatten_dict(nested_dict, prefix=""):
    items = ()
    for key, value in nested_dict.items():
        if isinstance(value, dict):
            value = flatten_dict(value, prefix=f"{key}.")
            items += tuple(value.items())
        else:
            items += ((key, value),)
    return {f"{prefix}{key}": value for key, value in items}


def unflatten_dict(flat_dict):
    nested_dict = dict()
    for key, value in flat_dict.items():
        subkeys = key.split(".")
        sublevel = nested_dict
        for subkey in subkeys[:-1]:
            if subkey not in sublevel:
                sublevel[subkey] = dict()
            sublevel = sublevel[subkey]
        sublevel[subkeys[-1]] = value
    return nested_dict


def split_grid_search(grid_search_config, repeats, devices, config_folder, logger):
    flat_config = flatten_dict(grid_search_config)

    # Get all possible combinations of parameters
    options_list = []
    for key, options in flat_config.items():
        key_options = []
        if not isinstance(options, list):
            raise TypeError(
                f"All options in the config file must be lists, "
                "but received: {key} -> {options}."
            )
        for option in options:
            key_options.append((key, option))
        options_list.append(key_options)
    options = list(it.product(*options_list))
    repeated_options = options * repeats

    # Put combinations to a dict and add the "device" field
    device_configs = {device: [] for device in devices}
    for run_id, params in enumerate(repeated_options):
        config = {key: value for (key, value) in params}
        config = unflatten_dict(config)
        device = devices[run_id % len(devices)]
        config["device"] = device
        device_configs[device].append(config)

    # Save all the condifgs for each device to the corresponding folder
    for device, configs in device_configs.items():
        device_folder = os.path.join(config_folder, f"{device}")
        os.makedirs(device_folder, exist_ok=True)
        for i, config in enumerate(configs):
            config_path = os.path.join(device_folder, f"{i:04}.json")
            with open(config_path, "w") as fout:
                json.dump(config, fout, indent=4)
    
    # Log the split
    logger.info(f"Split {len(options)} experiment(s) with {repeats} repetition(s) between devices {devices}.")
    logger.info(f"Total: {len(repeated_options)} experiment(s).")
    for device, configs in device_configs.items():
        logger.info(f"device {device} -> scheduled {len(configs)} experiment(s).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--grid_search_config")
    parser.add_argument("-r", "--repeats", type=int, default=1)
    parser.add_argument("-d", "--devices")
    parser.add_argument("-F", "--config_folder")
    parser.add_argument("-l", "--log_filename")
    args = parser.parse_args()

    with open(args.grid_search_config) as fin:
        config = json.load(fin)
    
    devices = args.devices.split(" ")
    logger = utils.get_logger(args.log_filename, "highlevel")

    split_grid_search(config, args.repeats, devices, args.config_folder, logger)

