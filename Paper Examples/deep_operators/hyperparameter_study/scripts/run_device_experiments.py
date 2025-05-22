import argparse
from run_single_experiment import run_experiment
import os
import utils


def run_experiments(device, tag, config_folder, artifacts_folder, highlevel_logger):
    config_files = []
    for config_file in sorted(os.listdir(config_folder)):
        if config_file.endswith(".json"):
            config_files.append(os.path.join(config_folder, config_file))

    os.makedirs(artifacts_folder, exist_ok=True)

    for i, config_file in enumerate(config_files):
        highlevel_logger.info(f"{device} -> started experiment #{i+1}/{len(config_files)}.")
        try:
            run_experiment(config_file, tag, artifacts_folder)
        except:
            highlevel_logger.error(
                f"{device} -> something went wrong with run #{i+1}/{len(config_files)}."
            )
            continue
        highlevel_logger.info(
            f"{device} -> finished experiment #{i+1}/{len(config_files)}."
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--device")
    parser.add_argument("-t", "--tag")
    parser.add_argument("-f", "--config_folder")
    parser.add_argument("-a", "--artifacts_folder")
    parser.add_argument("-l", "--log_filename")
    args = parser.parse_args()

    logger = utils.get_logger(args.log_filename, "highlevel")
    run_experiments(
        args.device, args.tag, args.config_folder, args.artifacts_folder, logger
    )
