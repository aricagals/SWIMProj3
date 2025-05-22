#!/bin/bash

# Set up base directories
base_config_dir="configs"
experiments_dir="experiments"
log_dir="$experiments_dir/_logs"
tmp_dir="$experiments_dir/_tmp"
scripts_dir="scripts"

device_str="0123" # devices to use for experiments
config_filename="grid_search.json" # configuration file for the job
tag="default_tag" # tag of the job used as a folder name for the results
repeats="1" # number of repetitions for each experiment
job_id=1 # id of the job

# Parse command line arguments
optstring="d:f:t:r:i:"
while getopts $optstring arg; do
    case $arg in
    d)
        device_str=$OPTARG
        if [ $device_str == "cpu" ]; then
            devices=("cpu")
        else
            devices=()
            for i in $(seq 1 ${#device_str})
            do
                device_id=${device_str:i-1:1}
                devices+=("cuda:$device_id")
            done
        fi
        ;;
    f)
        config_filename=$OPTARG
        ;;
    t)
        tag=$OPTARG
        ;;
    r)
        repeats=$OPTARG
        ;;
    i)
        job_id=$OPTARG
    esac
done


config_dir=$base_config_dir/job_$job_id
log_dir=$log_dir/job_$job_id
tmp_dir=$tmp_dir/job_$job_id

# Create directories
mkdir -p $experiments_dir
mkdir -p $log_dir
mkdir -p $tmp_dir

# Check if there are running experiments on the devices
busy_devices=()
for device in ${devices[*]}
do
    device_dir=$config_dir/$device/
    if [ -d $device_dir ]; then
        echo "It seems that $device is busy with job #$job_id: $device_dir exists."
        busy_devices+=($device)
    fi
done
if [ ! -z $busy_devices ]; then 
    echo "Please stop the experiments first, and then delete config folders for devices [${busy_devices[*]}]."
    exit
fi

# Generate configs for the grid search
config_path=$base_config_dir/$config_filename
common_log=$log_dir/highlevel.log
python $scripts_dir/split_experiments.py -f $config_path  -r $repeats -d "${devices[*]}" -F $config_dir -l $common_log || exit 1

run_experiment() {
    device=$1
    full_tag=$experiments_dir/$tag
    device_dir=$config_dir/$device/
    device_tmp=$tmp_dir/$device
    device_log=$log_dir/$device.log

    python $scripts_dir/run_device_experiments.py -d $device -t $full_tag -f $device_dir -a $device_tmp -l $common_log &> $device_log
    # Clean up
    rm -r $device_dir
    rm -r $device_tmp
}

# Run experiments on the devices
for device in ${devices[*]}
do
    run_experiment $device &
done