Training deep neural operators
------------------------------

In the **dataset/**, you can find notebooks to generate the data (*generate_dataset.ipynb*)
and then to split it (*prepare_dataset.ipynb*). After running these notebooks, you should see
files *train.npy*, *val.npy*, and *test.npy* in the **dataset/** folder. 

The configuration files for the grid search can be found in **hyperparameter_study/configs/**.
To reproduce the results, one can run a script *run_experiments.sh* in **hyperparameter_study/**
with the desired parameters. For example, to run a set of experiments for sampled FNO on CPU
with 3 repetitions that saves the results to **hyperparameter_study/experiments/fno_swim_results/**,
you can use the following command:

.. code-block:: bash

    bash run_experiments.sh -f fno_swim.json -t fno_swim_results -d cpu -r 3

Note that you need to create the **experiments/** directory first.

The folder **hyperparameter_study/results/** contains .csv files with the results from the grid search.
Files *\*_full.csv* contain metrics for each configuration repeated three times with different random seeds.
These results are averaged in *\*_mean.csv* and then used for plotting. Two notebooks in the same folder
aggregate the results from the *sacred* folders (*aggregate_results.ipynb*) and plot the results (*view_results.ipynb*).