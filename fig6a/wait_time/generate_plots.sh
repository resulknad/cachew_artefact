#!/usr/bin/env bash

cd "$(dirname "$0")"

# Parameters
plot_script=${1:-"plotter.py"}
make_dirs=${2:-true}

# Constants

plot_types=("batch_timestamp_us" "wait_time_ms" "result_queue_size" "relative_wait_fraction" "had_to_wait" "throughput" "epoch_time" "relative_batch_time_improvement")
compute_mode=("compute" "cache")
#base_dir="/home/aymond/Documents/phd/utils/easl-utils/scripts/wait_time/results/"
#save_dir="/home/aymond/Documents/phd/utils/easl-utils/scripts/wait_time/results/plots"
base_dir="/home/dan/data/garbage/temp/wait_time_stats/results"
save_dir="/home/dan/data/garbage/temp/wait_time_stats/plots"
dirs=("efficientnet" "resnet" "retinanet" "simclr")


# Make the directory hierarchy for saving the plots
if [ "$make_dirs" == true ]; then
  for mode in "${compute_mode[@]}"; do
    for plot in "${plot_types[@]}"; do
      for type in plot timeseries; do
        for dir in "${dirs[@]}"; do
          model="$(cut -d'_' -f1 <<< ${dir} )"
          mkdir -p ${save_dir}/${mode}/${plot}/${type}/${dir}
        done
      done
    done
  done
fi

# Plot the data
for mode in "${compute_mode[@]}"; do
  for dir in "${dirs[@]}"; do
    # Test to see if results directory exist
    if [ -d ${base_dir}/${mode}/${dir} ]; then
      model="$(cut -d'_' -f1 <<< ${dir} )"
      echo "Plotting for ${model}"
      for plot in "${plot_types[@]}"; do
        if [ "${plot}" == "result_queue_size" ]; then
          python3 ${plot_script} --timeseries=False --column=${plot} --path=${base_dir}/${mode}/${dir} --save_path=${save_dir}/${mode}/${plot}/plot/${model}
        else
          extra_params=""
          if [ "${plot}" == "wait_time_ms" ]; then
            if [ "${mode}" == "cache" ]; then
              if [ "${model}" == "efficientnet" ]; then
                extra_params="--upper_bound_y=500"
              elif [ "${model}" == "retinanet" ]; then
                extra_params="--upper_bound_y=1000"
              elif [ "${model}" == "resnet" ]; then 
                extra_params="--upper_bound_y=500"
              fi
            else
              if [ "${model}" == "efficientnet" ]; then
                extra_params="--upper_bound_y=800"
              elif [ "${model}" == "retinanet" ]; then
                extra_params="--upper_bound_y=800"
              elif [ "${model}" == "resnet" ]; then 
                extra_params="--upper_bound_y=1000"
              fi
            fi
          elif [ "${plot}" == "had_to_wait" ]; then
            extra_params="--batch_delta=20"
          elif [ "${plot}" == "relative_batch_time_improvement" ]; then
            extra_params="--warmup_batches=200"
          fi
          python3 ${plot_script} ${extra_params} --timeseries=False --column=${plot} --path=${base_dir}/${mode}/${dir} --save_path=${save_dir}/${mode}/${plot}/plot/${model}
          python3 ${plot_script} ${extra_params} --timeseries=True --column=${plot} --path=${base_dir}/${mode}/${dir} --save_path=${save_dir}/${mode}/${plot}/timeseries/${model}
        fi
      done
    fi
  done
done
