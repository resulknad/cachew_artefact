
from absl import app
from absl import flags

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
import sys


FLAGS = flags.FLAGS
flags.DEFINE_boolean('timeseries', False, 'If true, produces the time-series plot, rather than averaging')
flags.DEFINE_string("path", "", "Location to the experiment directory")
flags.DEFINE_string("save_path", None, "The location where the plots should be saved")
flags.DEFINE_enum("column", "batch_timestamp_us", ["batch_timestamp_us", "wait_time_ms", "had_to_wait", "result_queue_size", "relative_wait_fraction", "throughput", "epoch_time", "relative_batch_time_improvement"], 
  "The column to plot. Based on this choice, the plot type will change")
flags.DEFINE_enum("extension_type", "pdf", ["pdf", "png", "jpg"], "The type of the plots saved on disk")
flags.DEFINE_integer("batch_delta", 50, "Only applies when column=='batch_timestamp_us'. Will compute time it takes to process `batch_delta` batches")
flags.DEFINE_integer("subsample_rate", 10, "The rate at which to sub-sample the dataset in time-series plots")
flags.DEFINE_integer("upper_bound_y", None, "Defines an upped bound on the y axis")
flags.DEFINE_integer("warmup_epochs", 0, "The number of epochs to discard as warmup epochs. Applies to the epoch time and throughput plots")
flags.DEFINE_integer("warmup_batches", 0, "The number of batches to drop at the beginning of an epoch")


def get_epoch_times(path):
  res = {}

  normal = re.compile("TimeHistory: [0-9]+.?[0-9]* seconds")
  effnet = re.compile("- [0-9]+s -")
  simclr = re.compile("[0-9]+ seconds")

  for subdir in [name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]:
    times = []
    worker_count = subdir.split("_")[0]
    for subsubdir in [name for name in os.listdir(os.path.join(path, subdir)) if os.path.isdir(os.path.join(path, subdir, name))]:
      results_path = os.path.join(path, subdir, subsubdir, "output.log")
      if os.path.isfile(results_path):

        with open(results_path, "r") as f:
          for line in f.readlines():
            match_normal = normal.search(line)
            match_effnet = effnet.search(line)
            match_simclr = simclr.search(line)

            # Check if there's a match; only look for the first one
            if match_normal or match_effnet or match_simclr:
              if match_normal:
                times.append(float(match_normal.group(0).split()[1]))
              elif match_effnet: 
                times.append(float(match_effnet.group(0).split()[1][:-1]))
              else:
                times.append(float(match_simclr.group(0).split()[0]))
              break

    if times:
      res[worker_count] = times
    else:
      print(f"WARNING: Could not epoch retrieve time from {path} for {worker_count} workers", file=sys.stderr)

  return res


def get_results(path, warmup_batches=0):
  res = {}

  for subdir in [name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]:
    results_path = os.path.join(path, subdir, "run_1", "metrics.csv")
    if os.path.isfile(results_path):
      worker_count = subdir.split("_")[0]
      res[worker_count] = pd.read_csv(results_path)

      # In case multiple epochs were executed in a run, we need to discard rest of epochs
      repeats = res[worker_count].index[res[worker_count]["batch_timestamp_us"] == "batch_timestamp_us"].tolist()
      if repeats:
        res[worker_count] = res[worker_count].head(repeats[0])
      res[worker_count] = res[worker_count].iloc[warmup_batches:]

  return res


def line_plot(d, x_label, y_label, title, upper_bound_y=None, save_path=None, 
  plot_name=None):
  """
  d has the structure: "label": (x, y, std_dev)
  dev may be omitted
  """
  line_types = ["solid"]  # , "densely dashed", "dashed", "dashdot"]
  idx = 0
  for k, v in d.items():
    line_type = line_types[idx % len(line_types)]
    if len(v) == 3:
      plt.errorbar(v[0], v[1], v[2], label=k, linestyle=line_type, marker='^')
    else:
      plt.errorbar(v[0], v[1], linestyle=line_type, label=k)
    idx += 1

  plt.legend()
  plt.grid()
  plt.title(title)
  plt.xlabel(x_label)
  plt.ylabel(y_label)

  if upper_bound_y:
    plt.ylim(top=upper_bound_y)

  if save_path and plot_name:
    plt.savefig(os.path.join(save_path, plot_name))
  else:
    plt.show()


def plot_delta_batch_time(path, save_path, batch_delta, timeseries=False, 
  extension_type="png", subsample_rate=25, upper_bound_y=None):
  results = get_results(path)

  lbl = os.path.basename(os.path.normpath(path)).split("_")[0]
  row_count = min([v.shape[0] for _, v in results.items()]) - batch_delta
  col_count = len(results)

  # Compute the deltas
  if timeseries:
    data = {}
    for k, v in results.items():
      temp = v["batch_timestamp_us"].astype(np.float64)
      a = temp.tail(-batch_delta)
      b = temp.head(-batch_delta)
      x = np.arange(a.shape[0], step=subsample_rate, dtype=np.int32)
      y = (a.iloc[x].to_numpy() - b.iloc[x].to_numpy()) / 10 ** 6  # convert from us to s
      data[k] = (x, y)
    x_label = "Batch Index"
    plot_name=f"batch_time_timeseries_{lbl}"
  else:
    x = []
    y = []
    dev = []
    keys = sorted(list(results.keys()), key=lambda x: int(x))
    for k in keys:
      v = results[k]
      temp = v["batch_timestamp_us"].head(row_count).astype(np.float64)
      a = temp.tail(-batch_delta)
      b = temp.head(-batch_delta)
      temp = (a.to_numpy() - b.to_numpy()) / 10 ** 6  # convert from us to s
      x.append(int(k))
      y.append(np.mean(temp))
      dev.append(np.std(temp))
    data = {lbl: (x, y, dev)}
    x_label = "Worker Count"
    plot_name=f"batch_time_plot_{batch_delta}_batches_{lbl}"

  # Run plotting
  line_plot(data, x_label=x_label, y_label="Time [s]", 
    title=f"Last {batch_delta} batches processing time for {lbl}", 
    upper_bound_y=upper_bound_y, save_path=save_path, 
    plot_name=f"{plot_name}.{extension_type}")


def plot_wait_time(path, save_path, timeseries=False, extension_type="pdf", 
  subsample_rate=25, upper_bound_y=None):
  results = get_results(path)
  lbl = os.path.basename(os.path.normpath(path)).split("_")[0]

  if timeseries:
    data = {}
    for k, v in results.items():
      x = np.arange(v["wait_time_ms"].shape[0], step=subsample_rate, dtype=np.int32)
      y = v["wait_time_ms"].iloc[x].to_numpy(dtype=np.float64)
      data[k] = (x, y)
    x_label = "Batch Index"
    plot_name=f"wait_time_timeseries_{lbl}"
  else:
    x = []
    y = []
    dev = []
    keys = sorted(list(results.keys()), key=lambda x: int(x))
    for k in keys:
      temp = results[k]["wait_time_ms"].astype(np.float64)
      x.append(int(k))
      y.append(np.mean(temp))
      dev.append(np.std(temp))
    data = {lbl: (x, y, dev)}
    x_label = "Worker Count"
    plot_name=f"wait_time_plot_{lbl}"

  # Run plotting
  line_plot(data, x_label=x_label, y_label="Time [ms]", 
    title=f"Wait time per batch", save_path=save_path, 
    upper_bound_y=upper_bound_y, plot_name=f"{plot_name}.{extension_type}")


def plot_relative_wait_fraction(path, save_path, batch_delta, timeseries=False, 
  extension_type="pdf", subsample_rate=25):
  results = get_results(path)
  lbl = os.path.basename(os.path.normpath(path)).split("_")[0]
  
  row_count = min([v.shape[0] for _, v in results.items()]) - batch_delta
  col_count = len(results)

  # Compute the deltas
  if timeseries:
    data = {}
    for k, v in results.items():
      temp = v["batch_timestamp_us"].astype(np.float64)
      a = temp.tail(-batch_delta)
      b = temp.head(-batch_delta)
      diffs = (a.to_numpy() - b.to_numpy()) / 10 ** 3  # convert from us to ms

      # Get the cumulative wait times
      temp = v["wait_time_ms"].astype(np.float64).rolling(
        min_periods=1, window=batch_delta).sum()
      temp = temp.tail(-batch_delta).to_numpy()

      # Compute the fraction
      x = np.arange(a.shape[0], step=subsample_rate, dtype=np.int32)
      y = (temp / diffs)[x]
      data[k] = (x, y)
    x_label = "Batch Index"
    plot_name=f"wait_time_fraction_timeseries_{batch_delta}_batches_{lbl}"
  else:
    x = []
    y = []
    dev = []
    keys = sorted(list(results.keys()), key=lambda x: int(x))
    for k in keys:
      v = results[k]
      
      # Get the diffs 
      temp = v["batch_timestamp_us"].head(row_count).astype(np.float64)
      a = temp.tail(-batch_delta)
      b = temp.head(-batch_delta)
      diffs = (a.to_numpy() - b.to_numpy()) / 10 ** 3  # convert from us to ms

      # Get the cumulative wait times
      temp = v["wait_time_ms"].head(row_count).astype(np.float64).rolling(
        min_periods=1, window=batch_delta).sum()
      temp = temp.tail(-batch_delta).to_numpy()

      # Compute the fraction
      temp = temp / diffs
      x.append(int(k))
      y.append(np.mean(temp))
      dev.append(np.std(temp))
    data = {lbl: (x, y, dev)}
    x_label = "Worker Count"
    plot_name=f"wait_time_fraction_plot_{batch_delta}_batches_{lbl}"

  # Run plotting
  line_plot(data, x_label=x_label, y_label="Fraction [%]", 
    title=f"Relative wait time in last {batch_delta} batches for {lbl}", 
    save_path=save_path, plot_name=f"{plot_name}.{extension_type}")


def bar_plot(d, x_label, y_label, title, save_path=None, plot_name=None):
  """
  d has the structure: (x, y, std_dev) ; dev may be omitted
  """
  x = np.arange(len(d[0]))
  if len(d) == 3:
    plt.bar(x, d[1], yerr=d[2], align="center")
  else:
    plt.bar(x, d[1], align="center")

  plt.xticks(x, labels=d[0])
  plt.title(title)
  plt.xlabel(x_label)
  plt.ylabel(y_label)

  plt.ylim(bottom=0)

  if save_path and plot_name:
    plt.savefig(os.path.join(save_path, plot_name))
  else:
    plt.show()  


def plot_result_queue_size(path, save_path, batch_delta, extension_type="pdf"):
  results = get_results(path)
  lbl = os.path.basename(os.path.normpath(path)).split("_")[0]

  y = []
  dev = []
  keys = sorted(list(results.keys()), key=lambda x: int(x))
  for k in keys:
    v = results[k]
    temp = v["result_queue_size"].astype(np.int32).rolling(
        min_periods=1, window=batch_delta).sum().tail(-batch_delta)
    y.append(np.mean(temp))
    dev.append(np.std(temp))

  bar_plot((keys, y, dev), x_label="Worker Count", y_label="Queue Size", 
    title=f"Average queue size for {lbl}", save_path=save_path, 
    plot_name=f"average_queue_size_{lbl}.{extension_type}")


def step_plot(d, x_label, y_label, title, save_path=None, 
  plot_name=None):
  """
  d has the structure: "label": (x, y)
  dev may be omitted
  """
  line_types = ["solid"]  # , "densely dashed", "dashed", "dashdot"]
  idx = 0
  for k, v in d.items():
    line_type = line_types[idx % len(line_types)]
    if len(v):
      plt.step(v[0], v[1], label=k, linestyle=line_type)
    idx += 1

  plt.legend()
  plt.grid()
  plt.title(title)
  plt.xlabel(x_label)
  plt.ylabel(y_label)

  if save_path and plot_name:
    plt.savefig(os.path.join(save_path, plot_name))
  else:
    plt.show()


def plot_had_to_wait(path, save_path, batch_delta, timeseries=False, 
  extension_type="pdf", subsample_rate=25):
  results = get_results(path)
  lbl = os.path.basename(os.path.normpath(path)).split("_")[0]

  if timeseries:
    data = {}
    for k, v in results.items():
      y = v["had_to_wait"].astype(np.int32).rolling(
        min_periods=1, window=batch_delta).sum().tail(-batch_delta).to_numpy()
      x = np.arange(y.shape[0], step=subsample_rate, dtype=np.int32)
      data[k] = (x, y[x])
    step_plot(data, x_label="Batch Index", y_label="Wait Count", 
      title=f"Wait count in last {batch_delta} batches for {lbl}", 
      save_path=save_path, 
      plot_name=f"avearge_waits_timeseries_{batch_delta}_batches_{lbl}.{extension_type}")
  else:
    y = []
    dev = []
    keys = sorted(list(results.keys()), key=lambda x: int(x))
    for k in keys:
      v = results[k]
      temp = v["had_to_wait"].astype(np.int32).rolling(
        min_periods=1, window=batch_delta).sum().tail(-batch_delta)
      y.append(np.mean(temp))
      dev.append(np.std(temp))

    bar_plot((keys, y, dev), x_label="Worker Count", y_label="Average Waits", 
      title=f"Average waits in {batch_delta} batches for {lbl}", 
      save_path=save_path, 
      plot_name=f"avearge_waits_{batch_delta}_batches_{lbl}.{extension_type}")


def plot_epoch_time(path, save_path, upper_bound_y=None, warmup_epochs=0, 
  extension_type="pdf"):
  results = get_epoch_times(path)
  lbl = os.path.basename(os.path.normpath(path)).split("_")[0]  

  x = []
  y = []
  dev = []
  keys = sorted(list(results.keys()), key=lambda x: int(x))
  for k in keys:
    temp = results[k][warmup_epochs:]
    x.append(int(k))
    y.append(np.mean(temp, dtype=np.float64))
    dev.append(np.std(temp, dtype=np.float64))
  data = {lbl: (x, y, dev)}
  
  print("Epoch times: ")
  print(y)
  print("Workers: ")
  print(x)

    # Run plotting
  line_plot(data, x_label="Worker Count", y_label="Epoch Time [s]", 
    title=f"Epoch time for {lbl}", upper_bound_y=upper_bound_y, 
    save_path=save_path, plot_name=f"epoch_time_{lbl}.{extension_type}")


def plot_throughput(path, save_path, upper_bound_y=None, warmup_epochs=0, 
  extension_type="pdf"):
  results = get_epoch_times(path)
  lbl = os.path.basename(os.path.normpath(path)).split("_")[0]

  # Sizes are in MiB
  data_sizes = {
    "resnet": 367098,
    "retinanet": 588115,
    "efficientnet": 269386,
    "simclr": 1474886
  }

  x = []
  y = []
  dev = []
  keys = sorted(list(results.keys()), key=lambda x: int(x))
  for k in keys:
    temp = [data_sizes[lbl] / x for x in results[k][warmup_epochs:]]
    x.append(int(k))
    y.append(np.mean(temp, dtype=np.float64))
    dev.append(np.std(temp, dtype=np.float64))
  data = {lbl: (x, y, dev)}

    # Run plotting
  line_plot(data, x_label="Worker Count", y_label="Throughput [MiB/s]", 
    title=f"Throughput values for {lbl}", upper_bound_y=upper_bound_y, 
    save_path=save_path, plot_name=f"throughput_{lbl}.{extension_type}")


def plot_relative_batch_time_improvement(path, save_path, batch_delta, 
  warmup_batches=0, timeseries=False, extension_type="pdf", subsample_rate=25):
  results = get_results(path, warmup_batches)
  lbl = os.path.basename(os.path.normpath(path)).split("_")[0]

  # Compute the deltas
  if timeseries:
    data = {}
    for k, v in results.items():
      temp = v["batch_timestamp_us"].astype(np.float64)

      # Get the last x batch times
      a = temp.tail(-batch_delta)
      b = temp.head(-batch_delta)
      c = (a - b) / 10 ** 3  # Convert from us to ms

      # Get the improvements
      a = c.tail(-1)
      b = c.head(-1)
      y = 1.0 - a / b

      x = np.arange(y.shape[0], step=subsample_rate, dtype=np.int32)
      data[k] = (x, y.iloc[x])
    x_label = "Batch Index"
    plot_name=f"relative_imporvement_{batch_delta}_batches_time_timeseries_{lbl}"
  else:
    x = []
    y = []
    dev = []
    keys = sorted(list(results.keys()), key=lambda x: int(x))
    for k in keys:
      v = results[k]
      temp = v["batch_timestamp_us"].astype(np.float64)
      a = temp.tail(-batch_delta)
      b = temp.head(-batch_delta)
      temp = (a.to_numpy() - b.to_numpy()) / 10 ** 3  # convert from us to s
      x.append(int(k))
      y.append(np.mean(temp))
      dev.append(np.std(temp))

    temp = []
    for i in range(1, len(y)):
      temp.append((1.0 - y[i] / y[i - 1]) * 100)
    y = temp

    data = {lbl: (x[1:], y, [0] * (len(x) - 1))}
    x_label = "Worker Count"
    plot_name=f"batch_time_plot_{batch_delta}_batches_{lbl}"
    pass

  # Run plotting
  line_plot(data, x_label=x_label, y_label="Relative Improvement [%]", 
    title=f"Last {batch_delta} batches processing time for {lbl}", 
    save_path=save_path, plot_name=f"{plot_name}.{extension_type}")


def main(argv):
  del(argv)

  path = FLAGS.path
  column = FLAGS.column
  save_path = FLAGS.save_path
  batch_delta = FLAGS.batch_delta
  timeseries = FLAGS.timeseries
  subsample_rate = FLAGS.subsample_rate
  extension_type = FLAGS.extension_type
  upper_bound_y = FLAGS.upper_bound_y
  warmup_epochs = FLAGS.warmup_epochs
  warmup_batches = FLAGS.warmup_batches

  if column == "batch_timestamp_us":
    plot_delta_batch_time(path, save_path, batch_delta, timeseries, 
      extension_type=extension_type, subsample_rate=subsample_rate, 
      upper_bound_y=upper_bound_y)
  elif column == "wait_time_ms":
    plot_wait_time(path, save_path, timeseries, extension_type=extension_type, 
      subsample_rate=subsample_rate, upper_bound_y=upper_bound_y)
  elif column == "relative_wait_fraction":
    plot_relative_wait_fraction(path, save_path, batch_delta, timeseries, 
      extension_type=extension_type, subsample_rate=subsample_rate)
  elif column == "result_queue_size":
    plot_result_queue_size(path, save_path, batch_delta, 
      extension_type=extension_type)
  elif column == "had_to_wait":
    plot_had_to_wait(path, save_path, batch_delta, timeseries=timeseries,
      extension_type=extension_type, subsample_rate=subsample_rate)
  elif column == "epoch_time":
    plot_epoch_time(path, save_path, upper_bound_y=upper_bound_y, 
      warmup_epochs=warmup_epochs, extension_type=extension_type)
  elif column == "throughput":
    plot_throughput(path, save_path, upper_bound_y=upper_bound_y, 
      warmup_epochs=warmup_epochs, extension_type=extension_type)
  elif column == "relative_batch_time_improvement":
    plot_relative_batch_time_improvement(path, save_path, batch_delta, 
      warmup_batches=warmup_batches, timeseries=timeseries, 
      extension_type=extension_type, subsample_rate=25)


if __name__ == '__main__':
  app.run(main)
