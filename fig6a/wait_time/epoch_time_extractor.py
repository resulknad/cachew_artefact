
from absl import app
from absl import flags

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
import sys


FLAGS = flags.FLAGS
flags.DEFINE_string("path", "", "Location to the experiment directory")
flags.DEFINE_string("save_path", None, "The location where the csv should be saved")


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


def main(argv):
  del(argv)

  path = FLAGS.path
  save_path = FLAGS.save_path

  columns = ["model_name", "compute_mode", "worker_count", "epoch_time"]
  df = pd.DataFrame(columns=columns)
  idx = 0

  for mode in [name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]:
    for model in [name for name in os.listdir(os.path.join(path, mode)) if os.path.isdir(os.path.join(path, mode, name))]:
      res = get_epoch_times(os.path.join(path, mode, model))
      for key in sorted(list(res.keys()), key=lambda x: int(x)):
        row = [model, mode, int(key), res[key][0]]
        df. loc[idx] = row
        idx += 1

  print(df)
  df.to_csv(save_path, index=False)



if __name__ == '__main__':
  app.run(main)
