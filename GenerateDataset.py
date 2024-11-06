from Connectome import *
from utils import *
import numpy as np
import pandas as pd
import os
import re

directory = './repeated_10_scale_500'
dataframes = []

for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
    label = re.match(r"(\d+)_", filename)
    label = label.group(1)
    print(f"Analyzing connectome {label}")
    # I am going to assume it is a valid file
    connectome = Connectome(file=f, label=label)
    connectome.read_matrix()
    connectome.read_net(use_3d=True)
    cent_metrics = connectome.store_centrality_metrics()
    df = connectome.gather_attributes()
    dataframes.append(df)
    print("finished")

dataset = pd.concat(dataframes, ignore_index=True)
dataset.to_csv("connectome_data.csv", index=False)

        

