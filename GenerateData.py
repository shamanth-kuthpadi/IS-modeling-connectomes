from Connectome import *
from utils import *
import numpy as np
import pandas as pd
import os
import re
from tqdm import tqdm

directory = './data/repeated_10_scale_500'
dataframes = []
files = os.listdir(directory)
total_files = len(files)

for filename in tqdm(files, desc="Analyzing connectomes", unit="file"):
    f = os.path.join(directory, filename)
    label = re.match(r"(\d+)_", filename)
    label = label.group(1)
    connectome = Connectome(file=f, label=label)
    connectome.read_net()
    cent_metrics = connectome.store_centrality_metrics()
    evec_mapper = connectome.store_eigenvectors()
    df = connectome.gather_attributes()
    dataframes.append(df)

dataset = pd.concat(dataframes, ignore_index=True)
dataset.to_csv("connectome_data.csv", index=False)