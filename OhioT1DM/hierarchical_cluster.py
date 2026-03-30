import argparse
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
import pandas as pd
import warnings
import math
import numpy as np
import joblib
from scipy import stats
from dtaidistance import dtw
import itertools
from tqdm import tqdm
import os
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import squareform

warnings.filterwarnings("ignore")

def hierarchical_cluster(risk_profiles_directory, output_directory):
    years = ['2018', '2020']
    patients_2018 = ['559', '563', '570', '575', '588', '591']
    patients_2020 = ['540', '544', '552', '567', '584', '596']

    os.makedirs(output_directory, exist_ok=True)

    timeseries = joblib.load(risk_profiles_directory/"risk_profiles.pkl")

    numbers = []
    labels = []
    for i in range(len(patients_2018)):
        numbers.append(i)
        labels.append("A_" + str(i))
    for i in range(len(patients_2020)):
        numbers.append(i + len(patients_2018))
        labels.append("B_" + str(i))

    # --- Threshold settings ---
    threshold_orig = 4.5e7

    dist = math.inf
    for x in itertools.permutations(numbers):
        ts = []
        lb = []
        for j in range(len(x)):
            ts.append(timeseries[int(x[j])])
            lb.append(labels[int(x[j])])
        # --- DTW distance matrix ---
        ds = dtw.distance_matrix_fast(ts)
        # Ensure it's float (important for division)
        ds = ds.astype(float)

        # --- Min–max normalization ---
        d_min = np.min(ds)
        d_max = np.max(ds)

        if d_max > d_min:
            ds_norm = (ds - d_min) / (d_max - d_min)
        else:
            # Edge case: all distances identical
            ds_norm = np.zeros_like(ds)

        # --- Convert to condensed form ---
        ds = squareform(ds_norm)

        # --- Hierarchical clustering ---
        Z = linkage(ds, method='complete')

        # --- Plot dendrogram ---
        plt.figure(figsize=(10, 6))
        dendrogram(Z, labels=lb)#, truncate_mode='lastp', p=5)

        # Overlay all thresholds
        plt.axhline(y=threshold_orig, linestyle='--')

        plt.title("DTW + Complete Linkage Dendrogram with Threshold Sweep")
        plt.xlabel("Patient Index")
        plt.ylabel("DTW Distance")
        plt.savefig(output_directory/"Dendrogram.pdf")
        plt.close()

        # --- Original clustering ---
        clusters_orig = fcluster(Z, t=2, criterion='maxclust')
        original_cluster_id = 1
        less_vuln_orig = np.where(clusters_orig == original_cluster_id)[0]
        break

    AllPatientIDs = np.arange(0, 12)
    LessVulnerablePatientIDs = np.array(less_vuln_orig)
    MoreVulnerablePatientIDs = list(set(AllPatientIDs) - set(LessVulnerablePatientIDs))

    joblib.dump(AllPatientIDs, output_directory/"AllPatientIDs.pkl")
    joblib.dump(LessVulnerablePatientIDs, output_directory/"LessVulnerablePatientIDs.pkl")
    joblib.dump(MoreVulnerablePatientIDs, output_directory/"MoreVulnerablePatientIDs.pkl")

    print("All Patient IDs: ", end="")
    print(np.array(labels)[list(AllPatientIDs)])
    print("Less Vulnerable Patient IDs: ", end="")
    print(np.array(labels)[list(LessVulnerablePatientIDs)])
    print("More Vulnerable Patient IDs: ", end="")
    print(np.array(labels)[list(MoreVulnerablePatientIDs)])



if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Run OhioT1DM script: python cluster.py <risk_profiles_directory> <output_directory>",
        epilog="Example: python cluster.py output output/cluster_output"
    )
    parser.add_argument("risk_profiles_dir", nargs="?", default="output", help="Directory containing risk profiles")
    parser.add_argument("out_dir", nargs="?", default="output/cluster_output", help="Output directory")

    args = parser.parse_args()

    SCRIPT_DIR = Path(__file__).resolve().parent

    output_directory = SCRIPT_DIR / args.out_dir
    risk_profiles_directory = SCRIPT_DIR / args.risk_profiles_dir

    hierarchical_cluster(risk_profiles_directory, output_directory)