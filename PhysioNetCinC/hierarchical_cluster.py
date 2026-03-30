import math
import argparse
from pathlib import Path
import pandas as pd
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
from sklearn.metrics import silhouette_score

def jaccard_index(a, b):
    a_set, b_set = set(a), set(b)
    return len(a_set & b_set) / len(a_set | b_set)

def get_less_vulnerable_cluster(clusters, reference_indices):
    """
    Automatically find cluster most overlapping with original less-vulnerable set
    """
    unique_clusters = np.unique(clusters)
    best_cluster = None
    best_score = 0

    for c in unique_clusters:
        indices = np.where(clusters == c)[0]
        score = jaccard_index(reference_indices, indices)
        if score > best_score:
            best_score = score
            best_cluster = indices

    return best_cluster, best_score

def hierarchical_cluster(risk_profiles_directory, output_directory):
    os.makedirs(output_directory, exist_ok=True)
    records = joblib.load(risk_profiles_directory/"risk_profiles.pkl")

    # Each record is expected as: [patient_label (str), risk_profile (list)]
    labels = [record[0] for record in records]
    timeseries = [record[1] for record in records]

    numbers = []
    for i in range(len(timeseries)):
        numbers.append(i)

    i=0
    dist = math.inf

    for x in itertools.permutations(numbers):
        ts = []
        lb = []
        for j in range(len(x)):
            ts.append(np.array(timeseries[int(x[j])]))
            lb.append(labels[int(x[j])])

        # --- DTW distance matrix ---
        ds = dtw.distance_matrix_fast(ts)

        # --- Hierarchical clustering ---
        Z = linkage(ds, method='complete')

        # --- Threshold settings ---
        threshold_orig = 50
        scales = [1.0] #[0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15]
        thresholds = [threshold_orig * s for s in scales]

        # --- Plot dendrogram ---
        plt.figure(figsize=(10, 6))
        dendrogram(Z, labels=lb, truncate_mode='lastp', p=5)


        # Overlay all thresholds
        for t in thresholds:
            plt.axhline(y=t, linestyle='--')

        plt.title("DTW + Complete Linkage Dendrogram with Threshold Sweep")
        plt.xlabel("Patient Index")
        plt.ylabel("DTW Distance")
        plt.savefig(output_directory/"Dendrogram.pdf")
        plt.close()

        # --- Original clustering ---
        clusters_orig = fcluster(Z, t=threshold_orig, criterion='distance')
        original_cluster_id = 1
        less_vuln_orig = np.where(clusters_orig == original_cluster_id)[0]
        break

        # results = {}

        # for scale in scales:
        #     t_new = threshold_orig * scale
        #     clusters_new = fcluster(Z, t=t_new, criterion='distance')

        #     matched_cluster, j_score = get_less_vulnerable_cluster(clusters_new, less_vuln_orig)

        #     # silhouette = silhouette_score(ds, clusters_new, metric='precomputed')

        #     results[scale] = {
        #         "threshold": t_new,
        #         "jaccard": j_score,
        #         # "silhouette": silhouette,
        #         "cluster_size": len(matched_cluster)
        #     }

        # print(results)

        # i += 1
    
    AllPatientIDs = np.arange(0, len(timeseries))
    LessVulnerablePatientIDs = np.array(less_vuln_orig)
    MoreVulnerablePatientIDs = np.array(list(set(AllPatientIDs) - set(LessVulnerablePatientIDs)))

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
        description="Run PhysioNetCinC script: python hierarchical_cluster.py <risk_profiles_directory> <output_directory>",
        epilog="Example: python hierarchical_cluster.py output output/cluster_output"
    )
    parser.add_argument("risk_profiles_dir", nargs="?", default="output", help="Directory containing risk profiles")
    parser.add_argument("out_dir", nargs="?", default="output/cluster_output", help="Output directory")

    args = parser.parse_args()

    SCRIPT_DIR = Path(__file__).resolve().parent

    output_directory = SCRIPT_DIR / args.out_dir
    risk_profiles_directory = SCRIPT_DIR / args.risk_profiles_dir

    hierarchical_cluster(risk_profiles_directory, output_directory)