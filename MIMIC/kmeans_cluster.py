import csv
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
import joblib
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import argparse
from pathlib import Path
import warnings
import os
warnings.filterwarnings("ignore")


def kmeans_cluster(risk_profiles_directory, output_directory):
    os.makedirs(output_directory, exist_ok=True)

    adversarial_output = joblib.load(risk_profiles_directory/"adversarial_output.pkl")
    benign_output = joblib.load(risk_profiles_directory/"target_output.pkl")

    df = pd.DataFrame(joblib.load(risk_profiles_directory/"risk_profiles.pkl"))
    
    most_vulnerable = []
    for i in range(len(benign_output)):
        if adversarial_output[i] >= 0.5 and benign_output[i] < 0.5:
            most_vulnerable.append(i)

    model = KMeans(n_clusters=2)
    model.fit(df)
    predictions = model.predict(df)

    # Get the cluster centroids
    cluster_centers = model.cluster_centers_

    # Calculate the Euclidean distance between all pairs of centroids
    # cdist returns a distance matrix where element (i, j) is the distance between centroid i and centroid j
    distances_between_centroids = cdist(cluster_centers, cluster_centers, metric='euclidean')

    print('Cluster Centroids:\n', cluster_centers)
    print('\nDistance Matrix between Centroids:\n', distances_between_centroids)
    print('\nCluster Labels:')
    print(model.labels_)


    clusterA = []
    clusterB = []
    for i in range(len(predictions)):
        if predictions[i] == 0:
            clusterA.append(i)
        else:
            clusterB.append(i)


    countA = 0
    countB = 0
    for i in range(len(most_vulnerable)):
        if most_vulnerable[i] in clusterA:
            countA += 1
        elif most_vulnerable[i] in clusterB:
            countB += 1


    print('Cluster A Patients: '+str(len(clusterA)))
    print('Cluster B Patients: '+str(len(clusterB)))
    print('Most Vulnerable in Cluster A: '+str(countA))
    print('Most Vulnerable in Cluster B: '+str(countB))
    print('Percentage of Most Vulnerable in Cluster A: '+ str((countA/(countA+countB))*100))
    print('Percentage of Most Vulnerable in Cluster B: '+ str((countB/(countA+countB))*100))

    if countA > countB:
        joblib.dump(np.array(clusterA), output_directory/"MoreVulnerablePatientIDs.pkl")
        joblib.dump(np.array(clusterB), output_directory/"LessVulnerablePatientIDs.pkl")
    else:
        joblib.dump(np.array(clusterA), output_directory/"LessVulnerablePatientIDs.pkl")
        joblib.dump(np.array(clusterB), output_directory/"MoreVulnerablePatientIDs.pkl")

    all_patient_ids = sorted(clusterA + clusterB)
    joblib.dump(np.array(all_patient_ids), output_directory/"AllPatientIDs.pkl")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Run MIMIC script: python kmeans_cluster.py <risk_profiles_directory> <output_directory>",
        epilog="Example: python kmeans_cluster.py output output/cluster_output"
    )
    parser.add_argument("risk_profiles_dir", nargs="?", default="output", help="Directory containing risk profiles")
    parser.add_argument("out_dir", nargs="?", default="output/cluster_output", help="Output directory")

    args = parser.parse_args()

    SCRIPT_DIR = Path(__file__).resolve().parent

    output_directory = SCRIPT_DIR / args.out_dir
    risk_profiles_directory = SCRIPT_DIR / args.risk_profiles_dir

    kmeans_cluster(risk_profiles_directory, output_directory)