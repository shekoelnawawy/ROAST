import argparse
from pathlib import Path

from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
import joblib
from sklearn.cluster import KMeans
import os
import warnings
from tqdm import tqdm
from scipy.spatial.distance import cdist

warnings.filterwarnings("ignore")


def calculate_percentage_mispredictions(risk_profiles_directory, output_directory):
    training_sets = ['training_setA', 'training_setB']

    benign_files = []
    adversarial_files = []

    percentage_mispredictions = open(output_directory/"percentage_mispredictions.csv", 'w')
    percentage_mispredictions.write('TrainingSet,PatientID,PercentageMisprediction\n')

    for training_set in training_sets:
        benign_path = risk_profiles_directory/training_set/"Predictions"/"Benign"
        adversarial_path = risk_profiles_directory/training_set/"Predictions"/"Adversarial"
        for f in os.listdir(benign_path):
            if os.path.isfile(benign_path/f) and not f.lower().startswith('.') and f.lower().endswith('psv'):
                output_benign = open(benign_path/f, 'r')
                header = output_benign.readline().strip()
                column_names = header.split('|')
                target_benign = np.loadtxt(output_benign, delimiter='|')
            else:
                raise Exception('Benign output file does not exist!')

            if os.path.isfile(adversarial_path/f) and not f.lower().startswith('.') and f.lower().endswith('psv'):
                output_adversarial = open(adversarial_path/f, 'r')
                header = output_adversarial.readline().strip()
                target_adversarial = np.loadtxt(output_adversarial, delimiter='|')
            else:
                raise Exception('Adversarial output file does not exist!')

            count_differences = 0
            for i in range(len(target_benign)):
                if target_benign[i][1] != target_adversarial[i][1]:
                    count_differences += 1
            percentage = (count_differences/len(target_benign))*100
            percentage_mispredictions.write(training_set[-1]+','+f[:-4]+','+f"{percentage:.2f}"+'\n')

    percentage_mispredictions.close()


def kmeans_cluster(risk_profiles_directory, output_directory):
    os.makedirs(output_directory, exist_ok=True)
    risk_profiles = joblib.load(risk_profiles_directory/"risk_profiles.pkl")

    unique_PatientIDs = np.arange(len(risk_profiles))#[item[0] for item in risk_profiles]
    df = pd.DataFrame([item[1] for item in risk_profiles]).ffill(axis=1)
    model = KMeans(n_clusters=2, verbose = 1)
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

    calculate_percentage_mispredictions(risk_profiles_directory, output_directory)
    mispredictions = pd.read_csv(output_directory/"percentage_mispredictions.csv")


    most_vulnerable_threshold = 20
    real_patient_ids = [item[0] for item in risk_profiles]
    most_vulnerable = mispredictions[mispredictions['PercentageMisprediction']>most_vulnerable_threshold]['PatientID'].tolist()
    most_vulnerable = [real_patient_ids.index(str(pid)) for pid in most_vulnerable]

    clusterA = []
    clusterB = []
    for i in range(len(predictions)):
        if predictions[i] == 0:
            clusterA.append(unique_PatientIDs[i])
        else:
            clusterB.append(unique_PatientIDs[i])

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
        description="Run PhysioNetCinC script: python kmeans_cluster.py <risk_profiles_directory> <output_directory>",
        epilog="Example: python kmeans_cluster.py output output/cluster_output"
    )
    parser.add_argument("risk_profiles_dir", nargs="?", default="output", help="Directory containing risk profiles")
    parser.add_argument("out_dir", nargs="?", default="output/cluster_output", help="Output directory")

    args = parser.parse_args()

    SCRIPT_DIR = Path(__file__).resolve().parent

    output_directory = SCRIPT_DIR / args.out_dir
    risk_profiles_directory = SCRIPT_DIR / args.risk_profiles_dir

    kmeans_cluster(risk_profiles_directory, output_directory)