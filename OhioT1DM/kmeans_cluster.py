import pandas as pd
import joblib
import numpy as np
import glob
import os
from pathlib import Path
from sklearn import preprocessing
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import argparse
from pathlib import Path
from tslearn.clustering import TimeSeriesKMeans

def calculate_percentage_mispredictions(risk_profiles_directory, output_directory):
    percentage_mispredictions = open(output_directory/"percentage_mispredictions.csv", 'w')
    percentage_mispredictions.write('TrainingSet,PatientID,PercentageMisprediction\n')
    
    years = ['2018', '2020']
    patients_2018 = ['559', '563', '570', '575', '588', '591']
    patients_2020 = ['540', '544', '552', '567', '584', '596']

    
    for year in years:
        if year == '2018':
            patients = patients_2018
        else:
            patients = patients_2020
        for patient in patients:
            try:
                adversarial_output_path = risk_profiles_directory/year/patient/"predicted_output.pkl"
                benign_output_path = risk_profiles_directory/year/patient/"actual_output.pkl"
                adversarial_output = joblib.load(adversarial_output_path)
                benign_output = joblib.load(benign_output_path)
            except FileNotFoundError:
                print(f"{year} results not available. Skipping...")
                continue
            adversarial_output = adversarial_output.flatten()
            benign_output = benign_output.flatten()
            


            target_values = 0
            mispredictions = 0
            for i in range(len(benign_output)):
                if 70 < benign_output[i] < 180:
                    target_values += 1
                    if adversarial_output[i] > 180:
                        mispredictions += 1
                elif benign_output[i] < 70:
                    target_values += 1
                    if adversarial_output[i] > 180:
                        mispredictions += 1

            percentage_mispredictions.write(year+','+patient+','+f"{(mispredictions/target_values)*100:.2f}"+'\n')
            
    percentage_mispredictions.close()

def kmeans_cluster(risk_profiles_directory, output_directory):
    os.makedirs(output_directory, exist_ok=True)
    risk_profiles = joblib.load(risk_profiles_directory/"risk_profiles.pkl")
    patients = ['559', '563', '570', '575', '588', '591', '540', '544', '552', '567', '584', '596']
    unique_PatientIDs = np.arange(len(risk_profiles))

    ts = [x[1] for x in risk_profiles]

    model = TimeSeriesKMeans(
        n_clusters=2,
        metric="dtw",
        verbose=True
    )

    predictions = model.fit_predict(ts)

    

    # Get the cluster centroids
    # cluster_centers = model.cluster_centers_

    # Calculate the Euclidean distance between all pairs of centroids
    # cdist returns a distance matrix where element (i, j) is the distance between centroid i and centroid j
    # distances_between_centroids = cdist(cluster_centers, cluster_centers, metric='euclidean')

    # print('Cluster Centroids:\n', cluster_centers)
    # print('\nDistance Matrix between Centroids:\n', distances_between_centroids)
    # print('\nCluster Labels:')
    # print(model.labels_)

    calculate_percentage_mispredictions(risk_profiles_directory, output_directory)
    mispredictions = pd.read_csv(output_directory/"percentage_mispredictions.csv")


    most_vulnerable_threshold = 20
    most_vulnerable = mispredictions[mispredictions['PercentageMisprediction']>most_vulnerable_threshold]['PatientID'].tolist()
    most_vulnerable = [patients.index(str(pid)) for pid in most_vulnerable]

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
        description="Run OhioT1DM script: python kmeans_cluster.py <risk_profiles_directory> <output_directory>",
        epilog="Example: python kmeans_cluster.py output output/cluster_output"
    )
    parser.add_argument("risk_profiles_dir", nargs="?", default="output", help="Directory containing risk profiles")
    parser.add_argument("out_dir", nargs="?", default="output/cluster_output", help="Output directory")

    args = parser.parse_args()

    SCRIPT_DIR = Path(__file__).resolve().parent

    output_directory = SCRIPT_DIR / args.out_dir
    risk_profiles_directory = SCRIPT_DIR / args.risk_profiles_dir

    kmeans_cluster(risk_profiles_directory, output_directory)