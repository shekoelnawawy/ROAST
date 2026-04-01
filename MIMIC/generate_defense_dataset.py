import joblib
import numpy as np
import os
import argparse
from pathlib import Path
from sklearn.model_selection import KFold, train_test_split
import random

def create_mixed_data(benign_data, adversarial_data, features):
    """Create mixed benign/adversarial data with random selection."""
    instances = []
    for i in range(len(benign_data)):
        for j in range(72):
            if (benign_data[i][j] != adversarial_data[i][j]).any():
                rand = random.randint(0, 1)
                if rand % 2 == 0:
                    instance = np.append(benign_data[i][j][features], 0)
                else:
                    instance = np.append(adversarial_data[i][j][features], 1)
            else:
                instance = np.append(benign_data[i][j][features], 0)
            instances.append(instance)
    return instances

def generate_defense_dataset(cluster_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    data_dir = Path(__file__).resolve().parent / "output"

    features = [143, 83, 89, 118, 183, 114, 177, 47, 75, 100]
    adversarial_data = joblib.load(data_dir / 'adversarial_data.pkl').reshape(-1,72, 432)
    benign_data = joblib.load(data_dir / 'benign_data.pkl').reshape(-1,72, 432)

    # Create mixed data with random selection
    instances = create_mixed_data(benign_data, adversarial_data, features)
    data = np.array(instances).reshape(-1, 72, len(features) + 1)

    # Create benign-only data for 'all'
    benign_instances = []
    for i in range(len(benign_data)):
        for j in range(72):
            instance = np.append(benign_data[i][j][features], 0)
            benign_instances.append(instance)

    data_benign = np.array(benign_instances).reshape(-1, 72, len(features) + 1)

    LessVulnerablePatientIDs = joblib.load(cluster_dir / 'LessVulnerablePatientIDs.pkl')
    MoreVulnerablePatientIDs = joblib.load(cluster_dir / 'MoreVulnerablePatientIDs.pkl')
    AllPatientIDs = joblib.load(cluster_dir / 'AllPatientIDs.pkl')

    np.save(out_dir / 'mimic_train_less_0.npy', data[LessVulnerablePatientIDs].reshape(-1,data.shape[2]))
    np.save(out_dir / 'mimic_train_more_0.npy', data[MoreVulnerablePatientIDs].reshape(-1,data.shape[2]))
    np.save(out_dir / 'mimic_train_all_0.npy', data_benign.reshape(-1,data_benign.shape[2]))

    for run in range(5):

        split = train_test_split(AllPatientIDs, train_size=len(LessVulnerablePatientIDs))
        train_indices = split[0]
        # test_indices = split[1][:math.floor(0.2 * len(AllPatientIDs))]

        np.save(out_dir / f'mimic_train_samples_{run}.npy', data[train_indices].reshape(-1, data.shape[2]))


    cv=0
    kf = KFold(n_splits=5)
    for train_indices, test_indices in kf.split(AllPatientIDs):
        np.save(out_dir / f'mimic_test_all_{cv}.npy', data[test_indices].reshape(-1, data.shape[2]))
        cv+=1


if __name__ == '__main__':
	parser = argparse.ArgumentParser(
		description="Run MIMIC script: python generate_defense_dataset.py <cluster_output> <output_directory>",
		epilog="Example: python generate_defense_dataset.py output/cluster_output output/defense_dataset"
	)
	parser.add_argument("cluster_dir", nargs="?", default="output/cluster_output", help="Directory containing cluster output")
	parser.add_argument("out_dir", nargs="?", default="output/defense_dataset", help="Output directory")

	args = parser.parse_args()

	SCRIPT_DIR = Path(__file__).resolve().parent

	cluster_directory = SCRIPT_DIR / args.cluster_dir
	output_directory = SCRIPT_DIR / args.out_dir

	generate_defense_dataset(cluster_directory, output_directory)