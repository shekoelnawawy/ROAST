import joblib
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import argparse
from sklearn.model_selection import KFold, train_test_split
import random

# Create mixed benign/adversarial data for less, more, samples
def create_mixed_data(data_benign, data_adversarial):
    mixed_data = []
    # Feature columns (all except PatientID and Adversarial)
    feature_columns = [col for col in data_benign.columns if col != 'PatientID']

    for idx in range(len(data_benign)):
        benign_row = data_benign.iloc[idx][feature_columns].to_numpy()
        adversarial_row = data_adversarial.iloc[idx][feature_columns].to_numpy()

        if not np.array_equal(benign_row, adversarial_row):
            rand = random.randint(0, 1)
            if rand % 2 == 0:
                mixed_data.append(np.append(benign_row, 0))
            else:
                mixed_data.append(np.append(adversarial_row, 1))
        else:
            mixed_data.append(np.append(benign_row, 0))
    return np.array(mixed_data)

def get_benign_and_adversarial_data(data_dir):
    training_sets = ["training_setA", "training_setB"]

    # get feature names
    data_psv_dir = Path(__file__).resolve().parent / "input"
    psv_candidates = sorted(data_psv_dir.glob("*/*.psv"))
    if not psv_candidates:
        raise FileNotFoundError(
            f"Could not find *.psv in {data_psv_dir}"
        )
    psv_path = psv_candidates[0]

    with open(psv_path) as file_obj:
        header = file_obj.readline().strip()
        features = np.array(header.split('|')[:-1])

    benign_data_df = pd.DataFrame()
    adversarial_data_df = pd.DataFrame()

    for training_set in training_sets:
        benign_data_path = data_dir / training_set / "Data" / "Benign"
        adversarial_data_path = data_dir / training_set / "Data" / "Adversarial"

        for data_file in tqdm(Path(benign_data_path).glob("*.pkl")):
            benign_data = joblib.load(benign_data_path / data_file.name)
            adversarial_data = joblib.load(adversarial_data_path / data_file.name)

            benign_patient_df = pd.DataFrame(benign_data)
            benign_patient_df.columns = features
            benign_patient_df.insert(loc=0, column='PatientID', value=data_file.name[:-4])
            benign_data_df = pd.concat([benign_data_df, benign_patient_df], axis=0, ignore_index=True)

            adversarial_patient_df = pd.DataFrame(adversarial_data)
            adversarial_patient_df.columns = features
            adversarial_patient_df.insert(loc=0, column='PatientID', value=data_file.name[:-4])
            adversarial_data_df = pd.concat([adversarial_data_df, adversarial_patient_df], axis=0, ignore_index=True)


    # pre-processing
    selected_features = ['PatientID', 'HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'Age', 'Gender']
    benign_data_df = benign_data_df.ffill()
    benign_data_df = benign_data_df.fillna(0)

    benign_data = benign_data_df.loc[:, selected_features]
    adversarial_data = adversarial_data_df.loc[:, selected_features]

    # Compare row by row
    row_diff = (benign_data != adversarial_data).any(axis=1)
    adversarial_data.insert(len(adversarial_data.columns), "Adversarial", row_diff)

    joblib.dump(benign_data, data_dir / "benign_data.pkl")
    joblib.dump(adversarial_data, data_dir / "adversarial_data.pkl")


def generate_defense_dataset(cluster_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    data_dir = Path(__file__).resolve().parent / "output"

    file_path_benign = Path(data_dir / 'benign_data.pkl')
    file_path_adversarial = Path(data_dir / 'adversarial_data.pkl')

    if not file_path_benign.exists() or not file_path_adversarial.exists():
        get_benign_and_adversarial_data(data_dir)

    AllPatientsDataBenign = joblib.load(file_path_benign)
    AllPatientsDataAdversarial = joblib.load(file_path_adversarial)

    AllPatientIDs = joblib.load(cluster_dir / 'AllPatientIDs.pkl')
    MoreVulnerablePatientIDs = joblib.load(cluster_dir / 'MoreVulnerablePatientIDs.pkl')
    LessVulnerablePatientIDs = joblib.load(cluster_dir / 'LessVulnerablePatientIDs.pkl')

    # Create mixed data with random selection
    all_mixed_data = create_mixed_data(AllPatientsDataBenign, AllPatientsDataAdversarial)

    # Filter by patient IDs for less and more
    benign_less_indices = AllPatientsDataBenign[AllPatientsDataBenign['PatientID'].isin(LessVulnerablePatientIDs)].index
    benign_more_indices = AllPatientsDataBenign[AllPatientsDataBenign['PatientID'].isin(MoreVulnerablePatientIDs)].index

    mixed_less = all_mixed_data[benign_less_indices]
    mixed_more = all_mixed_data[benign_more_indices]

    # less, more, samples use mixed data (with random selection)
    np.save(out_dir / 'sepsis_train_less_0.npy', mixed_less)
    np.save(out_dir / 'sepsis_train_more_0.npy', mixed_more)

    # all uses benign data only
    np.save(out_dir / 'sepsis_train_all_0.npy', AllPatientsDataBenign.drop(columns=['PatientID']).to_numpy().astype(float))

    for run in range(5):

        split = train_test_split(AllPatientIDs, train_size=len(LessVulnerablePatientIDs))
        train_indices = split[0]
        # test_indices = split[1][:math.floor(0.2 * len(AllPatientIDs))]

        train_mask = AllPatientsDataBenign['PatientID'].isin(train_indices)
        mixed_samples = all_mixed_data[train_mask.to_numpy()]
        np.save(out_dir / f'sepsis_train_samples_{run}.npy', mixed_samples)


    cv=0
    kf = KFold(n_splits=5)
    for train_indices, test_indices in kf.split(AllPatientIDs):
        test_mask = AllPatientsDataAdversarial['PatientID'].isin([AllPatientIDs[i] for i in test_indices])
        np.save(out_dir / f'sepsis_test_all_{cv}.npy', AllPatientsDataAdversarial[test_mask].drop(columns=['PatientID']).to_numpy().astype(float))
        cv+=1


if __name__ == '__main__':
	parser = argparse.ArgumentParser(
		description="Run PhysioNetCinC script: python generate_defense_dataset.py <cluster_output> <output_directory>",
		epilog="Example: python generate_defense_dataset.py output/cluster_output output/defense_dataset"
	)
	parser.add_argument("cluster_dir", nargs="?", default="output/cluster_output", help="Directory containing cluster output")
	parser.add_argument("out_dir", nargs="?", default="output/defense_dataset", help="Output directory")

	args = parser.parse_args()

	SCRIPT_DIR = Path(__file__).resolve().parent

	cluster_directory = SCRIPT_DIR / args.cluster_dir
	output_directory = SCRIPT_DIR / args.out_dir

	generate_defense_dataset(cluster_directory, output_directory)