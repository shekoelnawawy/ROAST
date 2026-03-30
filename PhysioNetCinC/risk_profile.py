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
import csv
from scipy.spatial.distance import cdist

warnings.filterwarnings("ignore")


def risk_profile(output_directory):
    training_sets = ['training_setA', 'training_setB']
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
    adversarial_predictions_df = pd.DataFrame()

    for training_set in training_sets:
        benign_data_path = output_directory/training_set/"Data"/"Benign"
        adversarial_predictions_path = output_directory/training_set/"Predictions"/"Adversarial"
        for f in tqdm(os.listdir(benign_data_path)):
            if os.path.isfile(benign_data_path/f) and not f.lower().startswith('.') and f.lower().endswith('pkl'):
                benign_data = joblib.load(benign_data_path/f)
                per_patient_df = pd.DataFrame(benign_data)
                per_patient_df.columns = features
                per_patient_df.insert(loc=0, column='PatientID', value=f[:-4])
                benign_data_df = pd.concat([benign_data_df, per_patient_df], axis=0, ignore_index=True)
            else:
                raise Exception('Benign data file does not exist!')

            predictions_f = f[:-4] + '.psv'
            if os.path.isfile(adversarial_predictions_path/predictions_f) and not predictions_f.lower().startswith('.') and predictions_f.lower().endswith('psv'):
                adversarial_predictions_file = open(adversarial_predictions_path/predictions_f, 'r')
                header = adversarial_predictions_file.readline().strip()
                per_patient_df = pd.DataFrame(np.loadtxt(adversarial_predictions_file, delimiter='|')[:, 1])
                adversarial_predictions_df = pd.concat([adversarial_predictions_df, per_patient_df], axis=0, ignore_index=True)
            else:
                raise Exception('Adversarial prediction file does not exist!')

    # pre-processing
    PatientIDs = benign_data_df['PatientID']
    selected_features = ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'Age', 'Gender']
    benign_data_df = benign_data_df.ffill()
    benign_data_df = benign_data_df.fillna(0)
    benign_data = np.array(benign_data_df.loc[:, selected_features])
    benign_data = preprocessing.normalize(benign_data)
    adversarial_output = np.array(adversarial_predictions_df)

    # logistic regression for feature importance
    # define dataset
    X = benign_data
    y = adversarial_output
    # define the model
    model = LogisticRegression()
    # fit the model
    model.fit(X, y)
    # get importance
    importance = model.coef_[0]

    timeseries = importance * benign_data

    risk_profiles = []
    risk_scores = []
    old_PatientID = PatientIDs[0]
    for j in range(len(timeseries)):
        if PatientIDs[j] == old_PatientID:
            risk_scores.append(float(sum(timeseries[j,:])))
            if j == len(timeseries)-1:
                risk_profiles.append((old_PatientID, risk_scores))
        else:
            risk_profiles.append((old_PatientID, risk_scores))
            old_PatientID = PatientIDs[j]
            risk_scores = []
            risk_scores.append(float(sum(timeseries[j,:])))


    joblib.dump(risk_profiles, output_directory/'risk_profiles.pkl')



if __name__ == '__main__':
	parser = argparse.ArgumentParser(
		description="Run MIMIC script: python risk_profile.py <output_directory>",
		epilog="Example: python risk_profile.py output"
	)
	parser.add_argument("out_dir", nargs="?", default="output", help="Output directory")

	args = parser.parse_args()

	SCRIPT_DIR = Path(__file__).resolve().parent

	output_directory = SCRIPT_DIR / args.out_dir

	risk_profile(output_directory)