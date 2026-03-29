import xml.etree.ElementTree as etree
import pandas as pd
import joblib
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

def preprocess_data(raw_data_dir, processed_data_dir):
    os.makedirs(processed_data_dir, exist_ok=True)

    columns = ['glucose', 'finger', 'basal', 'hr', 'gsr', 'carbs', 'temp_basal',
            'dose', 'bwz_carb_input']

    xmlkeys = ["glucose_level", "finger_stick", "basal", "basis_heart_rate", "basis_gsr",
            "meal", "temp_basal", "bolus", ]

    dict = {}
    
    for dir in os.listdir(raw_data_dir):
        data_dir = Path(raw_data_dir)/dir

        for fff in data_dir.iterdir():  # iterates over all entries
            fff = str(fff)

            if not fff.endswith('.xml'):
                continue
            tree = etree.parse(fff)
            finaltime = []
            # loop through outpus
            num = 0
            for x in xmlkeys:
                time = []
                val = []
                val2 = []
                time2 = []
                rawtime = []
                # loop through instancies
                for f in tree.iter(x):
                    # actual instances loop

                    for g in f:
                        # divide time by 300 to get 5 minute intervals
                        if num < 5:
                            val.append(float(g.items()[1][1]))
                            time.append(pd.to_datetime(g.items()[0][1], dayfirst=True).timestamp() / 300)
                            rawtime.append(g.items()[0][1])
                        if num == 5:
                            val.append(float(g.items()[2][1]))
                            time.append(pd.to_datetime(g.items()[0][1], dayfirst=True).timestamp() / 300)
                        if num == 6:
                            val.append(float(g.items()[2][1]))
                            time.append(pd.to_datetime(g.items()[0][1], dayfirst=True).timestamp() / 300)
                            time2.append(pd.to_datetime(g.items()[1][1], dayfirst=True).timestamp() / 300)
                        if num == 7:
                            val.append(float(g.items()[3][1]))
                            time.append(pd.to_datetime(g.items()[0][1], dayfirst=True).timestamp() / 300)
                            time2.append(pd.to_datetime(g.items()[1][1], dayfirst=True).timestamp() / 300)
                if len(time) == 0:
                    if num == 6:
                        num = num + 1
                        continue;
                time = np.array(time)
                val = np.array(val)
                sorter = np.argsort(time)
                time = time[sorter]
                val = val[sorter]

                if num > 5:
                    time2 = np.array(time2)
                    time2 = time2[sorter]

                # get basetime
                if num == 0:
                    if 'test' in fff:
                        joblib.dump(rawtime, Path(processed_data_dir) / f"{Path(fff).name[:3]}.timestamps.pkl")

                    # Nawawy's start
                    basetime = np.linspace(int(time.copy()[0]), int(time.copy()[-1] + 1),
                                        int(time.copy()[-1] + 1 - time.copy()[0]))  # -time.copy()[0]
                    # Nawawy's end

                    dict[''] = basetime
                    zerotime = time.copy()[0]
                    out = np.array(val)
                # do interpolation
                time = np.array(time) - zerotime
                val = np.array(val)
                out = np.full(len(basetime), np.nan)
                # for basal and basal 0s, use carry forward imputation
                if num == 2:
                    for i in range(len(time)):
                        if int(time[i]) < len(basetime):
                            out[int(time[i]):] = val[i]
                # basal 0s just shows when the pump is off so update basal array
                elif num == 6:
                    out = dict['basal']
                    time2 = np.array(time2) - zerotime
                    for i in range(len(time)):
                        if int(time[i]) < len(basetime):
                            out[int(time[i]):int(time2[i])] = val[i]
                # For other variables, just put each value at the closest 5 minute time point.
                else:
                    for i in range(len(time)):
                        if int(time[i]) < len(basetime):
                            out[int(time[i])] = val[i]

                # add to dictionary
                if num == 6:
                    dict['basal'] = out
                else:
                    dict[columns[num]] = out
                # move onto next.
                num = num + 1

            # save data frame
            df = pd.DataFrame(dict)
            df.set_index('')

            # Nawawy's start
            df.insert(len(df.columns), "postprandial", False)
            for i in range(len(df)):
                if df.loc[i, 'carbs'] > 0:
                    k = i
                    while k < i + 24 and k < len(
                            df):  # postprandial is two hours after meal so 24 stands for 24 intervals of 5 minutes (i.e., 2 hours)
                        df.loc[k, 'postprandial'] = True
                        k += 1
            # Nawawy's end

            if 'test' in fff:
                df.to_pickle(Path(processed_data_dir) / f"{Path(fff).name[:3]}.test.pkl")
            if 'train' in fff:
                df.to_pickle(Path(processed_data_dir) / f"{Path(fff).name[:3]}.train.pkl")



if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Pre-process raw CGM data into usable format",
        epilog="Example: python convert_data.py data/raw data/processed"
    )
    parser.add_argument("raw_data_dir", nargs="?", default=Path("data/raw"), help="Directory containing raw dataset")
    parser.add_argument("processed_data_dir", nargs="?", default=Path("data/processed"), help="Directory where processed dataset will be stored")

    args = parser.parse_args()
    
    SCRIPT_DIR = Path(__file__).resolve().parent

    preprocess_data(SCRIPT_DIR/args.raw_data_dir/"2018", SCRIPT_DIR/args.processed_data_dir/"2018data")
    preprocess_data(SCRIPT_DIR/args.raw_data_dir/"2020", SCRIPT_DIR/args.processed_data_dir/"2020data")