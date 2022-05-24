import glob
import os
import pandas as pd
from tqdm import tqdm

def main():
    prediction_files = glob.glob('../predictions/5fold/*/predict_results_None.txt')
    for i, file in tqdm(enumerate(prediction_files)):
        print(file)
        df_pred = pd.read_csv(file, sep="\t")
        if i == 0:
            df_merged = df_pred
        else:
            df_merged["logit_0"] = df_merged["logit_0"] + df_pred["logit_0"]
            df_merged["logit_1"] = df_merged["logit_1"] + df_pred["logit_1"]
    df_merged = df_merged.drop("index", axis=1)
    with open('../predictions/5fold_prediction.csv', 'w') as f:
        for i, pred in enumerate(df_merged.idxmax(axis=1)):
            f.write(f'test_{str(i).zfill(4)}.txt,{pred[-1]}\n')

if __name__ == "__main__":
    main()
