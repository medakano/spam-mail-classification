import glob
import os
import pandas as pd

from tqdm import tqdm

# convert original train data into GLUE format

def main():
    # read labels of train data
    df_labels = pd.read_csv("../data/original/train_master.tsv", sep="\t", index_col=0)
    labels = df_labels["label"]

    # read texts of train data
    df = pd.DataFrame(columns=["sentence", "label"])
    train_files = glob.glob('../data/original/train/*')
    for i, file in tqdm(enumerate(train_files), total=len(labels)):
        with open(os.path.join(file), "r") as f:
            sentence = f.read()
            sentence = sentence.replace("\n", "")
        df = pd.concat([df, pd.DataFrame({"sentence":sentence, "label":labels[i]}, index=[i])])
    df = df.sample(frac=1, random_state=0) # shuffle

    # Cross-Validation k fold
    k = 5
    size = int(len(labels)/k)
    for i in range(k):
        save_path = os.path.join("../data/dataset_5fold", f"fold_{i+1}")
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        dev_start = i*size
        dev_end   = (i+1)*size
        df[dev_start:dev_end].to_csv(os.path.join(save_path, "dev.csv"), sep=",", index=False)
        pd.concat([df[:dev_start], df[dev_end:]]).to_csv(os.path.join(save_path, "train.csv"), sep=",", index=False)

if __name__ == "__main__":
    main()
