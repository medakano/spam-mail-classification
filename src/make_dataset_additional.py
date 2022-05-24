import glob
import os
import pandas as pd

from tqdm import tqdm

# Get dataset from https://www.kaggle.com/code/owenpatrickfalculan/spam-email-classification/data
# I used it for additional training data, however the score was not improved.
# make_dataset.py was used for my best prediction.

def main():
    # read labels of train data
    df_labels = pd.read_csv("../data/original/train_master.tsv", sep="\t", index_col=0)
    labels = df_labels["label"]

    # read texts of the original train data
    df = pd.DataFrame(columns=["sentence", "label"])
    train_files = glob.glob('../data/original/train/*')
    for i, file in tqdm(enumerate(train_files), total=len(labels)):
        with open(os.path.join(file), "r") as f:
            sentence = f.read()
            sentence = sentence.replace("\n", "")
        df = pd.concat([df, pd.DataFrame({"sentence":sentence, "label":labels[i]}, index=[i])])

    # read additional data
    df_spamassasin = pd.read_csv("../data/additional_data/completeSpamAssassin.csv", index_col=0)
    df_lingspam = pd.read_csv("../data/additional_data/lingSpam.csv", index_col=0)
    df_additional = pd.concat([df_spamassasin, df_lingspam])
    df_additional = df_additional.astype(str)
    remove_indention = lambda x: '"' + x.replace("\n", "") + '"' # ValueError: text input must of type `str` (single example) 対策で " を追加
    df_additional["Body"] = df_additional["Body"].map(remove_indention)

    # concat data
    df_additional = df_additional.loc[:, ["Body","Label"]]
    df_additional = df_additional.rename(columns={"Body":"sentence", "Label":"label"})
    df = pd.concat([df, df_additional])
    df = df.sample(frac=1) # shuffle

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
