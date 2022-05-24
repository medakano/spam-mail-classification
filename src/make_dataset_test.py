import glob
import os
import pandas as pd
from tqdm import tqdm

# convert original test data into GLUE format

def main():
    df = pd.DataFrame(columns=["sentence"])
    test_files = glob.glob('../data/original/test/*')
    for i, file in tqdm(enumerate(test_files)):
        with open(os.path.join(file), "r") as f:
            sentence = f.read()
            sentence = sentence.replace("\n", "")
        df = pd.concat([df, pd.DataFrame({"sentence":sentence}, index=[i])])
    # dummy labels
    # Because an error occurs when there is only one label type,
    # I wrote this uncool code.
    df["label"] = "0"
    df["label"][0] = "1"

    df.to_csv("../data/test.csv", index=False)

if __name__ == "__main__":
    main()
