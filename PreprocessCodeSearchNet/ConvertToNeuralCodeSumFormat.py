""" Preprocessing CodeSearchNet data for use with our NeuralCodeSum.
NB: Run Preprocess.py first.

Author: Jesse Phillips <j.m.phillips@lancaster.ac.uk>
"""
import json


def generateData(dir: str) -> None:
    """ Saves data from our json in NeuralCodeSum's format.
    Args:
        dir (string): the name of the directory holding the dataset.
    """

    print(f"Generating Final Dataset: {dir}")
    with open(f"{dir}/pl_dataset.json", 'r', encoding="UTF-8") as fp:
        data = json.load(fp)

    for item in data:
        codeFile = f"{dir}/code.original_subtoken"
        summaryFile = f"{dir}/javadoc.original"
        with open(codeFile, 'a', encoding="UTF-8") as fp:
            fp.write(item["source"] + "\n")
        with open(summaryFile, 'a', encoding="UTF-8") as fp:
            fp.write(item["target"] + "\n")


if __name__ == "__main__":
    datasets = ["test", "valid", "train"]
    files = "C:/<YOUR_CODESEARCHNET_FILEPATH_GOES_HERE>/java"
    for dataset in datasets:
        generateData(f"{files}/{dataset}")
