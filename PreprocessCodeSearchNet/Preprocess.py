""" Preprocessing CodeSearchNet data for use with our model.

Author: Jesse Phillips <j.m.phillips@lancaster.ac.uk>
"""
import os
import json
import subprocess


def generateJSONData(dir: str, file: str) -> None:
    """ Pulls JSON data from the format used by CodeSearchNet.
    Args:
        dir (string): the name of the directory holding the dataset.
        file (string): the name of the dataset to use.
    """
    print(f"Generating Dataset: {dir}")
    data = []
    code = []
    comments = []

    with open(f"{dir}/{file}.jsonl", 'r', encoding="UTF-8") as fp:
        jsonInput = [json.loads(line) for line in fp]
    for item in jsonInput:
        code.append(item["code"])
        summary = ""
        for word in item["docstring_tokens"]:
            summary += f"{word} "
        comments.append(summary[:-1])

    for cnt in range(len(code)):
        data.append({})
        data[cnt]["source"] = code[cnt]
        data[cnt]["target"] = comments[cnt]
    jsonData = json.dumps(data, indent=4)
    os.mkdir(f"{dir}/{file}")
    with open(f"{dir}/{file}/pl1_dataset.json", 'w', encoding="UTF-8") as fp:
        fp.write(jsonData)


def callJava():
    """ Calls our Java Source Code Cleaner, which uses JavaDatasetCleaner's
    JavaDatasetPreprocessor class (Phillips et al., 2022).
    """
    print("Compile JavaCodeSearchNetCleaner, then press Enter.\n[ENTER]: ")
    input()
    command = f"java -jar ./JavaCodeSearchNetCleaner.jar"
    cmd = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
    cmd.communicate()


def generateFinalJSONData(dir: str) -> None:
    """ Saves data from the pseudo-json stored by the Java program.
    Args:
        dir (string): the name of the directory holding the dataset you want.
    """

    print(f"Generating Final Dataset: {dir}")
    code = []
    comments = []

    with open(f"{dir}/dataset.json", 'r', encoding="UTF-8") as fp:
        for line in fp:
            if line.startswith("    \"text\": \""):
                code.append(line[13:][:-3])
            elif line.startswith("    \"summary\": \""):
                comments.append(line[16:][:-2])

    data = []
    for cnt in range(len(code)):
        data.append({})
        data[cnt]["source"] = code[cnt]
        data[cnt]["target"] = comments[cnt]
    jsonData = json.dumps(data, indent=4)

    with open(f"{dir}/pl_dataset.json", 'w', encoding="UTF-8") as fp:
        fp.write(jsonData)


def cleanDir(file: str) -> None:
    """ Removes temporary dataset files created while processing the data.
    Args:
        file (string): the directory holding the dataset.
    """
    os.remove(f"{file}/pl1_dataset.json")
    os.remove(f"{file}/dataset.json")


if __name__ == "__main__":
    datasets = ["train", "valid", "test"]
    files = "C:/<YOUR_CODESEARCHNET_FILEPATH_GOES_HERE>/java"
    for dataset in datasets:
        generateJSONData(files, dataset)
    callJava()
    for dataset in datasets:
        generateFinalJSONData(f"{files}/{dataset}")
