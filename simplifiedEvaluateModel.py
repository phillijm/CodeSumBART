""" simplifiedEvaluateModel.py - Evaluates a model for neural source code
                                 summarisation.  Averages only.

Author: Jesse Phillips <j.m.phillips@lancaster.ac.uk>
"""
import os
import sys
import json
import evaluate
import torch
from multiprocessing import Process
from meteor import Meteor
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, \
                         AutoTokenizer, \
                         pipeline

torch.cuda.empty_cache()  # Clear the PyTorch cache - saves GPU memory.

# Set path seperator for windows or *nix.
if os.name == "nt":
    psep = "\\"
else:
    psep = "/"


def generateJSONData(dir):
    """ Generates and saves JSON data for use from the format used by Funcom.

    Args:
        dir (string): the name of the directory holding the dataset you want.
    """
    if Path(f".{psep}dataset{psep}{dir}{psep}dataset.json").is_file():
        return

    import json
    print(f"Generating Dataset: {dir}")
    data = []
    with open(f".{psep}dataset{psep}{dir}{psep}code.original_subtoken",
              encoding='UTF-8') as fp:
        code = fp.readlines()

    with open(f".{psep}dataset{psep}{dir}{psep}javadoc.original",
              encoding='UTF-8') as fp1:
        comments = fp1.readlines()
    for cnt in range(len(code) - 1):
        data.append({})
        data[cnt]["text"] = code[cnt]
        data[cnt]["summary"] = comments[cnt]
    jsonData = json.dumps(data, indent=4)

    with open(f".{psep}dataset{psep}{dir}{psep}dataset.json", "w") as fp:
        fp.write(jsonData)


def getFrugalScores(predictions: list, references: list) -> list:
    """ Gets a list of FrugalScores for a list of prediction/reference pairs.

    Args:
        predictions (list): the list of predictions.
        references (list): the list of references.

    Returns:
        float: the FrugalScore.
    """
    metric = evaluate.load("frugalscore")
    fs = metric.compute(predictions=predictions,
                        references=references,
                        batch_size=48,
                        device='gpu')["scores"]
    return sum(fs) / len(fs)


def getBertScores(predictions: list, references: list) -> list:
    """ Gets a list of BERTScores for a list of prediction/reference pairs.

    Args:
        predictions (list): the list of predictions.
        references (list): the list of references.

    Returns:
        float: the BERTScore F1.
    """
    metric = evaluate.load("bertscore")
    f1 = metric.compute(predictions=predictions,
                        references=references,
                        model_type="microsoft/deberta-xlarge-mnli",
                        lang="en",
                        device="cuda",
                        batch_size=48)["f1"]
    return sum(f1) / len(f1)


def getAverageBleu(references: list,
                   predictions: list,
                   n: int = 1,
                   smoothed: bool = False) -> float:
    """ Returns a BLEU-n value for a corpus of reference-prediction pairs.

    Args:
        references (list): a list of all references.
        predictions (list): a list of all predictions.
        n (int): the BLEU n-gram value (IE: BLEU-1, BLEU-2) (defaults to 1).
        smoothed (bool): apply smoothing (defaults to false).

    Returns:
        float: the BLEU-n value.
    """
    metric = evaluate.load("bleu")
    refs = [[label] for label in references]
    return metric.compute(predictions=predictions,
                          references=refs,
                          max_order=n,
                          smooth=smoothed)["bleu"]


def getAverageRouge(labels: list,
                    predictions: list,
                    rougeType: str) -> float:
    """ Returns a ROUGE value for a corpus of reference-prediction pairs.

    Args:
        labels (list): the labels associated with the predictions.
        predictions (list): the predictions returned by the model.
        rougeType (string): the type (IE: rouge1, rougeL).

    Returns:
        float: the final computed metric.
    """
    metric = evaluate.load("rouge")
    refs = [[label] for label in labels]
    return metric.compute(predictions=predictions,
                          references=refs,
                          rouge_types=[rougeType],
                          use_stemmer=True)[rougeType]


def getAverageMeteor(meteor: Meteor,
                     predictions: list,
                     references: list,
                     path: str = ".") -> float:
    """ METEOR 1.5.  Uses the official Java package.
    Gets corpus METEOR score for prediction/reference pairs.

    Args:
        meteor (Meteor): an instance of the Meteor class.
        predictions (list): the predictions returned by the model.
        references (list): the references associated with the predictions.
        path (str): the output path for METEOR text files (defaults to ".").

    Returns:
        float: the METEOR score.
    """
    meteor.storeData(predictions, references, path)
    meteor.callMeteor(path)
    return meteor.getFinalScore(path, "tmpstdout.txt")


def bleuNProcess(references: list,
                 predictions: list,
                 n: int = 1,
                 smoothed: bool = False):
    """ Generates and saves a BLEU score.  Called by a multiprocessing.Process.

    Args:
        references (list): the list of references.
        predictions (list): the list of predictions.
        n (int): n-gram length of BLEU score.  Defaults to 1
        smoothed (bool): apply a smoothing algorithm?  Defaults to False.
    """
    if smoothed:
        with open(f".{psep}asbleu{n}.txt", 'w') as fp:
            fp.write(str(getAverageBleu(references, predictions, n, smoothed)))
        print(f"    Calculated Smoothed BLEU-{n}")
    else:
        with open(f".{psep}ableu{n}.txt", 'w') as fp:
            fp.write(str(getAverageBleu(references, predictions, n, smoothed)))
        print(f"    Calculated BLEU-{n}")


def rougeNProcess(references: list,
                  predictions: list,
                  rougeType: str):
    """ Generates and saves ROUGE score.  Called by a multiprocessing.Process.

    Args:
        references (list): the list of references.
        predictions (list): the list of predictions.
        rougeType (str): Which ROUGE score to generate.  Defaults to ROUGE-L.
    """
    with open(f".{psep}a{rougeType}.txt", 'w') as fp:
        fp.write(str(getAverageRouge(references, predictions, rougeType)))
    print(f"    Calculated {rougeType}")


def removeTmpFiles():
    """ Removes temporary files used to store the evaluation.
    """
    os.remove(f".{psep}ableu1.txt")
    os.remove(f".{psep}ableu2.txt")
    os.remove(f".{psep}ableu3.txt")
    os.remove(f".{psep}ableu4.txt")
    os.remove(f".{psep}asbleu4.txt")
    os.remove(f".{psep}arouge1.txt")
    os.remove(f".{psep}arougeL.txt")
    os.remove(f".{psep}pres.txt")
    os.remove(f".{psep}refs.txt")
    os.remove(f".{psep}tmpstderr.txt")
    os.remove(f".{psep}tmpstdout.txt")


def main(argv: list):
    modelBaseName = "t5-small"
    modelPath = f".{psep}{argv[0]}"
    tokenizer = AutoTokenizer.from_pretrained(modelBaseName)
    model = AutoModelForSeq2SeqLM.from_pretrained(modelPath)
    # ------------------------- Find/Generate dataset -------------------------
    generateJSONData("evaluation")
    with open(f".{psep}dataset{psep}evaluation{psep}dataset.json") as fp:
        data = json.load(fp)
    # ---------------------- Find/Generate model outputs ----------------------
    if not Path(f".{psep}out.txt").is_file():
        print("No model outputs found.  Generating...")
        sentences = [sentence["text"] for sentence in data]
        batchSize = 48
        summarizationPipe = pipeline("summarization",
                                     model=model,
                                     tokenizer=tokenizer,
                                     device=0,  # Use 0 for GPU
                                     batch_size=batchSize,
                                     num_workers=16)
        try:
            with open(f".{psep}out.txt", 'w') as fp:
                for out in tqdm(summarizationPipe(sentences,
                                                  batch_size=batchSize)):
                    fp.write(str(out) + "\n")
            # Stop hogging GPU memory - shouldn't be needed but is.
            import gc
            torch.cuda.empty_cache()
            gc.collect()
        except MemoryError as e:
            sys.exit(e)
    else:
        print("Found model outputs!  Scoring...")

    with open(f".{psep}out.txt", 'r') as fp:
        predictions = [line.rstrip('\'}\n') for line in fp]
    predictions = [ln.rstrip('\"}\n') for ln in predictions]
    predictions = [ln.replace("{'summary_text': '", '') for ln in predictions]
    predictions = [ln.replace("{'summary_text': \"", '') for ln in predictions]
    references = [reference["summary"] for reference in data]
    for x in predictions:
        x = x.rstrip("\n")
    for x in references:
        x = x.rstrip("\n")
    processes = []
    # ------------------------ Generate metric scores -------------------------
    print("Calculating FrugalScore...")
    aFrugalScores = getFrugalScores(predictions, references)
    print("Calculating BERTScore...")
    aBertScores = getBertScores(predictions, references)
    print("Calculating BLEU & ROUGE scores...")
    processes.append(Process(target=bleuNProcess,
                             args=(references, predictions)))
    processes.append(Process(target=bleuNProcess,
                             args=(references, predictions, 2)))
    processes.append(Process(target=bleuNProcess,
                             args=(references, predictions, 3)))
    processes.append(Process(target=bleuNProcess,
                             args=(references, predictions, 4)))
    processes.append(Process(target=bleuNProcess,
                             args=(references, predictions, 4, True)))
    processes.append(Process(target=rougeNProcess,
                             args=(references, predictions, "rouge1")))
    processes.append(Process(target=rougeNProcess,
                             args=(references, predictions, "rougeL")))
    for p in processes:
        p.start()
    for p in processes:
        p.join()
    with open(f".{psep}ableu1.txt", 'r') as fp:
        aBleu1s = float(fp.readline())
    with open(f".{psep}ableu2.txt", 'r') as fp:
        aBleu2s = float(fp.readline())
    with open(f".{psep}ableu3.txt", 'r') as fp:
        aBleu3s = float(fp.readline())
    with open(f".{psep}ableu4.txt", 'r') as fp:
        aBleu4s = float(fp.readline())
    with open(f".{psep}asbleu4.txt", 'r') as fp:
        aSBleu4s = float(fp.readline())
    with open(f".{psep}arouge1.txt", 'r') as fp:
        aRouge1s = float(fp.readline())
    with open(f".{psep}arougeL.txt", 'r') as fp:
        aRougeLs = float(fp.readline())
    print("Calculating METEOR scores...")
    meteor = Meteor()
    aMeteors = getAverageMeteor(meteor, predictions, references)
    # -------------------------- Save metric scores ---------------------------
    print("Calculating and saving averages...")
    with open(f".{psep}eval.txt", 'w') as fp:
        fp.write(f""" Results:
        FrugalScore: {aFrugalScores}
        BERTScore: {aBertScores}
        BLEU-1: {aBleu1s}
        BLEU-2: {aBleu2s}
        BLEU-3: {aBleu3s}
        BLEU-4: {aBleu4s}
        Smoothed BLEU-4: {aSBleu4s}
        ROUGE-1: {aRouge1s}
        ROUGE-L: {aRougeLs}
        METEOR: {aMeteors}""")
    # -------- Print metrics to the screen and delete temporary files ---------
    with open(f".{psep}eval.txt", 'r') as fp:
        print(fp.read())
    removeTmpFiles()


if __name__ == "__main__":
    main(sys.argv[1:])
