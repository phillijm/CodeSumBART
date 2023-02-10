""" evaluateModel.py - Evaluates a model for neural source code summarisation.

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
        list: the FrugalScores.
    """
    metric = evaluate.load("frugalscore")
    refs = [label for label in references]
    return metric.compute(predictions=predictions,
                          references=refs,
                          batch_size=48,
                          device='gpu')["scores"]


def getBertScores(predictions: list, references: list) -> list:
    """ Gets a list of BERTScores for a list of prediction/reference pairs.

    Args:
        predictions (list): the list of predictions.
        references (list): the list of references.

    Returns:
        list: the BERTScore F1s.
    """
    metric = evaluate.load("bertscore")
    refs = [label for label in references]
    return metric.compute(predictions=predictions,
                          references=refs,
                          model_type="microsoft/deberta-xlarge-mnli",
                          lang="en",
                          device="cuda",
                          batch_size=48)["f1"]


def getAverage(metrics: list) -> float:
    """ Averages a list of floats (mean).

    Args:
        metrics (list): the list to average.

    Returns:
        float: the average.
    """
    return sum(metrics) / len(metrics)


def getBleus(references: list,
             predictions: list,
             n: int = 1,
             smoothed: bool = False) -> list:
    """ Returns a list of all BLEU-n values for reference-prediction pairs.

    Args:
        references (list): a list of all references.
        predictions (list): a list of all predictions.
        n (int): the BLEU n-gram value (IE: BLEU-1, BLEU-2) (defaults to 1).
        smoothed (bool): apply smoothing (defaults to false).

    Returns:
        list: the BLEU-n values.
    """
    bleus = []
    for cnt in range(len(references)):
        bleus.append(getBleu(references[cnt], predictions[cnt], n, smoothed))
    return bleus


def getBleu(reference: str,
            prediction: str,
            n: int = 1,
            smoothed: bool = False) -> float:
    """ Returns a BLEU-n value for a reference-prediction pair.

    Args:
        references (list): a reference.
        predictions (list): a prediction.
        n (int): the BLEU n-gram value (IE: BLEU-1, BLEU-2) (defaults to 1).
        smoothed (bool): apply smoothing (defaults to false).

    Returns:
        float: the BLEU-n value.
    """
    return getAverageBleu([[reference]], [prediction], n, smoothed)


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


def getRouges(references: list,
              predictions: list,
              rougeType: str) -> list:
    """ Returns a list of all ROUGE values for reference-prediction pairs.

    Args:
        references (list): a list of all references.
        predictions (list): a list of all predictions.
        rougeType (string): the type (IE: rouge1, rougeL) (defaults to rougeL).

    Returns:
        list: the ROUGE-n values.
    """
    rouges = []
    for cnt in range(len(references)):
        rouges.append(getRouge(references[cnt], predictions[cnt], rougeType))
    return rouges


def getRouge(reference: str,
             prediction: str,
             rougeType: str) -> float:
    """ Returns a ROUGE value for a reference-prediction pair.

    Args:
        references (list): a reference.
        predictions (list): a prediction.
        rougeType (string): the type (IE: rouge1, rougeL) (defaults to rougeL).

    Returns:
        float: the ROUGE value.
    """
    return getAverageRouge([reference], [prediction], rougeType)


def getAverageRouge(labels: list,
                    predictions: list,
                    rougeType: str) -> float:
    """ Returns a ROUGE value for a corpus of reference-prediction pairs.

    Args:
        labels (list): the labels associated with the predictions.
        predictions (list): the predictions returned by the model.
        rougeType (string): the type (IE: rouge1, rougeL) (defaults to rougeL).

    Returns:
        float: the final computed metric.
    """
    metric = evaluate.load("rouge")
    refs = [[label] for label in labels]
    return metric.compute(predictions=predictions,
                          references=refs,
                          rouge_types=[rougeType],
                          use_stemmer=True)[rougeType]


def getMeteor(predictions: list,
              references: list,
              meteor: Meteor,
              path: str = ".") -> list:
    """ METEOR 1.5.  Uses the official Java package.
    Gets a list of METEOR scores for a list of prediction/reference pairs.

    Args:
        predictions (list): the list of predictions.
        references (list): the list of references.
        meteor (Meteor): an instance of the Meteor class.
        path (str): the output path for METEOR text files (defaults to ".").

    Returns:
        list: the METEOR scores.
    """
    meteor.storeData(predictions, references, path)
    meteor.callMeteor(path)
    meteors = meteor.getResults(path, "tmpstdout.txt", len(predictions))
    return meteors


def getAverageMeteor(meteor: Meteor, path: str = ".") -> float:
    """ METEOR 1.5.  Uses the official Java package.
    Gets corpus METEOR score for prediction/reference pairs.

    Args:
        meteor (Meteor): an instance of the Meteor class.
        path (str): the output path for METEOR text files (defaults to ".").

    Returns:
        float: the METEOR score.
    """
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
        with open(f".{psep}sbleu{n}.txt", 'w') as fp:
            for val in getBleus(references, predictions, n, smoothed):
                fp.write(str(val) + '\n')
        with open(f".{psep}asbleu{n}.txt", 'w') as fp:
            fp.write(str(getAverageBleu(references, predictions, n, smoothed)))
        print(f"    Calculated Smoothed BLEU-{n}")
    else:
        with open(f".{psep}bleu{n}.txt", 'w') as fp:
            for val in getBleus(references, predictions, n, smoothed):
                fp.write(str(val) + '\n')
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
    with open(f".{psep}{rougeType}.txt", 'w') as fp:
        for val in getRouges(references, predictions, rougeType):
            fp.write(str(val) + '\n')
    with open(f".{psep}a{rougeType}.txt", 'w') as fp:
        fp.write(str(getAverageRouge(references, predictions, rougeType)))
    print(f"    Calculated {rougeType}")


def removeTmpFiles():
    """ Removes temporary files used to store the evaluation.
    """
    os.remove(f".{psep}bleu1.txt")
    os.remove(f".{psep}ableu1.txt")
    os.remove(f".{psep}bleu2.txt")
    os.remove(f".{psep}ableu2.txt")
    os.remove(f".{psep}bleu3.txt")
    os.remove(f".{psep}ableu3.txt")
    os.remove(f".{psep}bleu4.txt")
    os.remove(f".{psep}ableu4.txt")
    os.remove(f".{psep}sbleu4.txt")
    os.remove(f".{psep}asbleu4.txt")
    os.remove(f".{psep}rouge1.txt")
    os.remove(f".{psep}arouge1.txt")
    os.remove(f".{psep}rougeL.txt")
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
    predictions = [ln.replace("{'summary_text': '", '') for ln in predictions]
    references = [reference["text"] for reference in data]
    references = [ln.replace("{'summary_text': '", '') for ln in references]
    processes = []
    # ------------------------ Generate metric scores -------------------------
    print("Calculating FrugalScore...")
    frugalScores = getFrugalScores(predictions, references)
    print("Calculating BERTScore...")
    bertScores = getBertScores(predictions, references)
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
    with open(f".{psep}bleu1.txt", 'r') as fp:
        bleu1s = [line.rstrip() for line in fp]
    with open(f".{psep}bleu2.txt", 'r') as fp:
        bleu2s = [line.rstrip() for line in fp]
    with open(f".{psep}bleu3.txt", 'r') as fp:
        bleu3s = [line.rstrip() for line in fp]
    with open(f".{psep}bleu4.txt", 'r') as fp:
        bleu4s = [line.rstrip() for line in fp]
    with open(f".{psep}sbleu4.txt", 'r') as fp:
        sBleu4s = [line.rstrip() for line in fp]
    with open(f".{psep}rouge1.txt", 'r') as fp:
        rouge1s = [line.rstrip() for line in fp]
    with open(f".{psep}rougeL.txt", 'r') as fp:
        rougeLs = [line.rstrip() for line in fp]
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
    meteors = getMeteor(references, predictions, meteor)
    aMeteors = getAverageMeteor(meteor)
    # -------------------------- Save metric scores ---------------------------
    print("Saving scores...")
    with open(f".{psep}eval.json", 'w') as fp:
        fp.write("[")
        for cnt in range(len(references) - 1):
            fp.write(f"""
        {{
            \"id\": {cnt},
            \"prediction\": \"{predictions[cnt].rstrip()}\",
            \"reference\": \"{references[cnt].rstrip()}\",
            \"bleu1\": {bleu1s[cnt]},
            \"bleu2\": {bleu2s[cnt]},
            \"bleu3\": {bleu3s[cnt]},
            \"bleu4\": {bleu4s[cnt]},
            \"smoothbleu4\": {sBleu4s[cnt]},
            \"frugalscore\": {frugalScores[cnt]},
            \"bertscore\": {bertScores[cnt]},
            \"rouge1\": {rouge1s[cnt]},
            \"rougel\": {rougeLs[cnt]},
            \"meteor\": {meteors[cnt]}
        }},
    """)
        fp.write("] ")
    print("Calculating and saving averages...")
    with open(f".{psep}eval.txt", 'w') as fp:
        fp.write(f""" Results:
        FrugalScore: {getAverage(frugalScores)}
        BERTScore: {getAverage(bertScores)}
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
