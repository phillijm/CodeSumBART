""" T5 (Seq-2-Seq) Encoder/Decoder Transformer
    We tuned this model on the Funcom Dataset, using BLEU-1 evaluation.

Author: Jesse Phillips <j.m.phillips@lancaster.ac.uk>
"""

import torch
from torch.utils.data import DataLoader
from accelerate import Accelerator
import evaluate
from transformers import AutoModelForSeq2SeqLM, \
                         AutoTokenizer, \
                         AutoConfig, \
                         DataCollatorForSeq2Seq, \
                         AdamW, \
                         get_scheduler
from datasets import load_dataset
from tqdm.auto import tqdm

torch.cuda.empty_cache()  # Clear the PyTorch cache - saves GPU memory.
torch.backends.cudnn.benchmark = True  # Benchmark & run fastest convolutions.

baseName = "t5-small"
metricName = "bleu"
outputDir = "./Exp1-FuncomDataset-Results-B1"
tokenizer = AutoTokenizer.from_pretrained(baseName)
ignoreID = torch.tensor(-100, device=torch.device("cuda:0"))
padTokenID = torch.tensor(tokenizer.pad_token_id,
                          device=torch.device("cuda:0"))


def generateJSONData(dir):
    """ Generates and saves JSON data for use from the format used by Funcom.

    Args:
        dir (string): the name of the directory holding the dataset you want.
    """
    from pathlib import Path
    if Path(f"./dataset/{dir}/dataset.json").is_file():
        return

    import json
    print(f"Generating Dataset: {dir}")
    data = []
    with open(f"./dataset/{dir}/code.original_subtoken",
              encoding='UTF-8') as fp:
        code = fp.readlines()
    with open(f"./dataset/{dir}/javadoc.original", encoding='UTF-8') as fp1:
        comments = fp1.readlines()
    for cnt in range(len(code) - 1):
        data.append({})
        data[cnt]["text"] = code[cnt]
        data[cnt]["summary"] = comments[cnt]
    jsonData = json.dumps(data, indent=4)

    with open(f"./dataset/{dir}/dataset.json", "w") as fp:
        fp.write(jsonData)


def preprocessFunction(examples):
    """ Preprocess the dataset for use with T5

    Args:
        examples (Dataset): the dataset to process.

    Returns:
        inputs for the model, based on the data provided.
    """
    inputs = ["summarize:" + doc for doc in examples["text"]]
    # if memory spikes kill it, add padding="max_length".
    model_inputs = tokenizer(inputs, max_length=512, truncation=True)

    labels = tokenizer(text_target=examples["summary"],
                       max_length=128,
                       truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def computeMetrics(predictions, labels):
    """ Compute the metric used in epoch validation.

    Args:
        predictions (list): the predictions returned by the model.
        labels (list): the labels associated with the predictions.

    Returns:
        Dict: the final computed metric.
    """
    metric = evaluate.load(metricName)
    refs = [[label] for label in labels]
    bleu = metric.compute(predictions=predictions,
                          references=refs,
                          max_order=1,
                          smooth=False)["bleu"]
    return {"BLEU-1": bleu}


generateJSONData("training")
generateJSONData("validation")

print("Loading datasets: training")
trainingData = load_dataset("json",
                            data_files=f"./dataset/training/dataset.json",
                            split="train")
print("Loading datasets: validation")
validationData = load_dataset("json",
                              data_files=f"./dataset/validation/dataset.json",
                              split="train")

print("Processing datasets")
trainingDataset = trainingData.map(preprocessFunction, batched=True)
validationDataset = validationData.map(preprocessFunction, batched=True)

print("Loading model")

# ------------------------------- Train Model ---------------------------------

# Load untrained model
config = AutoConfig.from_pretrained(baseName)
model = AutoModelForSeq2SeqLM.from_config(config)
dataCollator = DataCollatorForSeq2Seq(tokenizer=tokenizer,
                                      model=model,
                                      return_tensors="pt")

# ------------------------ Load Data Into DataLoaders -------------------------
trainingDataset.set_format("torch",
                           columns=['input_ids', 'attention_mask', 'labels'])
validationDataset.set_format("torch",
                             columns=['input_ids', 'attention_mask', 'labels'])
trainDataLoader = DataLoader(
    trainingDataset,
    collate_fn=dataCollator,
    batch_size=48,
    pin_memory=True,
    num_workers=4,
    prefetch_factor=4,
    shuffle=True)

evalDataLoader = DataLoader(
    validationDataset,
    collate_fn=dataCollator,
    batch_size=48,
    pin_memory=True,
    num_workers=4,
    prefetch_factor=4,  # May be unnecessary
    shuffle=True)

optimizer = AdamW(model.parameters(), lr=2e-5)
accelerator = Accelerator()
model, optimizer, trainDataLoader, evalDataLoader = accelerator.prepare(
    model, optimizer, trainDataLoader, evalDataLoader)

numTrainEpochs = 200
numUpdateStepsPerEpoch = len(trainDataLoader)
numTrainingSteps = numTrainEpochs * numUpdateStepsPerEpoch
lrScheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=numTrainingSteps)

# ------------------------------ Training Loop --------------------------------
previousValidationResult = 0.0
nonImprovingEpochs = 0
minEpochs = 20
bestModel = None
for epoch in range(numTrainEpochs):
    # Training
    model.train()
    for batch in tqdm(trainDataLoader):
        outputs = model(**batch)
        loss = outputs.loss
        accelerator.backward(loss)
        optimizer.step()
        lrScheduler.step()
        optimizer.zero_grad()
    # Model Validation
    model.eval()
    for batch in tqdm(evalDataLoader):
        with torch.no_grad():
            batch = {k: v.to(torch.device("cuda:0")) for k, v in batch.items()}
            outputs = model(**batch)
        logits = outputs.logits
        decodedPreds = tokenizer.batch_decode(torch.argmax(logits, dim=-1),
                                              skip_special_tokens=True)
        decodedPreds = [pred.strip() for pred in decodedPreds]

        labels = batch["labels"]
        labels = labels.where(labels != ignoreID, padTokenID)
        decodedLabels = tokenizer.batch_decode(labels,
                                               skip_special_tokens=True)
        decodedLabels = [[label.strip()] for label in decodedLabels]

    result = computeMetrics(decodedPreds, decodedLabels)
    print(f"Epoch {epoch}: {result}")

# ------------------------------- Save Model ----------------------------------
    if result["BLEU-1"] > previousValidationResult:  # Don't kill best model.
        previousValidationResult = result["BLEU-1"]
        accelerator.wait_for_everyone()
        unwrappedModel = accelerator.unwrap_model(model)
        unwrappedModel.save_pretrained(outputDir,
                                       save_function=accelerator.save)
        if accelerator.is_main_process:
            tokenizer.save_pretrained(outputDir)
        nonImprovingEpochs = 0
    else:
        if epoch >= minEpochs:  # Train for minimum number of epochs anyway.
            nonImprovingEpochs += 1
            if nonImprovingEpochs > 5:  # Don't waste compute if not improving.
                with open("TrainData.txt", 'w') as fp:
                    fp.write(f"actualEpochs: {epoch}")
                break
        model = model.from_pretrained(outputDir).to(torch.device("cuda:0"))
with open("TrainData.txt", 'a') as fp:
    fp.write(f"baseName: {baseName}")
    fp.write(f"metricName: {metricName}")
    fp.write(f"outputDir: {outputDir}")
    fp.write(f"numEpochs: {numTrainEpochs}")
