""" BART (Seq-2-Seq) Encoder/Decoder Transformer using PyTorch Lightning.
    We tuned this model on the Funcom Dataset.
    From template code found at: https://lightning-transformers.readthedocs.io
                                 /en/latest/tasks/nlp/summarization.html
                            and: https://gist.github.com/ajazturki10
                                 /247ac21e001025d8b65c7418edd4faf5#file-model-py

Author: Jesse Phillips <j.m.phillips@lancaster.ac.uk>
"""
import os
import torch
import time
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from lightning_transformers.task.nlp.summarization import (
    SummarizationDataModule)
from transformers import BartForConditionalGeneration, \
                         AutoTokenizer, \
                         AutoConfig
import evaluate
import gc
from meteor import Meteor

base = "facebook/bart-base"
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=base)
ignoreID = torch.tensor(-100)
padTokenID = torch.tensor(tokenizer.pad_token_id)


def transformDataset(dataPath: str) -> str:
    """ Given a filepath to a dataset formatted for NeuralCodeSum, formats it
        for PyTorch Lightning Transformers.

    Args:
        dataPath (string): the path to the dataset.
    Returns:
        string: the path of the new dataset.
    """
    import json
    with open(dataPath, "r", encoding="UTF-8") as fp:
        dataset = json.load(fp)
        newDataset = []
    for d in dataset:
        d["source"] = d["text"]
        d["target"] = d["summary"]
        del d["text"]
        del d["summary"]
        newDataset.append(d)
    dataPathList = dataPath.split("/")
    dataPathList[-1] = f"pl_{dataPathList[-1]}"
    dataPath = "/".join(dataPathList)
    with open(f"{dataPath}", "w", encoding="UTF-8") as fp:
        fp.write(json.dumps(newDataset, indent=4))
    return f"{dataPath}"


class AdamWBARTBleuModel(pl.LightningModule):
    def getMetric(self, metricName: str) -> evaluate.EvaluationModule:
        """ Gets a metric from HuggingFace's Evaluate API.
            Tries three times because their network can get flaky when busy.
        Args:
            metricName (str): the name of the metric to use.
        Returns:
            EvaluationModule: the metric.
        """
        try:
            return evaluate.load(metricName)
        except Exception:
            time.sleep(60)
            try:
                return evaluate.load(metricName)
            except Exception:
                time.sleep(60)
                try:
                    return evaluate.load(metricName)
                except Exception as e:
                    print(f"could not access HuggingFace {metricName}")
                    raise e

    def __init__(self, model: str, *args: tuple, **kwargs: dict) -> None:
        """ model constructor.
        Args:
            model (str): the base name for the HuggingFace model.
            args (tuple): model args.
            kwargs (dict): model keyword args.
        """
        super(AdamWBARTBleuModel, self).__init__()
        self.use_stemmer = kwargs["use_stemmer"]
        self.val_target_max_length = kwargs["val_target_max_length"]
        self.num_beams = kwargs["num_beams"]
        self.compute_generate_metrics = kwargs["compute_generate_metrics"]

        config = AutoConfig.from_pretrained(model)
        self.model = BartForConditionalGeneration(config=config)
        self.model.init_weights()
        self.save_hyperparameters(logger=False)

    def configure_optimizers(self) -> torch.optim.AdamW:
        """ configures the optimizer(s) for our model (AdamW)
        Returns:
            AdamW: the optimizer used.
        """
        return torch.optim.AdamW(self.parameters(), lr=2e-5)

    def forward(self,
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                labels=None,
                decoder_attention_mask=None) -> tuple:
        """ forward step for our model
        Args:
            input_ids (torch.Tensor): the inputs
            attention_mask (torch.Tensor): the attention mask
            labels (None/torch.Tensor): the target labels
            decoder_attention_mask (None/torch.tensor): dec. att. mask if used
        Returns:
            Tuple: the loss and logits.
        """
        outputs = self.model(input_ids=input_ids,
                             attention_mask=attention_mask,
                             labels=labels,
                             decoder_attention_mask=decoder_attention_mask)
        return outputs.loss, outputs.logits

    def on_train_epoch_start(self):
        if os.path.exists(checkpointCallback.best_model_path):
            if checkpointCallback.best_model_path != \
               checkpointCallback.last_model_path:
                oldWeights = torch.load(checkpointCallback.best_model_path
                                        )['state_dict']
                model.load_state_dict(oldWeights)

                # Add noise to weights to tweak training.
                with torch.no_grad():
                    for par in model.parameters():
                        par.add_((torch.randn(par.size()) * 0.001).to("cuda"))

                model.train()

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        """ The training loop for our model.
        NB: call self or forward, not both.
        Args:
            batch (dict): the data batch
            batch_idx (int): the batch index
        Returns:
            torch.Tensor: the loss.
        """
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]

        loss, output = self(input_ids, attention_mask, labels)
        return loss

    def validation_step(self, batch: dict, batch_idx: int) -> float:
        """ The validation loop for our model.
        Args:
            batch (dict): the data batch
            batch_idx (int): the batch index
        Returns:
            float: the BLEU-1 score.
        """
        labels = batch["labels"]
        labels = labels.where(labels != ignoreID.to(self.device),
                              padTokenID.to(self.device))
        decodedLabels = tokenizer.batch_decode(labels,
                                               skip_special_tokens=True)
        decodedLabels = [[label.strip()] for label in decodedLabels]
        batch = {k: torch.Tensor(v) for k, v in batch.items()}
        outputs = self.model(**batch)
        logits = outputs.logits
        decodedPreds = tokenizer.batch_decode(torch.argmax(logits, dim=-1),
                                              skip_special_tokens=True)
        decodedPreds = [pred.strip() for pred in decodedPreds]
        metric = self.getMetric("bleu")
        bleu = metric.compute(predictions=decodedPreds,
                              references=decodedLabels,
                              max_order=1,
                              smooth=False)["bleu"]
        self.log("val_bleu", bleu, on_step=False, on_epoch=True, prog_bar=True)
        return bleu

    def test_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        """ Post-training evaluation.
        Args:
            batch (dict): the data batch
            batch_idx (int): the batch index
        Returns:
            dict: the outputs.
        """
        torch.cuda.empty_cache()
        gc.collect()
        with torch.no_grad():
            output = self.model(**batch)
        outs = {}
        logits = output.logits
        predictions = tokenizer.batch_decode(torch.argmax(logits, dim=-1),
                                             skip_special_tokens=True)
        predictions = [pred.strip() for pred in predictions]

        labels = batch["labels"]
        labels = labels.where(labels != ignoreID.to(self.device),
                              padTokenID.to(self.device))
        references = tokenizer.batch_decode(labels,
                                            skip_special_tokens=True)
        references = [label.strip() for label in references]
        refs = [[label] for label in references]

        metric = self.getMetric("bleu")
        for n in range(4):
            outs["BLEU-" + str(n+1)] = metric.compute(predictions=predictions,
                                                      references=refs,
                                                      max_order=n+1,
                                                      smooth=False)["bleu"]
        outs["Smoothed BLEU-4"] = metric.compute(predictions=predictions,
                                                 references=refs,
                                                 max_order=4,
                                                 smooth=True)["bleu"]
        metric = self.getMetric("rouge")
        outs["ROUGE-L"] = metric.compute(predictions=predictions,
                                         references=refs,
                                         rouge_types=["rougeL"],
                                         use_stemmer=True)["rougeL"]
        outs["ROUGE-1"] = metric.compute(predictions=predictions,
                                         references=refs,
                                         rouge_types=["rouge1"],
                                         use_stemmer=True)["rouge1"]
        metric = self.getMetric("bertscore")
        f1 = metric.compute(predictions=predictions,
                            references=references,
                            model_type="microsoft/deberta-xlarge-mnli",
                            lang="en",
                            device="cuda",
                            batch_size=48)["f1"]
        outs["BERTScore"] = sum(f1) / len(f1)
        metric = self.getMetric("frugalscore")
        fs = metric.compute(predictions=predictions,
                            references=references,
                            batch_size=48,
                            device='gpu')["scores"]
        outs["FrugalScore"] = sum(fs) / len(fs)
        meteor = Meteor()  # Using my lib for the official implementation.
        meteor.storeData(predictions, references, ".")
        meteor.callMeteor(".")
        outs["METEOR"] = meteor.getFinalScore(".", "tmpstdout.txt")
        with open("eval.txt", "a", encoding="UTF-8") as fp:
            for k, v in outs.items():
                fp.write(f"{k}: {v}\n")
        return outs

    def on_test_epoch_end(self):
        bleu1, \
              bleu2, \
              bleu3, \
              bleu4, \
              sBleu4, \
              rougel, \
              rouge1, \
              bertscore, \
              frugalscore, \
              meteor = ([] for i in range(10))
        with open("eval.txt", "r", encoding="UTF-8") as fp:
            lines = [line.rstrip() for line in fp]
            for line in lines:
                if line.startswith("BLEU-1"):
                    bleu1.append(float(line.split(": ")[1]))
                if line.startswith("BLEU-2"):
                    bleu2.append(float(line.split(": ")[1]))
                if line.startswith("BLEU-3"):
                    bleu3.append(float(line.split(": ")[1]))
                if line.startswith("BLEU-4"):
                    bleu4.append(float(line.split(": ")[1]))
                if line.startswith("Smoothed BLEU-4"):
                    sBleu4.append(float(line.split(": ")[1]))
                if line.startswith("ROUGE-L"):
                    rougel.append(float(line.split(": ")[1]))
                if line.startswith("ROUGE-1"):
                    rouge1.append(float(line.split(": ")[1]))
                if line.startswith("BERTScore"):
                    bertscore.append(float(line.split(": ")[1]))
                if line.startswith("FrugalScore"):
                    frugalscore.append(float(line.split(": ")[1]))
                if line.startswith("METEOR"):
                    meteor.append(float(line.split(": ")[1]))
        with open("finalEval.txt", "w", encoding="UTF-8") as fp:
            fp.write(f"BLEU-1: {str(sum(bleu1) / len(bleu1))}\n")
            fp.write(f"BLEU-2: {str(sum(bleu2) / len(bleu2))}\n")
            fp.write(f"BLEU-3: {str(sum(bleu3) / len(bleu3))}\n")
            fp.write(f"BLEU-4: {str(sum(bleu4) / len(bleu4))}\n")
            fp.write(f"Smoothed BLEU-4: {str(sum(sBleu4) / len(sBleu4))}\n")
            fp.write(f"ROUGE-L: {str(sum(rougel) / len(rougel))}\n")
            fp.write(f"ROUGE-1: {str(sum(rouge1) / len(rouge1))}\n")
            fp.write(f"BERTScore: {str(sum(bertscore) / len(bertscore))}\n")
            fp.write(f"FrugalScore: \
{str(sum(frugalscore) / len(frugalscore))}\n")
            fp.write(f"METEOR: {str(sum(meteor) / len(meteor))}\n")


class FuncomDataModule(SummarizationDataModule):
    def __init__(self, *args: tuple, **kwargs: dict) -> None:
        super(FuncomDataModule, self).__init__(*args, **kwargs)


if __name__ == '__main__':
    torch.multiprocessing.freeze_support()

    torch.cuda.empty_cache()  # Clear the PyTorch cache - saves GPU memory.
    torch.backends.cudnn.benchmark = True  # Run fastest convolutions.
    gc.collect()

    # Transform old datasets into new ones
    trainingDataPath = "../dataset/training/dataset.json"
    validationDataPath = "../dataset/validation/dataset.json"
    evaluationDataPath = "../dataset/evaluation/dataset.json"
    trainingDataPath = transformDataset(trainingDataPath)
    validationDataPath = transformDataset(validationDataPath)
    evaluationDataPath = transformDataset(evaluationDataPath)

    checkpointCallback = ModelCheckpoint(monitor="val_bleu",
                                         save_top_k=3,
                                         mode="max",
                                         save_weights_only=True)

    dm = FuncomDataModule(batch_size=48,
                          num_workers=4,
                          max_source_length=128,
                          max_target_length=128,
                          train_file=os.path.abspath(trainingDataPath),
                          validation_file=os.path.abspath(validationDataPath),
                          test_file=os.path.abspath(evaluationDataPath),
                          tokenizer=tokenizer,
                          max_length=512,
                          padding="max_length")

    model = AdamWBARTBleuModel(model=base,
                               use_stemmer=True,
                               val_target_max_length=128,
                               num_beams=None,
                               compute_generate_metrics=True)

    trainer = pl.Trainer(accelerator="gpu",
                         devices=[0],  # GPU ID/s GO HERE
                         check_val_every_n_epoch=1,
                         max_epochs=200,
                         min_epochs=20,
                         callbacks=[checkpointCallback,
                                    EarlyStopping(monitor="val_bleu",
                                                  mode="max",
                                                  patience=5)])
    trainer.fit(model, dm)

    # Evaluate the model.
    trainer.test(ckpt_path="best", datamodule=dm)
