# EpochValidationExperiment

Replication package for our paper "How the Choice of Epoch Validation Metric Affects Transformers for Neural Source Code Summarisation".

## To run

- Download the Python scripts you would like to replicate, as well as the requirements.txt file.
- Download your dataset.  We used the Funcom datase ([Paper](https://aclanthology.org/N19-1394.pdf), [Dataset](http://leclair.tech/data/funcom/index_v5.html#procdata)), preprocessed with JavaDatasetCleaner ([Paper](https://www.lancaster.ac.uk/~elhaj/docs/gem2022.pdf), [GitHub](https://github.com/phillijm/JavaDatasetCleaner/))
  - You may need to edit the generateJSONData method of the script you would like to replicate if you aren't using the same data format we did.
- In a container or virtual environment, install the PyPI packages from requirements.txt.

``` bash
pip3 install -r requirements.txt
```

- Run the script you would like to replicate using the TOKENIZERS_PARALLELISM flag.

``` bash
TOKENIZERS_PARALLELISM=true python3 Exp1-FuncomDataset-XXX.py
```

## To evaluate your model

- Download the evaluateModel.py script, the meteor.py script, and METEOR v1.5 ([Paper](https://aclanthology.org/W14-3348/), [Download](https://www.cs.cmu.edu/~alavie/METEOR/)).
  - You will need to extract METEOR to a directory named "meteor".
- Run the evaluateModel.py script, passing it the directory of the model you would like to evaluate.

``` bash
python3 evaluateModel.py Exp1-FuncomDataset-Results-XXX
```

- The script will run the model and save its outputs, as well as testing them against a suite of metrics.
  - Model outputs will be saved to "out.txt".
  - Evaluations of all model outputs will be saved to "eval.json".
  - Overall model evaluation will be saved to "eval.txt".
