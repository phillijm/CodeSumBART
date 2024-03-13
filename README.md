# Metric-Oriented Pretraining of Neural Source Code Summarisation Transformers to Enable more Secure Software Development

Replication package for "Metric-Oriented Pretraining of Neural Source Code Summarisation Transformers to Enable more Secure Software Development".

## To run

- Download the Python scripts you would like to replicate, as well as the requirements.txt file.
- Download the meteor.py script, and METEOR v1.5 ([Paper](https://aclanthology.org/W14-3348/), [Download](https://www.cs.cmu.edu/~alavie/METEOR/)).
  - You will need to extract METEOR to a directory named "meteor".
- Download your dataset.  We used the Funcom dataset ([Paper](https://aclanthology.org/N19-1394.pdf), [Dataset](http://leclair.tech/data/funcom/index_v5.html#procdata)), preprocessed with JavaDatasetCleaner ([Paper](https://www.lancaster.ac.uk/~elhaj/docs/gem2022.pdf), [GitHub](https://github.com/phillijm/JavaDatasetCleaner/))
  - You will need to edit the transformDataset method of the script you would like to replicate if you aren't using the same data format we did.
- Install the PyPI packages from requirements.txt, as well as Java 17 or higher.

``` bash
pip install -r requirements.txt
```

- Run the script you would like to replicate using the TOKENIZERS_PARALLELISM flag.  We recommend caching Huggingface Datasets and Transformers.  Here is an example shell script to do that.

``` bash
#!/bin/bash

export HF_DATASETS_CACHE=./data/dataset_cache
export TRANSFORMERS_CACHE=./data/transformer_cache
export TOKENIZERS_PARALLELISM=true

#If training multiple models across multiple GPUs, it may help to train each
#one on its own GPU.  To do this, you can limit the GPUs visible to the script.
#If you do this, the model will train on the first one it can see by default.
#
#export CUDA_VISIBLE_DEVICES=0
#

python YOUR_CHOSEN_MODEL.py
```

## To evaluate your model

- The models used here are self-evaluating: once the model is trained, finalEval.txt is generated, which contains the results of the evaluation metrics.

## Datasets

We used the Funcom dataset ([Paper](https://aclanthology.org/N19-1394.pdf), [Dataset](http://leclair.tech/data/funcom/index_v5.html#procdata)), preprocessed with JavaDatasetCleaner ([Paper](https://www.lancaster.ac.uk/~elhaj/docs/gem2022.pdf), [GitHub](https://github.com/phillijm/JavaDatasetCleaner/)), which saves the dataset in the format used by NeuralCodeSum ([Paper](https://aclanthology.org/2020.acl-main.449.pdf)).  We used the Python method below to convert this into JSON which can be read by the transformDataset method in our model training code.

``` python
def saveJSONData(dir):
    """ Saves the dataset in JSON format.

    Args:
        dir (string): the name of the directory holding the dataset you want.
    """
    from pathlib import Path
    if Path(f"../dataset/{dir}/dataset.json").is_file():
        return

    import json
    print(f"Generating Dataset: {dir}")
    data = []
    with open(f"../dataset/{dir}/code.original_subtoken",
              encoding='UTF-8') as fp:
        code = fp.readlines()
    with open(f"../dataset/{dir}/javadoc.original", encoding='UTF-8') as fp1:
        comments = fp1.readlines()
    for cnt in range(len(code) - 1):
        # for cnt in range(5):
        data.append({})
        data[cnt]["text"] = code[cnt]
        data[cnt]["summary"] = comments[cnt]
    jsonData = json.dumps(data, indent=4)

    with open(f"../dataset/{dir}/dataset.json", "w") as fp:
        fp.write(jsonData)
```

We also evaluated our results using the CodeSearchNet dataset ([Paper](https://arxiv.org/pdf/1909.09436.pdf), [Dataset](https://github.com/github/CodeSearchNet)).  Our scripts for preprocessing that dataset can be found in the "PreprocessCodeSearchNet" directory.

The provided training code for the models expect the data to be in a directory named "dataset" above the model directory.
