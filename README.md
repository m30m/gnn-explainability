# Benchmark Design for GNN Explanation Methods
The is the source code for the paper _Benchmark Design for GNN Explanation Methods_

## Required libraries
You can install the required libraries by running:
```shell
pip install -r requirements.txt
```

## How to run the experiments:
You can use the `main.py` script for running the experiments. Here is the help manual for the script:
```
Usage: main.py [OPTIONS] EXPERIMENT:[infection|community|saturation]

Arguments:
  EXPERIMENT:[infection|community|saturation]
                                  Dataset to use  [required]

Options:
  --sample-count INTEGER          How many times to retry the whole experiment
                                  [default: 10]

  --num-layers INTEGER            Number of layers in the GNN model  [default: 4]

  --concat-features / --no-concat-features
                                  Concat embeddings of each convolutional
                                  layer for final fc layers  [default: True]

  --conv-type TEXT                Convolution class. Can be GCNConv or
                                  GraphConv  [default: GraphConv]
  --help                          Show this message and exit.
```
Experiment results in the paper were produced by the following commands:
```
python main.py infection
python main.py community
python main_node.py saturation --num-layers 1 # for the negative evidence experiment
```

You can run the `Pitfall2-Example.ipynb` notebook independently for experimenting with the toy dataset in pitfall 2 explanation.

## How to see the results:
Run the mlflow UI by running the following command in the root directory of the project:
```
mlflow ui
```
You can view the UI using URL `http://localhost:5000`.