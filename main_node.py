from enum import Enum

import mlflow
import typer

from benchmarks.community import Community
from benchmarks.infection import Infection


class Experiment(str, Enum):
    infection = "infection"
    community = "community"


def main(experiment: Experiment = typer.Argument(..., help="Dataset to use"),
         sample_count: int = typer.Option(10, help='How many times to retry the whole experiment'),
         num_layers: int = typer.Option(4, help='Number of layers in the GNN model'),
         concat_features: bool = typer.Option(True,
                                              help='Concat embeddings of each convolutional layer for final fc layers'),
         conv_type: str = typer.Option('GraphConv',
                                       help="Convolution class. Can be GCNConv or GraphConv"),
         ):
    mlflow.set_experiment(experiment.value)
    if experiment == Experiment.infection:
        benchmark_class = Infection
    if experiment == Experiment.community:
        benchmark_class = Community
    benchmark = benchmark_class(sample_count, num_layers, concat_features, conv_type)
    benchmark.run()


if __name__ == "__main__":
    typer.run(main)
