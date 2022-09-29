from enum import Enum
from pathlib import Path
from typing import List

import torch
from torch_geometric.data import Data
from torch_geometric.datasets import Amazon, Coauthor, Planetoid, WikiCS
from torch_geometric.transforms import (GDC, AddSelfLoops, Compose,
                                        NormalizeFeatures, RandomLinkSplit,
                                        RandomNodeSplit, ToDevice,
                                        ToUndirected)


class Dataset(Enum):
    Cora = "Cora"
    CiteSeer = "CiteSeer"
    PubMed = "PubMed"
    WikiCS = "WikiCS"
    CoauthorCS = "CoauthorCS"
    CoauthorPhysics = "CoauthorPhysics"
    AmazonPhoto = "AmazonPhoto"
    AmazonComputers = "AmazonComputers"


def load_dataset(dataset: Dataset, config):
    split_nodes = RandomNodeSplit(num_val=0.1, num_test=0.8)
    to_device = ToDevice("cuda")
    common_transform: List = [AddSelfLoops(), ToUndirected()]

    gdc_config = config["gdc"]

    data_path = Path(__file__).parent.parent / "_data_"
    gdc = GDC(
        diffusion_kwargs=gdc_config["diffusion"],
        sparsification_kwargs=gdc_config["sparsification"],
        exact=(gdc_config["diffusion"]["method"] != "ppr")
    )

    if dataset in [Dataset.Cora, Dataset.CiteSeer, Dataset.PubMed]:
        transforms = common_transform + [NormalizeFeatures(), to_device]
        data = Planetoid(
            root=str(data_path / "planetoid"), name=dataset.name,
            transform=Compose(transforms)
        )[0]
    elif dataset in [Dataset.CoauthorCS, Dataset.CoauthorPhysics]:
        transforms = common_transform + \
            [NormalizeFeatures(), split_nodes, to_device]
        data = Coauthor(
            root=str(data_path / "coauthor"),
            name="cs" if dataset == Dataset.CoauthorCS else "physics",
            transform=Compose(transforms)
        )[0]
    elif dataset in [Dataset.AmazonPhoto, Dataset.AmazonComputers]:
        transforms = common_transform + \
            [NormalizeFeatures(), split_nodes, to_device]
        data = Amazon(
            root=str(data_path / "amazon"),
            name="computers" if dataset.name == Dataset.AmazonComputers else "photo",
            transform=Compose(transforms)
        )[0]
    elif dataset in [Dataset.WikiCS]:
        transforms = common_transform + [to_device]
        data = WikiCS(
            root=str(data_path / "wikics"),
            transform=Compose(transforms)
        )[0]
        # for sake for demonstration, we simply pick the first split out of 20
        data.train_mask = data.train_mask[:, 0]
    else:
        raise Exception("Unknown dataset: " + dataset.name)

    if config["analyse_for"] == "link_prediction":
        edge_split = RandomLinkSplit(
            num_val=0.05,
            num_test=0.1,
            is_undirected=True,
            add_negative_train_samples=True,
            split_labels=True
        )
    else:
        edge_split = RandomLinkSplit(
            num_val=0.,
            num_test=0.,
            is_undirected=False,
            add_negative_train_samples=True,
            split_labels=True
        )

    train_data, val_data, test_data = edge_split(data)

    gdc_data = Data(
        x=torch.empty((train_data.x.shape[0], 1)),
        edge_index=torch.clone(train_data.edge_index)
    )

    return train_data, val_data, test_data, gdc(gdc_data)
