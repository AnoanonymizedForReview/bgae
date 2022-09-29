import argparse
from pathlib import Path

import torch

from bgae.model import Model
from bgae.utils.analysis import Analyser
from bgae.utils.data_loaders import Dataset, load_dataset
from bgae.utils.extras import (dict_to_print_str, load_config,
                               two_level_dict_to_print_str)


def train(config):
    print(("Running with configuration: ", config))
    analysis_task = config["analyse_for"]
    train_data, val_data, test_data, gdc_data = load_dataset(
        Dataset.__getitem__(config["target_dataset"]),
        config
    )
    X, Y = train_data.x, train_data.y
    inp_1 = X, train_data.pos_edge_label_index, train_data.pos_edge_label
    inp_2 = X, gdc_data.edge_index, gdc_data.edge_attr

    model = Model(x_dim=X.shape[1], **config).to("cuda")
    analyser = Analyser(
        n_cls=len(Y.unique()),
        labels_true=Y,
        train_mask=train_data.train_mask,
        test_mask=train_data.test_mask,
        runs_path=Path(config["runs_path"])
    )

    ##########TRAIN##############
    for epoch in range(1, 10001):
        _, loss_terms, _ = model.training_loop(inp_1, inp_2)

        if epoch % 20 == 0:
            ##########TEST##########
            with torch.no_grad():
                model.eval()
                embb1, _ = model.encoder(*inp_1)
                embb2, _ = model.encoder(*inp_2)
                embb = model.fused_z(embb1, embb2)
                embb_np = embb.detach().cpu().numpy()

                if analysis_task == "link_prediction":
                    analysis_scores = {
                        analysis_task: model.test_link_prediction(
                            embb,
                            test_data.pos_edge_label_index,
                            test_data.neg_edge_label_index
                        )
                    }
                else:
                    analysis_scores = analyser.run_for_task(
                        epoch,
                        embb_np,
                        analysis_task
                    )

                print(
                    f'Epoch: {epoch:04d}\t' +
                    dict_to_print_str(loss_terms) +
                    two_level_dict_to_print_str(analysis_scores)
                )
                analyser.write_scalars_to_tensorboard(
                    analysis_scores, epoch=epoch)


if __name__ == "__main__":
    loaded_config = load_config(Path(__file__).parent / "dataset_config.yaml")
    target_dataset = loaded_config["target_dataset"]
    assert len([m for m in Dataset if m.value == target_dataset]) == 1, \
        "target dataset must be a string of type Dataset"
    dataset_config = loaded_config[target_dataset]
    dataset_config["target_dataset"] = target_dataset

    parser = argparse.ArgumentParser(description='BGAE on a given dataset.')
    default_log_path = f"{Path(__file__).parent}/runs/dataset/{target_dataset}"
    parser.add_argument(
        "--runs_dir", default=default_log_path
    )
    args = parser.parse_args()

    dataset_config["runs_path"] = Path(args.runs_dir)
    train(dataset_config)
