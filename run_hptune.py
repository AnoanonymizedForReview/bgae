import argparse
from functools import partial
from pathlib import Path

import torch
from ray import tune
from ray.air import RunConfig
from ray.tune import CLIReporter
from ray.tune.logger import TBXLoggerCallback
from ray.tune.schedulers import ASHAScheduler

from bgae.model import Model
from bgae.utils.analysis import Analyser
from bgae.utils.data_loaders import Dataset, load_dataset
from bgae.utils.extras import (TARGET_METRICS, build_search_space,
                               dict_to_print_str,
                               join_two_level_dict_to_one_level_dict,
                               load_config, two_level_dict_to_print_str)


def train(config, target_metric):
    trial_logs_file_path = Path(tune.get_trial_dir()).parent / \
        f"{Path(tune.get_trial_dir()).parts[-1]}.txt"
    with trial_logs_file_path.open('a') as f:
        print(("Running with configuration: ", config), file=f)
    train_data, val_data, test_data, gdc_data = load_dataset(
        Dataset.__getitem__(config["target_dataset"]), config)

    X, Y = train_data.x, train_data.y
    inp_1 = X, train_data.pos_edge_label_index, train_data.pos_edge_label
    inp_2 = X, gdc_data.edge_index, gdc_data.edge_attr

    target_metric_k_best = []  # for potential early stopping

    model = Model(x_dim=X.shape[1], **config).to("cuda")
    analyser = Analyser(
        n_cls=len(Y.unique()),
        labels_true=Y,
        train_mask=train_data.train_mask,
        test_mask=train_data.test_mask
    )

    ##########TRAIN##############
    for epoch in range(1, 10001):
        # X1 = augment_X(X, 0.2)
        _, loss_terms, _ = model.training_loop(inp_1, inp_2)

        if epoch % 10 == 0:
            ##########TEST##########
            with torch.no_grad():
                model.eval()
                embb1, _ = model.encoder(*inp_1)
                embb2, _ = model.encoder(*inp_2)
                embb = model.fused_z(embb1, embb2)

                analysis_scores = analyser.run(
                    epoch, embb.detach().cpu().numpy())
                if config["analyse_for"] == "link_prediction":
                    link_pred_scores = model.test_link_prediction(
                        embb,
                        test_data.pos_edge_label_index,
                        test_data.neg_edge_label_index
                    )
                else:
                    link_pred_scores = {"auc": 0, "ap": 0}
                analysis_scores["link_prediction"] = link_pred_scores
                with trial_logs_file_path.open('a') as f:
                    print(f'Epoch: {epoch:04d}\t' + dict_to_print_str(loss_terms) + two_level_dict_to_print_str(
                        analysis_scores), file=f)
                flattened_scores_dict = join_two_level_dict_to_one_level_dict(
                    analysis_scores
                )

                if len(target_metric_k_best) < 5:
                    target_metric_k_best.append(
                        flattened_scores_dict[target_metric]
                    )
                    target_metric_k_best.sort(reverse=True)
                else:
                    if flattened_scores_dict[target_metric] > target_metric_k_best[-1]:
                        target_metric_k_best.pop()
                        target_metric_k_best.append(
                            flattened_scores_dict[target_metric])
                        target_metric_k_best.sort(reverse=True)
                    else:
                        pass

                flattened_scores_dict[target_metric + "_topk"] = sum(
                    target_metric_k_best
                ) / len(target_metric_k_best)
                tune.report(**flattened_scores_dict)


def run_trials(num_samples, target_metric, config: dict, runs_dir: str):
    merged_space = build_search_space(config)

    scheduler = ASHAScheduler(
        metric=target_metric,
        mode="max",
        max_t=150,  # every iteration has 10 epochs, so max_epochs=max_t*epochs_per_iter
        grace_period=150,  # reduce grace period to utilize early stopping.
        reduction_factor=2
    )

    reporter = CLIReporter(
        max_report_frequency=15,
        parameter_columns=list(merged_space.keys()),
        metric_columns=(
            [x for y in list(map(lambda x: [x, x + "_topk"], TARGET_METRICS.values())) for x in y] +
            ["training_iteration"]
        ),
        max_column_length=100
    )

    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(partial(train, target_metric=target_metric)),
            {"cpu": 16, "gpu": 1}
        ),
        tune_config=tune.TuneConfig(
            num_samples=num_samples,
            scheduler=scheduler
        ),
        run_config=RunConfig(
            progress_reporter=reporter,
            verbose=1,
            local_dir=runs_dir,
            callbacks=[TBXLoggerCallback()]
        ),
        param_space=merged_space)

    results = tuner.fit()
    print("Best config: ", results.get_best_result(
        metric=target_metric, mode="max"))


if __name__ == "__main__":
    loaded_config = load_config(Path(__file__).parent / "hptune_config.yaml")
    parser = argparse.ArgumentParser(description='HP Tuner.')
    default_log_path = f"{Path(__file__).parent}/runs/hptune"
    parser.add_argument("--runs_dir", default=default_log_path)
    args = parser.parse_args()

    for target_dataset in loaded_config["target_datasets"]:
        dataset_config = loaded_config[target_dataset]
        dataset_config["target_dataset"] = target_dataset
        run_trials(
            num_samples=1,
            target_metric=TARGET_METRICS[dataset_config["analyse_for"]],
            config=dataset_config,
            runs_dir=f"{args.runs_dir}/{target_dataset}_{dataset_config['analyse_for']}"
        )
