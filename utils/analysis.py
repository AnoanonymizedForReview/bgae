import pathlib
from datetime import datetime

from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, normalized_mutual_info_score, f1_score, adjusted_rand_score
from torch.utils.tensorboard import SummaryWriter

from bgae.utils.extras import map_labels


class Analyser:
    def __init__(self, n_cls=-1, labels_true=None, train_mask=None, test_mask=None, k=5, runs_path: pathlib.Path=None):
        if runs_path is None:
            self.runs_path = pathlib.Path().absolute().joinpath("runs/" + datetime.now().strftime('%b%d_%H-%M-%S'))
        else:
            self.runs_path = runs_path
        self.n_cls = n_cls
        self.labels_true = labels_true.detach().cpu().numpy()
        self.train_mask = train_mask.detach().cpu().numpy()
        self.test_mask = test_mask.detach().cpu().numpy()
        self.writer = SummaryWriter(log_dir=str(self.runs_path))
        self.k = 5
        self.top_k = {"clustering": {}, "classification": {}}

    def generate_top_k_scores(self, scores_dict, task):
        for k, v in scores_dict.items():
            metric_key = f"{k}_top_k"
            if metric_key in self.top_k[task].keys():
                if len(self.top_k[task][metric_key]) < self.k:
                    self.top_k[task][metric_key].append(v)
                    self.top_k[task][metric_key].sort(reverse=True)
                else:
                    if v > self.top_k[task][metric_key][-1]:
                        self.top_k[task][metric_key].pop()
                        self.top_k[task][metric_key].append(v)
                        self.top_k[task][metric_key].sort(reverse=True)
                    else:
                        pass
            else:
                self.top_k[task][metric_key] = [v]

    def run(self, epoch, embb):
        clustering_scores, clusters_pred = self.test_clustering(embb)
        classification_scores, classes_pred = self.test_classification(embb)

        self.generate_top_k_scores(clustering_scores, "clustering")
        self.generate_top_k_scores(classification_scores, "clustering")

        clustering_scores.update({k: sum(v) / len(v) for k, v in self.top_k["clustering"].items()})
        classification_scores.update({k: sum(v) / len(v) for k, v in self.top_k["classification"].items()})

        # self.writer.add_embedding(
        #     mat=embb, global_step=epoch, metadata_header=["true", "clusters"],
        #     metadata=np.concatenate((self.labels_true[:, None], clusters_pred[:, None]), axis=1).tolist()
        # )

        return {
            "clustering": clustering_scores,
            "classification": classification_scores
        }

    def run_for_task(self, epoch, embb, task):
        assert task in ["classification", "clustering"]
        return {task: getattr(self, "test_" + task)(embb)[0]}

    def write_heatmap(self, mat, epoch):
        normalized_mat = (mat / mat.max(dim=0)[0]).abs()
        normalized_mat = normalized_mat.reshape(1, normalized_mat.shape[0], normalized_mat.shape[1])
        self.writer.add_image("C", normalized_mat, epoch, dataformats="CHW")

    def test_clustering(self, embb):
        kmeans = KMeans(n_clusters=self.n_cls, n_init=10)  # Note: changed from 25
        # gmm = GaussianMixture(n_components=self.K, n_init=30)
        labels_pred = kmeans.fit_predict(embb)
        labels_pred = map_labels(labels_pred, self.labels_true)[1][labels_pred]
        acc = accuracy_score(self.labels_true, labels_pred)
        nmi = normalized_mutual_info_score(self.labels_true, labels_pred, average_method="arithmetic")
        ari = adjusted_rand_score(self.labels_true, labels_pred)
        f1_micro = f1_score(self.labels_true, labels_pred, average="micro")
        f1_macro = f1_score(self.labels_true, labels_pred, average="macro")
        return {"acc": acc, "nmi": nmi, "ari": ari, "f1_mic": f1_micro, "f1_mac": f1_macro}, labels_pred

    def test_classification(self, embb):
        clf = LogisticRegression(solver='lbfgs', multi_class='auto', max_iter=1000).fit(
            embb[self.train_mask], self.labels_true[self.train_mask])
        labels_pred = clf.predict(embb[self.test_mask]).astype(int)
        f1_micro = f1_score(self.labels_true[self.test_mask], labels_pred, average="micro")
        f1_macro = f1_score(self.labels_true[self.test_mask], labels_pred, average="macro")
        return {"f1_mic": f1_micro, "f1_mac": f1_macro}, labels_pred

    def write_scalars_to_tensorboard(self, scalars_dict, epoch):
        [self.writer.add_scalar(main_kw + ":" + child_kw, scalars_dict[main_kw][child_kw], epoch)
         for main_kw in scalars_dict.keys() for child_kw in scalars_dict[main_kw].keys()]
