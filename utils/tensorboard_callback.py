from typing import Dict

from ray.tune.logger import TBXLoggerCallback
from ray.tune.logger.logger import logger
from ray.tune.trial import Trial
from ray.tune.utils import flatten_dict
from ray.util import log_once


class TensorboardLoggerCallback(TBXLoggerCallback):
    """Tensorboard logger interface. I had to write it down coz the
    default one used tensorboardX"""

    def __init__(self):
        try:
            from torch.utils.tensorboard import SummaryWriter
            self._summary_writer_cls = SummaryWriter
        except ImportError:
            if log_once("pytorch-tb-install"):
                logger.info(
                    "pip install 'ray[tune]' to see TensorBoard files.")
            raise
        self._trial_writer: Dict["Trial", SummaryWriter] = {}
        self._trial_result: Dict["Trial", Dict] = {}

    def _try_log_hparams(self, trial: "Trial", result: Dict):
        # TBX currently errors if the hparams value is None.
        flat_params = flatten_dict(trial.evaluated_params)
        scrubbed_params = {
            k: v
            for k, v in flat_params.items()
            if isinstance(v, self.VALID_HPARAMS)
        }

        removed = {
            k: v
            for k, v in flat_params.items()
            if not isinstance(v, self.VALID_HPARAMS)
        }
        print((flat_params, scrubbed_params, removed))

        if removed:
            logger.info(
                "Removed the following hyperparameter values when "
                "logging to tensorboard: %s", str(removed))

        from torch.utils.tensorboard.summary import hparams
        try:
            experiment_tag, session_start_tag, session_end_tag = hparams(
                hparam_dict=scrubbed_params, metric_dict=result)
            self._trial_writer[trial].file_writer.add_summary(experiment_tag)
            self._trial_writer[trial].file_writer.add_summary(
                session_start_tag)
            self._trial_writer[trial].file_writer.add_summary(session_end_tag)
        except Exception:
            logger.exception("TensorboardX failed to log hparams. "
                             "This may be due to an unsupported type "
                             "in the hyperparameter values.")
