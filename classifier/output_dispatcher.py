from typing import Dict, NamedTuple, List

import torch
from torch import Tensor
from torch.distributions import Normal, kl_divergence
import torch.nn.functional as F

from .pipeline import EmbedClassifierPipeline
from .common.metrics import psnr


class LossDispatchResult(NamedTuple):
    computed_values: Dict[str, torch.Tensor]
    aggregated: torch.Tensor


class MetricsDispatchResult(NamedTuple):
    computed_values: Dict[str, torch.Tensor]


class OutputDispatcher:
    def __init__(
        self,
        loss_aggregation_weigths: Dict[str, float],
        metrics: List[str],
    ):
        self.loss_aggregation_weigths = loss_aggregation_weigths
        self.metrics = metrics

        self._prepare_loss_fns()
        self._prepare_metric_fns()

    @staticmethod
    def recon__mse(pipeline: EmbedClassifierPipeline):
        pred = pipeline.reconstruct_input().reconstructed_img
        gt = pipeline.gt_images()
        return F.mse_loss(pred, gt, reduction="none").flatten(1).sum(-1)

    @staticmethod
    def recon__psnr(pipeline: EmbedClassifierPipeline):
        pred = pipeline.reconstruct_input().reconstructed_img
        gt = pipeline.gt_images()
        return psnr(pred, gt).flatten(1).mean(-1)

    @staticmethod
    def vae__kl(pipeline: EmbedClassifierPipeline):
        vae_pred = pipeline.embeddings()
        std = torch.exp(vae_pred.log_var / 2)
        return kl_divergence(
            Normal(vae_pred.mu, std),
            Normal(torch.zeros_like(vae_pred.mu), torch.ones_like(std)),
        )

    @staticmethod
    def clr__cross_entropy(pipeline: EmbedClassifierPipeline):
        pred = pipeline.predict_labels()
        gt = pipeline.labels()
        return F.cross_entropy(pred, gt, reduction="none")

    @staticmethod
    def clr__accuracy(pipeline: EmbedClassifierPipeline):
        pred = pipeline.predict_labels()
        gt = pipeline.labels()
        correct = pred.max(1)[1].eq(gt).float()
        return correct

    def compute_losses(self, pipeline: EmbedClassifierPipeline) -> LossDispatchResult:
        loss_values = {
            name.replace("__", "/"): getattr(self, name)(pipeline)
            for name in self.train_loss_fn
        }

        with torch.no_grad():
            loss_values.update(
                {
                    fn_name.replace("__", "/"): getattr(self, fn_name)(pipeline)
                    for fn_name in self.eval_loss_fn
                }
            )

        for k in list(loss_values.keys()):
            if isinstance(loss_values[k], dict):
                d = loss_values.pop(k)
                loss_values.update({f"{k}.{in_k}": v for in_k, v in d.items()})

        losses, weights = [], []
        for name, w in self.loss_aggregation_weigths.items():
            losses.append(loss_values[name].mean())
            weights.append(w)
        if len(losses) > 0:
            losses = torch.stack(losses)
            weights = losses.new_tensor(weights)

            aggregated = (losses * weights).sum() / weights.sum()
        else:
            aggregated = torch.tensor(0)

        loss_values["_total"] = aggregated
        return LossDispatchResult(loss_values, aggregated)

    @torch.no_grad()
    def compute_metrics(
        self, pipeline: EmbedClassifierPipeline
    ) -> MetricsDispatchResult:
        metric_values = {
            name.replace("__", "/"): getattr(self, name)(pipeline)
            for name in self.metric_fn
        }

        return MetricsDispatchResult(metric_values)

    def _prepare_metric_fns(self):
        self.metric_fn = list(
            {n.replace("/", "__").split(".")[0] for n in self.metrics}
        )

    def _prepare_loss_fns(self):
        self.train_loss_fn = list(
            {
                n.replace("/", "__").split(".")[0]
                for n, weight in self.loss_aggregation_weigths.items()
                if weight != 0
            }
        )

        self.eval_loss_fn = list(
            {
                n.replace("/", "__").split(".")[0]
                for n, weight in self.loss_aggregation_weigths.items()
                if weight == 0
            }
        )
        self.eval_loss_fn = [
            n for n in self.eval_loss_fn if n not in self.train_loss_fn
        ]
        self.loss_aggregation_weigths = {
            name: w for name, w in self.loss_aggregation_weigths.items() if w != 0
        }
        return None


def filter_and_uncollate(batch_values: Dict[str, Tensor], pipeline):
    batch_values = {k: v.tolist() for k, v in batch_values.items() if k != "_total"}
    return uncollate(batch_values)
    # return batch_values


def uncollate(params):
    params = [
        dict(zip(params.keys(), t)) for t in zip(*[params[k] for k in params.keys()])
    ]
    return params
