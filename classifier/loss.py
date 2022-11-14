from typing import Dict, NamedTuple

import torch
from torch import Tensor
import torch.nn.functional as F

from .pipeline import EmbedClassifierPipeline


class LossDispatchResult(NamedTuple):
    computed_values: Dict[str, torch.Tensor]
    aggregated: torch.Tensor


class LossesDispatcher:
    def __init__(
        self,
        loss_aggregation_weigths: Dict[str, float],
    ):
        self.loss_aggregation_weigths = loss_aggregation_weigths
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

    @staticmethod
    def ae__mse(pipeline: EmbedClassifierPipeline):
        z = 0.1 - pipeline.estimate_pose().translations[..., 2]
        mask = z.detach() > 0
        return (z * mask).sum() / (mask.sum() + 1e-7)

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


def filter_and_uncollate_multilosses(batch_losses, pipeline):
    batch_losses = {k: v.tolist() for k, v in batch_losses.items()}
    batch_losses = {
        "name": pipeline.batch["name"],
        **{
            k: v
            for k, v in batch_losses.items()
            if isinstance(v, list) and len(v) == len(pipeline.batch["name"])
        },
    }
    return uncollate(batch_losses)


def uncollate(params):
    params = [
        dict(zip(params.keys(), t)) for t in zip(*[params[k] for k in params.keys()])
    ]
    return params
