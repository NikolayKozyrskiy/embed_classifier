import contextlib
from typing import Callable, List, Tuple, Type, Dict, Optional, NamedTuple

import torch
from torch import Tensor
import torch.nn.functional as F
from matches.shortcuts.dag import ComputationGraph, graph_node

from . import EClrConfig
from .models import AutoEncoder, AutoEncoderOutput, MPL


class EmbedClassifierOutput(NamedTuple):
    logits: Optional[Tensor]
    embeddings: Optional[Tensor]
    reconstructed_img: Optional[Tensor]


class EmbedClassifierPipeline(ComputationGraph):
    def __init__(self, config: EClrConfig, ae: AutoEncoder, classifier: MPL):
        super().__init__()
        self.config = config
        self.ae = ae
        self.classifier = classifier
        self.batch = None

    @contextlib.contextmanager
    def batch_scope(self, batch: Tensor):
        try:
            with self.cache_scope():
                self.batch = process_batch(batch)
                yield
        finally:
            self.batch = None

    @graph_node
    def embeddings(self) -> AutoEncoderOutput:
        return self.ae.embeddings(self.batch["image"])

    @graph_node
    def reconstruct_input(self) -> AutoEncoderOutput:
        return self.ae(self.batch["image"])

    @graph_node
    def predict_labels(self) -> Tensor:
        return self.classifier(self.embeddings(self.batch["image"]).embeddings)

    @graph_node
    def predict_all(self) -> EmbedClassifierOutput:
        ae_res: AutoEncoderOutput = self.reconstruct_input()
        logits = self.classifier(ae_res.embeddings)
        return EmbedClassifierOutput(
            logits, ae_res.embeddings, ae_res.reconstructed_img
        )


def process_batch(batch: Tensor):
    batch_ = {}
    batch_["image"] = batch[0]
    batch_["label"] = batch[1]
    return batch_
