import contextlib
from typing import Union, List, Tuple, Type, Dict, Optional, NamedTuple

import torch
from torch import Tensor
import torch.nn.functional as F
from matches.shortcuts.dag import ComputationGraph, graph_node

from .config import EClrConfig, DatasetName, TrainSetup
from .models import ae_from_config, AutoEncoder, AutoEncoderOutput, MPL


class EmbedClassifierOutput(NamedTuple):
    logits: Optional[Tensor]
    embeddings: Optional[Tensor]
    reconstructed_img: Optional[Tensor]


class EmbedClassifierPipeline(ComputationGraph):
    def __init__(
        self,
        config: EClrConfig,
        ae: AutoEncoder,
        classifier: Optional[MPL],
        classes_num: int,
    ):
        super().__init__()
        self.config = config
        self.ae = ae
        self.classifier = classifier
        self.classes_num = classes_num
        self.batch = None

    @contextlib.contextmanager
    def batch_scope(self, batch: List[Tensor]):
        try:
            with self.cache_scope():
                self.batch = process_batch(batch)
                yield
        finally:
            self.batch = None

    @graph_node
    def gt_images(self) -> Tensor:
        img = self.batch["image"]
        return img

    @graph_node
    def labels(self) -> Tensor:
        label = self.batch["label"]
        return label

    @graph_node
    @torch.no_grad()
    def embeddings(self) -> AutoEncoderOutput:
        return self.ae.embeddings(self.gt_images())

    @graph_node
    def reconstruct_input(self) -> AutoEncoderOutput:
        return self.ae(self.gt_images())

    @graph_node
    def predict_labels(self) -> Tensor:
        embeddings = self.embeddings().embeddings
        return self.classifier(embeddings)

    @graph_node
    def predict_all(self) -> EmbedClassifierOutput:
        ae_res: AutoEncoderOutput = self.reconstruct_input()
        logits = self.classifier(ae_res.embeddings)
        return EmbedClassifierOutput(
            logits, ae_res.embeddings, ae_res.reconstructed_img
        )


def process_batch(batch: List[Tensor]) -> Dict[str, Union[Tensor, str]]:
    batch_ = {}
    batch_["image"] = batch[0]
    batch_["label"] = batch[1]
    batch_["name"] = batch[2]
    return batch_


def pipeline_from_config(config: EClrConfig, device: str) -> EmbedClassifierPipeline:
    classes_num = 10 if config.dataset_name == DatasetName.CIFAR10 else 100
    ae = ae_from_config(config)
    if config.train_setup == TrainSetup.AE:
        clr = None
    else:
        clr = MPL(
            latent_dim=config.latent_dim,
            feature_size_lst=config.mlp_feature_size_lst,
            classes_num=classes_num,
            activation_fn=config.mlp_activation_fn,
            output_activation_fn=config.mlp_output_activation_fn,
        ).to(device)
        if config.ae_checkpoint_path is not None:
            ae.load_state_dict(
                torch.load(config.ae_checkpoint_path, map_location="cpu")
            )
    ae = ae.to(device)
    return EmbedClassifierPipeline(config, ae, clr, classes_num=classes_num)
