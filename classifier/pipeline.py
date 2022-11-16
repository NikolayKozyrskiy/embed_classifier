import contextlib
from typing import Union, List, Tuple, Type, Dict, Optional, NamedTuple, TYPE_CHECKING

import torch
from torch import Tensor
import torch.nn.functional as F
from matches.shortcuts.dag import ComputationGraph, graph_node

from .models import AE, MPL, AEOutput
from .config import DatasetName, TrainSetup, AEArchitecture

if TYPE_CHECKING:
    from .config import EClrConfig


class EmbedClassifierOutput(NamedTuple):
    logits: Optional[Tensor]
    embeddings: Optional[Tensor]
    reconstructed_img: Optional[Tensor]
    mu: Optional[Tensor]
    log_var: Optional[Tensor]


class EmbedClassifierPipeline(ComputationGraph):
    def __init__(
        self,
        config: "EClrConfig",
        ae: AE,
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
    def embeddings(self) -> AEOutput:
        return self.ae.embeddings(self.gt_images())

    @graph_node
    def reconstruct_input(self) -> AEOutput:
        return self.ae(self.gt_images())

    @graph_node
    def sample(self) -> Tensor:
        return self.ae.sample(6, device=self.batch["label"].device)

    @graph_node
    def predict_labels(self) -> Tensor:
        with torch.no_grad():
            self.ae.eval()
            embeddings = self.embeddings().embeddings
        return self.classifier(embeddings)

    @graph_node
    def predict_all(self) -> EmbedClassifierOutput:
        ae_res: AEOutput = self.reconstruct_input()
        logits = self.classifier(ae_res.embeddings)
        return EmbedClassifierOutput(
            logits,
            ae_res.embeddings,
            ae_res.reconstructed_img,
            ae_res.mu,
            ae_res.log_var,
        )


def process_batch(batch: List[Tensor]) -> Dict[str, Union[Tensor, str]]:
    batch_ = {}
    batch_["image"] = batch[0]
    batch_["label"] = batch[1]
    batch_["name"] = batch[2]
    return batch_


def pipeline_from_config(config: "EClrConfig", device: str) -> EmbedClassifierPipeline:
    if config.ae_architecture == AEArchitecture.VANILLA:
        from .models.ae_vanilla import ae_from_config
    elif config.ae_architecture == AEArchitecture.RESNET18:
        from .models.ae_resnet import ae_from_config
    else:
        raise NotImplementedError
    ae = ae_from_config(config).to(device)
    classes_num = 10 if config.dataset_name == DatasetName.CIFAR10 else 100
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
    return EmbedClassifierPipeline(config, ae, clr, classes_num=classes_num)
