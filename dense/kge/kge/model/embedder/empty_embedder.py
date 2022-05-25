from torch import Tensor
import torch.nn
import torch.nn.functional

from kge import Config, Dataset
from kge.job import Job
from kge.model import KgeEmbedder
from kge.misc import round_to_points

from typing import List


class EmptyEmbedder(KgeEmbedder):
    def __init__(
        self,
        config: Config,
        dataset: Dataset,
        configuration_key: str,
        vocab_size: int,
        init_for_load_only=False,
    ):
        super().__init__(
            config, dataset, configuration_key, init_for_load_only=init_for_load_only
        )
        self.vocab_size = vocab_size
        self.scorer = None

    def prepare_job(self, job: Job, **kwargs):
        from kge.job import TrainingJob
        super().prepare_job(job, **kwargs)

    def embed(self, indexes: Tensor) -> Tensor:
        return indexes

    def embed_all(self) -> Tensor:
        return torch.arange(
                self.vocab_size, dtype=torch.long, device=self.scorer.embeddings.weight.device
            )