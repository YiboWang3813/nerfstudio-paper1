"""Full-slice sampling with checkpointed permutation, cursor and RNG state."""

from dataclasses import dataclass, field
from typing import Type

import torch

from nerfstudio.data.datamanagers.base_datamanager import DataManager, DataManagerConfig
from nerfstudio.data.dataparsers.ultrasound_dataparser import UltrasoundDataParserConfig
from nerfstudio.data.datasets.ultrasound_dataset import UltrasoundDataset


@dataclass
class UltrasoundDataManagerConfig(DataManagerConfig):
    _target: Type = field(default_factory=lambda: UltrasoundDataManager)
    dataparser: UltrasoundDataParserConfig = field(default_factory=UltrasoundDataParserConfig)
    seed: int = 42


class _EvalLoader:
    def __init__(self, manager):
        self.manager = manager

    def __len__(self):
        return len(self.manager.eval_dataset)

    def __iter__(self):
        for index in range(len(self)):
            yield self.manager._get(self.manager.eval_dataset, index)


class UltrasoundDataManager(DataManager):
    def __init__(self, config, device="cpu", test_mode="val", world_size=1, local_rank=0, **kwargs):
        if world_size != 1:
            raise ValueError("Distributed ultrasound sampling has not been validated.")
        self.config, self.device, self.test_mode = config, device, test_mode
        super().__init__()
        if config.data is not None:
            config.dataparser.data = config.data
        parser = config.dataparser.setup()
        self.train_dataparser_outputs = parser.get_dataparser_outputs("train")
        self.train_dataset = UltrasoundDataset(self.train_dataparser_outputs)
        self.eval_dataset = UltrasoundDataset(parser.get_dataparser_outputs("test"))
        self.train_dataset.cameras = self.train_dataset.cameras.to(device)
        self.eval_dataset.cameras = self.eval_dataset.cameras.to(device)
        self._generator = torch.Generator().manual_seed(config.seed)
        self._order = torch.randperm(len(self.train_dataset), generator=self._generator)
        self._cursor = 0
        self.fixed_indices_eval_dataloader = _EvalLoader(self)
        self.eval_dataloader = self.fixed_indices_eval_dataloader
        self.eval_image_dataloader = self.fixed_indices_eval_dataloader

    def setup_train(self):
        pass

    def setup_eval(self):
        pass

    def _get(self, dataset, index):
        batch = dataset[index]
        batch["image"] = batch["image"].to(self.device)
        return dataset.cameras[index : index + 1], batch

    def next_train(self, step):
        if self._cursor == len(self._order):
            self._order = torch.randperm(len(self.train_dataset), generator=self._generator)
            self._cursor = 0
        index = int(self._order[self._cursor])
        self._cursor += 1
        self.train_count += 1
        return self._get(self.train_dataset, index)

    def next_eval(self, step):
        return self.next_eval_image(step)

    def next_eval_image(self, step):
        index = self.eval_count % len(self.eval_dataset)
        self.eval_count += 1
        return self._get(self.eval_dataset, index)

    def get_extra_state(self):
        return {
            "rng": self._generator.get_state(),
            "order": self._order.clone(),
            "cursor": self._cursor,
            "train_count": self.train_count,
            "eval_count": self.eval_count,
            "manifest_sha256": self.train_dataset.metadata["manifest_sha256"],
            "parser_identity_sha256": self.train_dataset.metadata["parser_identity_sha256"],
        }

    def set_extra_state(self, state):
        if state.get("parser_identity_sha256") != self.train_dataset.metadata["parser_identity_sha256"]:
            raise ValueError("Checkpoint dataset or pose geometry differs from the current dataset parser.")
        self._generator.set_state(state["rng"].cpu())
        self._order, self._cursor = state["order"].cpu(), state["cursor"]
        self.train_count, self.eval_count = state["train_count"], state["eval_count"]

    def get_param_groups(self):
        return {}

    def get_train_rays_per_batch(self):
        g = self.train_dataset.metadata["geometry"]
        return g["height"] * g["width"]

    def get_eval_rays_per_batch(self):
        return self.get_train_rays_per_batch()

    def get_datapath(self):
        return self.config.dataparser.data
