from collections import defaultdict
from collections.abc import Callable, Sequence
from typing import Any, Optional

import lightning as L
import numpy as np
import torch
from jaxtyping import Bool, Float, Int
from torch import nn
from torch.distributions import Distribution

from tsfm.loss.packed import (
    PackedDistributionLoss,
    PackedLoss,
    PackedNLLLoss,
)
from tsfm.module.norm import RMSNorm
from tsfm.module.position import (
    LearnedEmbedding,
    LearnedProjection,
)
from tsfm.optim import SchedulerType, get_scheduler
from tsfm.transform import (
    AddObservedMask,
    AddTimeIndex,
    AddVariateIndex,
    EvalCrop_AdaLength,
    EvalPad_AdaLength,
    EvalNextTokenPrediction,
    DummyValueImputation,
    ExtendMask,
    FlatPackCollection,
    FlatPackFields,
    GetPatchSize,
    ImputeTimeSeries,
    NextTokenPrediction,
    PackFields,
    PatchCrop,
    Patchify,
    SampleDimension,
    SelectFields,
    SequencifyField,
    Transformation,
)

import math

from .module import BasicModule
from tsfm.val.metrics import (
    MSE_mean,
    MAE_mean,
    MSE_median,
    MAE_median,
    MASE,
    MAPE,
    SMAPE,
    RMSE,
    NRMSE,
    ND,
    CRPS
)

class TransformerDecoderPretrain(L.LightningModule):
    seq_fields: tuple[str, ...] = (
        "target",
        "observed_mask",
        "time_id",
        "variate_id",
        "prediction_mask",
        "patch_size",
        "label",
        "label_observed_mask",
    )
    train_seq_fields: tuple[str, ...] = (
        "target",
        "observed_mask",
        "time_id",
        "variate_id",
        "prediction_mask",
        "patch_size",
    )
    pad_func_map: dict[str, Callable[[Sequence[int], np.dtype], np.ndarray]] = {
        "target": np.zeros,
        "observed_mask": np.zeros,
        "time_id": np.zeros,
        "variate_id": np.zeros,
        "prediction_mask": np.zeros,
        "patch_size": np.zeros,
    }
    eval_max_context_len: int = 36
    eval_max_prediction_len: int = 12
    
    def __init__(
        self,
        min_patches: int,
        min_mask_ratio: float,
        max_mask_ratio: float,
        num_training_steps: int,
        num_warmup_steps: int,
        max_dim: int = 1,
        module_kwargs: Optional[dict[str, Any]] = None,
        module: Optional[BasicModule] = None,
        num_samples: int = 100,
        beta1: float = 0.9,
        beta2: float = 0.98,
        loss_func: PackedDistributionLoss = PackedNLLLoss(),
        val_metric: Optional[PackedLoss | list[PackedLoss]] = [
            MSE_mean() ,MAE_mean(), MSE_median(), MAE_median(), MASE(), MAPE(), SMAPE(), RMSE(), NRMSE(), ND(), CRPS()
        ],
        lr: float = 1e-3,
        weight_decay: float = 1e-2,
        log_on_step: bool = False,
    ):
        assert (module is not None) or (
            module_kwargs is not None
        ), "if module is not provided, module_kwargs is required"
        assert (
            num_warmup_steps <= num_training_steps
        ), f"num_warmup_steps ({num_warmup_steps}) should be <= num_training_steps ({num_training_steps})."
        super().__init__()
        self.save_hyperparameters(ignore=["module"])
        self.module = BasicModule(**module_kwargs) if module is None else module
        print('initialization complete')
        
    def forward(
        self,
        target: Float[torch.Tensor, "*batch seq_len max_patch"],
        observed_mask: Bool[torch.Tensor, "*batch seq_len max_patch"],
        sample_id: Int[torch.Tensor, "*batch seq_len"],
        time_id: Int[torch.Tensor, "*batch seq_len"],
        variate_id: Int[torch.Tensor, "*batch seq_len"],
        patch_size: Int[torch.Tensor, "*batch seq_len"],
        prediction_mask: Int[torch.Tensor, "*batch seq_len"],
    ) -> Distribution:
        output = self.module(
            target=target,
            observed_mask=observed_mask,
            sample_id=sample_id,
            time_id=time_id,
            variate_id=variate_id,
            prediction_mask=prediction_mask,
            patch_size=patch_size,
        )
        return output
    
    def infer(
        self,
        target: Float[torch.Tensor, "*batch seq_len max_patch"],
        observed_mask: Bool[torch.Tensor, "*batch seq_len max_patch"],
        sample_id: Int[torch.Tensor, "*batch seq_len"],
        time_id: Int[torch.Tensor, "*batch seq_len"],
        variate_id: Int[torch.Tensor, "*batch seq_len"],
        patch_size: Int[torch.Tensor, "*batch seq_len"],
        prediction_mask: Int[torch.Tensor, "*batch seq_len"],
        max_iteration_num: int
    ) -> Distribution:
        # a fixed hyperparamter set for tsfm.data.EvalPadCollate
        context_length = self.eval_max_context_len
        
        context = target[:, :context_length, :]
        _target = context
        new_preds = None
        
        for i in range(max_iteration_num): 
            _observed_mask = observed_mask[:, : context_length + i, :]
            _sample_id = sample_id[:, : context_length + i]
            _time_id = time_id[:, : context_length + i]
            _variate_id = variate_id[:, : context_length + i]
            _prediction_mask = prediction_mask[:, : context_length + i]
            distr = self.module(
                _target,
                _observed_mask,
                _sample_id,
                _time_id,
                _variate_id,
                _prediction_mask,
                torch.ones_like(_time_id, dtype=torch.long) * 32,
            )
            preds = distr.sample(torch.Size((100,)))
            sampled_pred = torch.median(preds, dim=0).values
            # sampled_pred = preds.mean(dim=0)
            _target = torch.cat([_target, sampled_pred[:, -1:, :]], dim=1)
            new_preds = preds if new_preds is None else torch.cat([new_preds, preds[:, :, -1:, :]], dim=2)

        new_preds = new_preds.transpose(0, 1) # batch sample time features

        return distr, new_preds
        

    def training_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        
        output = self(
            **{field: batch[field] for field in list(self.train_seq_fields) + ["sample_id"]}
        )
        loss = self.hparams.loss_func(
            pred=output,
            target=batch["label"],
            observed_mask=batch["label_observed_mask"],
            prediction_mask=batch["prediction_mask"],
            sample_id=batch["sample_id"],
            variate_id=batch["variate_id"],
        )
        batch_size = (
            batch["sample_id"].max(dim=1).values.sum() if "sample_id" in batch else None
        )
        self.log(
            f"train/{self.hparams.loss_func.__class__.__name__}",
            loss,
            on_step=self.hparams.log_on_step,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
            batch_size=batch_size,
            rank_zero_only=True,
        )
        return loss
    
    def validation_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int, dataloader_idx: int = 0
    ) -> torch.Tensor:
        
        prediction_mask = batch["prediction_mask"]
        max_iteration_num = torch.max(torch.sum(prediction_mask, dim=1))
        
        # iterative inference
        distr, preds = self.infer(
            **{field: batch[field] for field in list(self.train_seq_fields) + ["sample_id"]},
            max_iteration_num=max_iteration_num
        )
        
        # truncate the sequence
        batch['label'] = batch['label'][:, 1 : (36 + max_iteration_num), :]
        batch['label_observed_mask'] = batch['label_observed_mask'][:, 1 : (36 + max_iteration_num), :]
        batch['prediction_mask'] = batch['prediction_mask'][:, 1 : (36 + max_iteration_num)]
        batch['sample_id'] = batch['sample_id'][:, 1 : (36 + max_iteration_num)]
        batch['variate_id'] = batch['variate_id'][:, 1 : (36 + max_iteration_num)]
        
        val_loss = self.hparams.loss_func(
            pred=distr,
            target=batch["label"],
            observed_mask=batch["label_observed_mask"],
            **{
                field: batch[field]
                for field in [
                    "prediction_mask",
                    "sample_id",
                    "variate_id",
                ]
            },
        )
        batch_size = (
            batch["sample_id"].max(dim=1).values.sum() if "sample_id" in batch else None
        )
        self.log(
            f"val/{self.hparams.loss_func.__class__.__name__}",
            val_loss,
            on_step=self.hparams.log_on_step,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
            batch_size=batch_size,
            rank_zero_only=True,
            add_dataloader_idx=True,
        )

        if self.hparams.val_metric is not None:
            val_metrics = (
                self.hparams.val_metric
                if isinstance(self.hparams.val_metric, list)
                else [self.hparams.val_metric]
            )
            for metric_func in val_metrics:
                metric = metric_func(
                    pred=preds,
                    target=batch["label"],
                    observed_mask=batch["label_observed_mask"],
                    **{
                        field: batch[field]
                        for field in [
                            "prediction_mask",
                            "sample_id",
                            "variate_id",
                        ]
                    },
                )

                self.log(
                    f"val/{metric_func.__class__.__name__}",
                    metric,
                    on_step=self.hparams.log_on_step,
                    on_epoch=True,
                    prog_bar=True,
                    logger=True,
                    sync_dist=True,
                    batch_size=batch_size,
                    rank_zero_only=True,
                    add_dataloader_idx=True,
                )

        return val_loss
    
    def configure_optimizers(self) -> dict:
        decay = set()
        no_decay = set()

        whitelist_params = (
            LearnedProjection,
            nn.Linear,
        )
        blacklist_params = (
            LearnedEmbedding,
            RMSNorm,
            nn.Embedding,
            nn.LayerNorm,
        )

        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                if not p.requires_grad:
                    continue

                fpn = f"{mn}.{pn}" if mn else pn
                if pn.endswith("bias"):
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_params):
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_params):
                    no_decay.add(fpn)

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert (
            len(inter_params) == 0
        ), f"parameters {str(inter_params)} made it into both decay/no_decay sets!"
        assert (
            len(param_dict.keys() - union_params) == 0
        ), f"parameters {str(param_dict.keys() - union_params)} were not separated into either decay/no_decay set!"

        optim_groups = [
            {
                "params": filter(
                    lambda p: p.requires_grad,
                    [param_dict[pn] for pn in sorted(list(decay))],
                ),
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": filter(
                    lambda p: p.requires_grad,
                    [param_dict[pn] for pn in sorted(list(no_decay))],
                ),
                "weight_decay": 0.0,
            },
        ]

        optimizer = torch.optim.AdamW(
            optim_groups,
            lr=self.hparams.lr,
            betas=(self.hparams.beta1, self.hparams.beta2),
            eps=1e-6,
        )
        scheduler = get_scheduler(
            SchedulerType.COSINE_WITH_RESTARTS,
            optimizer,
            num_warmup_steps=self.hparams.num_warmup_steps,
            num_training_steps=self.hparams.num_training_steps,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "train_loss",
                "interval": "step",
            },
        }
        
    @property
    def train_transform_map(self) -> dict[str, Callable[..., Transformation]]:
        def default_train_transform():
            return (
                SampleDimension(
                    max_dim=self.hparams.max_dim,
                    fields=("target",),
                    optional_fields=(),
                )
                + GetPatchSize(
                    min_time_patches=self.hparams.min_patches,
                    target_field="target",
                    patch_size=self.module.patch_size,
                    patch_size_constraints=None,
                    offset=True,
                )
                + PatchCrop(
                    min_time_patches=self.hparams.min_patches,
                    max_patches=self.module.max_seq_len,
                    will_flatten=True,
                    offset=True,
                    fields=("target",),
                    optional_fields=(),
                )
                + PackFields(
                    output_field="target",
                    fields=("target",),
                    feat=False,
                )
                + AddObservedMask(
                    fields=("target",),
                    optional_fields=(),
                    observed_mask_field="observed_mask",
                    collection_type=dict,
                )
                + ImputeTimeSeries(
                    fields=("target",),
                    optional_fields=(),
                    imputation_method=DummyValueImputation(value=0.0),
                )
                + Patchify(
                    max_patch_size=self.module.patch_size,
                    fields=("target", "observed_mask"),
                    optional_fields=(),
                )
                + NextTokenPrediction(
                    min_mask_ratio=self.hparams.min_mask_ratio,
                    max_mask_ratio=self.hparams.max_mask_ratio,
                    target_field="target",
                    prediction_mask_field="prediction_mask",
                    expected_ndim=3,
                )
                + AddVariateIndex(
                    fields=("target",),
                    optional_fields=(),
                    variate_id_field="variate_id",
                    expected_ndim=3,
                    max_dim=self.hparams.max_dim,
                    randomize=True,
                    collection_type=dict,
                )
                + AddTimeIndex(
                    fields=("target",),
                    optional_fields=(),
                    time_id_field="time_id",
                    expected_ndim=3,
                    collection_type=dict,
                )
                + FlatPackCollection(
                    field="variate_id",
                    feat=False,
                )
                + FlatPackCollection(
                    field="time_id",
                    feat=False,
                )
                + FlatPackCollection(
                    field="prediction_mask",
                    feat=False,
                )
                + FlatPackCollection(
                    field="observed_mask",
                    feat=True,
                )
                + FlatPackCollection(
                    field="label_observed_mask",
                    feat=True,
                )
                + FlatPackFields(
                    output_field="label",
                    fields=("label",),
                    optional_fields=(),
                    feat=True,
                )
                + FlatPackFields(
                    output_field="target",
                    fields=("target",),
                    optional_fields=(),
                    feat=True,
                )
                + SequencifyField(field="patch_size", target_field="target")
                + SelectFields(fields=list(self.seq_fields))
            )

        return defaultdict(lambda: default_train_transform)
    
    @property
    def val_transform_map(
        self,
    ) -> dict[str | type, Callable[..., Transformation]]:
        def default_val_transform(
            offset: int,
            distance: int,
            prediction_length: int,
            context_length: int,
            patch_size: int,
        ):
            return (
                SampleDimension(
                    max_dim=1,
                    fields=("target",),
                    optional_fields=(),
                )
                + GetPatchSize(
                    min_time_patches=2,
                    target_field="target",
                    patch_size=self.module.patch_size,
                    patch_size_constraints=None,
                    offset=True,
                )
                + EvalCrop_AdaLength(
                    offset,
                    distance,
                    prediction_length,
                    context_length,
                    fields=("target",),
                    optional_fields=(),
                )
                + PackFields(
                    output_field="target",
                    fields=("target",),
                )
                + EvalPad_AdaLength(
                    prediction_length=prediction_length,
                    context_length=context_length,
                    patch_size=self.module.patch_size,
                    fields=("target",),
                    optional_fields=(),
                )
                + AddObservedMask(
                    fields=("target",),
                    optional_fields=(),
                    observed_mask_field="observed_mask",
                    collection_type=dict,
                )
                + ImputeTimeSeries(
                    fields=("target",),
                    optional_fields=(),
                    imputation_method=DummyValueImputation(value=0.0),
                )
                + Patchify(
                    max_patch_size=self.module.patch_size,
                    fields=("target", "observed_mask"),
                    optional_fields=(),
                )
                + AddVariateIndex(
                    fields=("target",),
                    optional_fields=(),
                    variate_id_field="variate_id",
                    expected_ndim=3,
                    max_dim=self.hparams.max_dim,
                    randomize=True,
                    collection_type=dict,
                )
                + AddTimeIndex(
                    fields=("target",),
                    optional_fields=(),
                    time_id_field="time_id",
                    expected_ndim=3,
                    collection_type=dict,
                )
                + EvalNextTokenPrediction(
                    mask_length=math.ceil(prediction_length / patch_size),
                    target_field="target",
                    prediction_mask_field="prediction_mask",
                    expected_ndim=3,
                )
                + ExtendMask(
                    fields=tuple(),
                    optional_fields=(),
                    mask_field="prediction_mask",
                    expected_ndim=3,
                )
                + FlatPackCollection(
                    field="variate_id",
                    feat=False,
                )
                + FlatPackCollection(
                    field="time_id",
                    feat=False,
                )
                + FlatPackCollection(
                    field="prediction_mask",
                    feat=False,
                )
                + FlatPackCollection(
                    field="observed_mask",
                    feat=True,
                )
                + FlatPackFields(
                    output_field="target",
                    fields=("target",),
                    optional_fields=(),
                    feat=True,
                )
                + FlatPackCollection(
                    field="label_observed_mask",
                    feat=True,
                )
                + FlatPackFields(
                    output_field="label",
                    fields=("label",),
                    optional_fields=(),
                    feat=True,
                )
                + SequencifyField(field="patch_size", target_field="target")
                + SelectFields(fields=list(self.seq_fields))
            )

        return defaultdict(lambda: default_val_transform)