from __future__ import annotations

from typing import Any, List, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import autocast

from nnunetv2.training.loss.compound_losses import DC_and_BCE_loss, DC_and_CE_loss
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss, get_tp_fp_fn_tn
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import (
    OptimizedModule,
    collate_outputs,
    dist,
    dummy_context,
    nnUNetTrainer,
)
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans

from .crossseq_fusion_network import CrossSeqFusionUNet


class nnUNetTrainerCrossSeqFusion(nnUNetTrainer):
    _crossseq_dataset_config: dict[str, Any] = {}

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, device: torch.device = torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device=device)
        self.crossseq_config = dataset_json.get("livermri_crossseq", {})
        if not self.crossseq_config:
            raise RuntimeError(
                "dataset.json is missing the 'livermri_crossseq' section. "
                "Use the registered multibranch exporter before training this trainer."
            )

        type(self)._crossseq_dataset_config = self.crossseq_config
        self.branch_specs = list(self.crossseq_config.get("channel_layout", []))
        self.liver_mask_channel = self.crossseq_config.get("liver_mask_channel")
        self.dropout_config = dict(self.crossseq_config.get("dropout", {}))
        self.loss_config = dict(self.crossseq_config.get("loss", {}))
        self.drop_anchor_sequence = bool(self.crossseq_config.get("drop_anchor_sequence", False))
        self.image_channel_indices = sorted(
            {
                int(channel)
                for spec in self.branch_specs
                for channel in spec.get("image_channels", [])
            }
        )
        self.non_image_channel_indices = sorted(
            {
                int(channel)
                for spec in self.branch_specs
                for channel in spec.get("confidence_channels", [])
            }
        )
        if self.liver_mask_channel is not None:
            self.non_image_channel_indices.append(int(self.liver_mask_channel))
        self.aux_weight = float(self.loss_config.get("aux_weight", 0.3))
        self.consistency_weight = float(self.loss_config.get("consistency_weight", 0.1))
        self.outside_liver_weight = float(self.loss_config.get("outside_liver_weight", 0.15))
        self.aux_loss = self._build_single_output_loss()

    @staticmethod
    def build_network_architecture(
        architecture_class_name: str,
        arch_init_kwargs: dict,
        arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool = True,
    ) -> torch.nn.Module:
        crossseq_cfg = dict(nnUNetTrainerCrossSeqFusion._crossseq_dataset_config or {})
        branch_specs = list(crossseq_cfg.get("channel_layout", []))
        stem_channels = int(crossseq_cfg.get("branch_stem_channels", 16))
        liver_mask_channel = crossseq_cfg.get("liver_mask_channel")
        backbone = get_network_from_plans(
            architecture_class_name,
            arch_init_kwargs,
            arch_init_kwargs_req_import,
            input_channels=stem_channels,
            output_channels=num_output_channels,
            allow_init=True,
            deep_supervision=enable_deep_supervision,
        )
        return CrossSeqFusionUNet(
            backbone=backbone,
            branch_specs=branch_specs,
            stem_channels=stem_channels,
            num_output_channels=num_output_channels,
            liver_mask_channel=liver_mask_channel,
        )

    def _build_single_output_loss(self):
        if self.label_manager.has_regions:
            return DC_and_BCE_loss(
                {},
                {"batch_dice": self.configuration_manager.batch_dice, "do_bg": True, "smooth": 1e-5, "ddp": self.is_ddp},
                use_ignore_label=self.label_manager.ignore_label is not None,
                dice_class=MemoryEfficientSoftDiceLoss,
            )
        return DC_and_CE_loss(
            {"batch_dice": self.configuration_manager.batch_dice, "smooth": 1e-5, "do_bg": False, "ddp": self.is_ddp},
            {},
            weight_ce=1,
            weight_dice=1,
            ignore_label=self.label_manager.ignore_label,
            dice_class=MemoryEfficientSoftDiceLoss,
        )

    def _get_train_network(self):
        mod = self.network.module if self.is_ddp else self.network
        if isinstance(mod, OptimizedModule):
            mod = mod._orig_mod
        return mod

    def _get_main_logits(self, output):
        if self.enable_deep_supervision and isinstance(output, (tuple, list)):
            return output[0]
        return output

    def _target_for_aux(self, target):
        if isinstance(target, list):
            return target[0]
        return target

    def _drop_probability(self, strong: bool) -> float:
        warmup_fraction = float(self.dropout_config.get("warmup_fraction", 0.25))
        warmup_epochs = max(1, int(round(self.num_epochs * warmup_fraction)))
        if self.current_epoch < warmup_epochs and not strong:
            return float(self.dropout_config.get("warmup_drop_probability", 0.05))
        return float(self.dropout_config.get("main_drop_probability", 0.3))

    def _apply_sequence_dropout(self, data: torch.Tensor, strong: bool = False) -> torch.Tensor:
        probability = self._drop_probability(strong=strong)
        max_dropped = int(self.dropout_config.get("max_dropped_sequences", 2))
        if probability <= 0 or max_dropped <= 0:
            return data

        dropped = data.clone()
        for batch_index in range(dropped.shape[0]):
            candidate_indices = []
            for spec_index, spec in enumerate(self.branch_specs):
                if spec.get("is_anchor", False) and not self.drop_anchor_sequence:
                    continue
                image_channels = spec.get("image_channels", [])
                if not image_channels:
                    continue
                channel_view = dropped[batch_index, image_channels, ...]
                if channel_view.abs().sum().item() > 0:
                    candidate_indices.append(spec_index)
            if not candidate_indices:
                continue
            if np.random.uniform() > probability:
                continue
            n_drop = min(max_dropped, len(candidate_indices))
            n_drop = int(np.random.randint(1, n_drop + 1))
            dropped_indices = np.random.choice(candidate_indices, size=n_drop, replace=False)
            for spec_index in dropped_indices:
                spec = self.branch_specs[int(spec_index)]
                channels = list(spec.get("image_channels", [])) + list(spec.get("confidence_channels", []))
                dropped[batch_index, channels, ...] = 0
        return dropped

    def _sanitize_non_image_channels(self, data: torch.Tensor) -> torch.Tensor:
        if self.non_image_channel_indices:
            data[:, self.non_image_channel_indices, ...] = data[:, self.non_image_channel_indices, ...].clamp(0.0, 1.0)
        return data

    def _foreground_probability(self, logits: torch.Tensor) -> torch.Tensor:
        if self.label_manager.has_regions:
            return torch.sigmoid(logits)
        probabilities = torch.softmax(logits, dim=1)
        return probabilities[:, 1:, ...].sum(dim=1, keepdim=True)

    def _outside_liver_penalty(self, logits: torch.Tensor, liver_mask: torch.Tensor) -> torch.Tensor:
        foreground_probability = self._foreground_probability(logits)
        return (foreground_probability * (1.0 - liver_mask)).mean()

    def _consistency_loss(self, teacher_logits: torch.Tensor, student_logits: torch.Tensor) -> torch.Tensor:
        teacher_probability = self._foreground_probability(teacher_logits).detach()
        student_probability = self._foreground_probability(student_logits)
        return F.mse_loss(student_probability, teacher_probability)

    def train_step(self, batch: dict) -> dict:
        data = batch["data"].to(self.device, non_blocking=True)
        target = batch["target"]
        if isinstance(target, list):
            target = [item.to(self.device, non_blocking=True) for item in target]
        else:
            target = target.to(self.device, non_blocking=True)

        data = self._sanitize_non_image_channels(data)
        main_data = self._apply_sequence_dropout(data, strong=False)
        aux_target = self._target_for_aux(target)
        warmup_fraction = float(self.dropout_config.get("warmup_fraction", 0.25))
        warmup_epochs = max(1, int(round(self.num_epochs * warmup_fraction)))

        self.optimizer.zero_grad(set_to_none=True)
        with autocast(self.device.type, enabled=True) if self.device.type == "cuda" else dummy_context():
            train_network = self._get_train_network()
            if hasattr(train_network, "forward_train"):
                outputs = train_network.forward_train(main_data)
            else:
                outputs = {"main": self.network(main_data), "aux": []}

            main_output = outputs["main"]
            total_loss = self.loss(main_output, target)
            main_loss = total_loss

            aux_loss_value = torch.zeros((), device=self.device)
            aux_outputs = outputs.get("aux", [])
            if aux_outputs:
                aux_losses = [self.aux_loss(aux_output, aux_target) for aux_output in aux_outputs]
                aux_loss_value = torch.stack(aux_losses).mean()
                total_loss = total_loss + self.aux_weight * aux_loss_value

            outside_liver_loss = torch.zeros((), device=self.device)
            if self.liver_mask_channel is not None:
                liver_mask = main_data[:, int(self.liver_mask_channel): int(self.liver_mask_channel) + 1, ...].clamp(0.0, 1.0)
                outside_liver_loss = self._outside_liver_penalty(self._get_main_logits(main_output), liver_mask)
                total_loss = total_loss + self.outside_liver_weight * outside_liver_loss

            consistency_loss = torch.zeros((), device=self.device)
            if self.consistency_weight > 0 and self.current_epoch >= warmup_epochs:
                strong_data = self._apply_sequence_dropout(data, strong=True)
                strong_outputs = train_network.forward_train(strong_data) if hasattr(train_network, "forward_train") else {
                    "main": self.network(strong_data),
                    "aux": [],
                }
                consistency_loss = self._consistency_loss(
                    self._get_main_logits(main_output),
                    self._get_main_logits(strong_outputs["main"]),
                )
                total_loss = total_loss + self.consistency_weight * consistency_loss

        if self.grad_scaler is not None:
            self.grad_scaler.scale(total_loss).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()

        return {
            "loss": float(total_loss.detach().cpu()),
            "loss_main": float(main_loss.detach().cpu()),
            "loss_aux": float(aux_loss_value.detach().cpu()),
            "loss_consistency": float(consistency_loss.detach().cpu()),
            "loss_outside_liver": float(outside_liver_loss.detach().cpu()),
        }

    def on_train_epoch_end(self, train_outputs: List[dict]):
        outputs = collate_outputs(train_outputs)
        scalar_keys = ["loss", "loss_main", "loss_aux", "loss_consistency", "loss_outside_liver"]

        reduced: dict[str, float] = {}
        if self.is_ddp:
            for key in scalar_keys:
                gathered = [None for _ in range(dist.get_world_size())]
                dist.all_gather_object(gathered, outputs[key])
                reduced[key] = float(np.vstack(gathered).mean())
        else:
            for key in scalar_keys:
                reduced[key] = float(np.mean(outputs[key]))

        self.logger.log("train_losses", reduced["loss"], self.current_epoch)
        self.logger.log("train_main_losses", reduced["loss_main"], self.current_epoch)
        self.logger.log("train_aux_losses", reduced["loss_aux"], self.current_epoch)
        self.logger.log("train_consistency_losses", reduced["loss_consistency"], self.current_epoch)
        self.logger.log("train_outside_liver_losses", reduced["loss_outside_liver"], self.current_epoch)

    def validation_step(self, batch: dict) -> dict:
        data = batch["data"].to(self.device, non_blocking=True)
        target = batch["target"]
        if isinstance(target, list):
            target = [item.to(self.device, non_blocking=True) for item in target]
        else:
            target = target.to(self.device, non_blocking=True)

        data = self._sanitize_non_image_channels(data)
        with autocast(self.device.type, enabled=True) if self.device.type == "cuda" else dummy_context():
            output = self.network(data)
            l = self.loss(output, target)

        if self.enable_deep_supervision:
            output = output[0]
            target = target[0]

        axes = [0] + list(range(2, output.ndim))
        if self.label_manager.has_regions:
            predicted_segmentation_onehot = (torch.sigmoid(output) > 0.5).long()
        else:
            output_seg = output.argmax(1)[:, None]
            predicted_segmentation_onehot = torch.zeros(output.shape, device=output.device, dtype=torch.float16)
            predicted_segmentation_onehot.scatter_(1, output_seg, 1)
            del output_seg

        if self.label_manager.has_ignore_label:
            if not self.label_manager.has_regions:
                mask = (target != self.label_manager.ignore_label).float()
                target[target == self.label_manager.ignore_label] = 0
            else:
                mask = ~target[:, -1:] if target.dtype == torch.bool else 1 - target[:, -1:]
                target = target[:, :-1]
        else:
            mask = None

        tp, fp, fn, _ = get_tp_fp_fn_tn(predicted_segmentation_onehot, target, axes=axes, mask=mask)

        tp_hard = tp.detach().cpu().numpy()
        fp_hard = fp.detach().cpu().numpy()
        fn_hard = fn.detach().cpu().numpy()
        if not self.label_manager.has_regions:
            tp_hard = tp_hard[1:]
            fp_hard = fp_hard[1:]
            fn_hard = fn_hard[1:]

        return {"loss": l.detach().cpu().numpy(), "tp_hard": tp_hard, "fp_hard": fp_hard, "fn_hard": fn_hard}
