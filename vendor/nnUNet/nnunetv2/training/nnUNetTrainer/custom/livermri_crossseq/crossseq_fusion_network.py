from __future__ import annotations

from typing import Any

import torch
from torch import nn


class BranchStem3D(nn.Module):
    def __init__(self, in_channels: int, stem_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, stem_channels, kernel_size=3, padding=1, bias=True),
            nn.InstanceNorm3d(stem_channels, affine=True),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(stem_channels, stem_channels, kernel_size=3, padding=1, bias=True),
            nn.InstanceNorm3d(stem_channels, affine=True),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class CrossSeqFusionUNet(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        branch_specs: list[dict[str, Any]],
        stem_channels: int,
        num_output_channels: int,
        liver_mask_channel: int | None = None,
    ):
        super().__init__()
        if not branch_specs:
            raise ValueError("CrossSeqFusionUNet requires at least one branch specification.")

        self.backbone = backbone
        self.decoder = getattr(backbone, "decoder", None)
        self.branch_specs = branch_specs
        self.liver_mask_channel = liver_mask_channel
        self.branch_names = [spec["sequence_name"] for spec in branch_specs]

        self.stems = nn.ModuleList()
        self.gates = nn.ModuleList()
        self.aux_heads = nn.ModuleList()
        for spec in branch_specs:
            in_channels = len(spec.get("image_channels", []))
            if in_channels <= 0:
                raise ValueError(f"Invalid image channel mapping for branch {spec}")
            self.stems.append(BranchStem3D(in_channels, stem_channels))
            self.gates.append(
                nn.Sequential(
                    nn.Conv3d(stem_channels + 1, stem_channels, kernel_size=1, bias=True),
                    nn.LeakyReLU(inplace=True),
                    nn.Conv3d(stem_channels, 1, kernel_size=1, bias=True),
                    nn.Sigmoid(),
                )
            )
            self.aux_heads.append(nn.Conv3d(stem_channels, num_output_channels, kernel_size=1, bias=True))

    def _presence_from_image(self, image_tensor: torch.Tensor) -> torch.Tensor:
        shape = [image_tensor.shape[0], 1] + [1] * (image_tensor.ndim - 2)
        present = (image_tensor.abs().flatten(2).sum(-1) > 0).float()
        return present.view(*shape)

    def _split_branch_inputs(self, x: torch.Tensor, branch_spec: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        image_tensor = x[:, branch_spec["image_channels"], ...]
        conf_channels = branch_spec.get("confidence_channels", [])
        if conf_channels:
            confidence_tensor = x[:, conf_channels, ...].mean(1, keepdim=True).clamp_(0.0, 1.0)
        else:
            confidence_tensor = torch.ones(
                (x.shape[0], 1, *x.shape[2:]),
                dtype=x.dtype,
                device=x.device,
            )
        return image_tensor, confidence_tensor

    def compute_fusion(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor]:
        branch_features: list[torch.Tensor] = []
        aux_logits: list[torch.Tensor] = []
        gate_maps: list[torch.Tensor] = []

        for idx, spec in enumerate(self.branch_specs):
            image_tensor, confidence_tensor = self._split_branch_inputs(x, spec)
            branch_feature = self.stems[idx](image_tensor)
            aux_logits.append(self.aux_heads[idx](branch_feature))
            presence = self._presence_from_image(image_tensor)
            gate_map = self.gates[idx](torch.cat([branch_feature, confidence_tensor], dim=1)) * presence
            branch_features.append(branch_feature)
            gate_maps.append(gate_map)

        stacked_gates = torch.stack(gate_maps, dim=1)
        stacked_gates = stacked_gates / (stacked_gates.sum(dim=1, keepdim=True) + 1e-6)
        fused = torch.zeros_like(branch_features[0])
        for idx, feature in enumerate(branch_features):
            fused = fused + stacked_gates[:, idx] * feature
        return fused, aux_logits, stacked_gates

    def forward(self, x: torch.Tensor):
        fused, _, _ = self.compute_fusion(x)
        return self.backbone(fused)

    def forward_train(self, x: torch.Tensor) -> dict[str, Any]:
        fused, aux_logits, fusion_weights = self.compute_fusion(x)
        return {
            "main": self.backbone(fused),
            "aux": aux_logits,
            "fusion_weights": fusion_weights,
        }
