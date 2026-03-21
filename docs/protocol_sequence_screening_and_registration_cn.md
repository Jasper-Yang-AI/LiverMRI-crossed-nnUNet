# 单序列筛选与配准协议

这个协议对应新的 `livermri_crossseq` 工程命名。

## 阶段 1：单序列筛选

入口脚本：
- [run_livermri_sequence_screening.ps1](D:/livermri_crossseq_nnunetv2/LiverMRI-CrossSeq-nnUNetv2/run_livermri_sequence_screening.ps1)

核心配置：
- [livermri_crossseq_dataset.yaml](D:/livermri_crossseq_nnunetv2/LiverMRI-CrossSeq-nnUNetv2/configs/dataset/livermri_crossseq_dataset.yaml)

这一阶段完成：

1. 生成 manifest
2. 生成 patient-level folds
3. 生成 liver ROI
4. 导出 liver-ROI 约束的单序列训练数据
5. 生成单序列筛选任务脚本

## 阶段 2：Anchor 规划

推荐入口：

```powershell
python -m scripts.workflow.plan_anchor_pilot `
  --screening-summary .\outputs\sequence_screening\summary\sequence_screening_summary.csv `
  --manifest .\workspaces\livermri_crossseq\manifests\manifest_with_folds_liver_roi.csv `
  --study-config .\configs\dataset\livermri_crossseq_dataset.yaml `
  --crossseq-config .\configs\experiment\livermri_crossseq_fusion.yaml
```

## 阶段 3：配准任务生成

```powershell
python -m scripts.experiment.prepare_registration_manifest `
  --manifest .\outputs\sequence_screening\manifests\manifest_with_folds_liver_roi.csv `
  --study-config .\configs\dataset\livermri_crossseq_dataset.yaml `
  --anchor-json .\outputs\sequence_screening\summary\recommended_anchor.json `
  --out-dir .\outputs\sequence_screening\registration
```

输出 manifest 默认包含：

- `registered_image_path_expected`
- `registered_liver_mask_path_expected`
- `registered_label_path_expected`
- `registration_confidence_path_expected`
- `jacobian_determinant_path_expected`
- `registration_metrics_path_expected`

## 阶段 4：跨序列融合

入口脚本：
- [run_livermri_crossseq_pipeline.ps1](D:/livermri_crossseq_nnunetv2/LiverMRI-CrossSeq-nnUNetv2/run_livermri_crossseq_pipeline.ps1)

主实验配置：
- [livermri_crossseq_fusion.yaml](D:/livermri_crossseq_nnunetv2/LiverMRI-CrossSeq-nnUNetv2/configs/experiment/livermri_crossseq_fusion.yaml)
