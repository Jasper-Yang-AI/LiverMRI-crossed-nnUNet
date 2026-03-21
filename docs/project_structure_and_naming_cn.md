# 工程结构与命名规则

本项目的标准工程名为 `livermri_crossseq`。

## 1. 顶层目录规则

- `configs/dataset`
  数据集级配置，只放数据来源、序列别名、ROI 参数、实验定义入口。
- `configs/experiment`
  实验级配置，只放跨序列融合、anchor、registration、训练策略。
- `configs/shared`
  共享静态表，例如 `label_map.yaml`。
- `docs`
  文档与规范，不放运行产物。
- `scripts/common`
  脚本公共工具。
- `scripts/dataset`
  manifest、fold、导出、dataset 组织。
- `scripts/preprocess`
  ROI 生成与预处理。
- `scripts/experiment`
  单序列筛选任务生成、anchor 规划、registration manifest。
- `scripts/eval`
  预测后处理、评估、汇总。
- `scripts/qc`
  自检和质量控制。
- `scripts/workflow`
  高层工作流入口。
- `src/livermri_crossseq`
  新工程 Python 包，放项目业务逻辑。
- `vendor/nnUNet`
  上游源码，只做 vendoring，不把项目逻辑塞进去。

## 2. 文件命名规则

- Python 文件统一使用 `snake_case.py`
- PowerShell 工作流统一使用 `run_livermri_<workflow>.ps1`
- 数据集配置统一使用 `livermri_crossseq_<scope>.yaml`
- 实验配置统一使用 `livermri_crossseq_<experiment>.yaml`
- 中文文档统一使用 `<topic>_cn.md`

## 3. 语义命名规则

- 用 `dataset` 代替笼统的 `data`
- 用 `preprocess` 代替阶段性很强的旧名字
- 用 `experiment` 代替 `experiments`
- 用 `eval` 代替 `evaluation`
- 用 `qc` 代替 `quality`
- 用 `workflow` 表示高层入口，不再单独用含糊的 `crossseq` 目录名
- 优先使用“动作 + 对象 + 场景”的脚本名
  例如 `generate_sequence_screening_jobs.py`、`attach_liver_roi_to_manifest.py`

## 4. 运行产物命名规则

- 所有运行时文件只放在 `outputs/` 或 `workspaces/`
- 单序列筛选产物统一放在 `outputs/sequence_screening/`
- 单序列筛选任务脚本统一放在 `outputs/sequence_screening/jobs/`
- 跨序列工作空间统一放在 `workspaces/livermri_crossseq/`
- 带 ROI 的 manifest 统一命名为 `manifest_with_folds_liver_roi.csv`
- `nnUNet_raw*`、`nnUNet_preprocessed*`、`nnUNet_results*` 都视为运行目录，不属于源码结构

## 5. 当前标准文件

- 数据集配置：
  [livermri_crossseq_dataset.yaml](D:/livermri_crossseq_nnunetv2/LiverMRI-CrossSeq-nnUNetv2/configs/dataset/livermri_crossseq_dataset.yaml)
- 融合实验配置：
  [livermri_crossseq_fusion.yaml](D:/livermri_crossseq_nnunetv2/LiverMRI-CrossSeq-nnUNetv2/configs/experiment/livermri_crossseq_fusion.yaml)
- 单序列筛选入口：
  [run_livermri_sequence_screening.ps1](D:/livermri_crossseq_nnunetv2/LiverMRI-CrossSeq-nnUNetv2/run_livermri_sequence_screening.ps1)
- 跨序列主流程入口：
  [run_livermri_crossseq_pipeline.ps1](D:/livermri_crossseq_nnunetv2/LiverMRI-CrossSeq-nnUNetv2/run_livermri_crossseq_pipeline.ps1)
