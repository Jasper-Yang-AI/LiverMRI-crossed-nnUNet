# LiverMRI-CrossSeq-nnUNetv2

面向 liver MRI 多序列稳健性研究的 nnUNetv2 工程。

这版工程已经按你的实验设计重构为“组别驱动 + 实验编号驱动”：

- 组别：`P1 / P2 / P3 / P456 / E_all`
- 实验：`A / M1 / M2 / M3 / U1 / U2 / U3 / U4`

主配置见 [study_config.yaml](/d:/livermri_crossseq_nnunetv2/LiverMRI-CrossSeq-nnUNetv2/configs/study_config.yaml)。

## 分组定义

- `P1 = T2 + T2WI`
- `P2 = DWI`
- `P3 = ADC`
- `P456 = T1 + InPhase + OutPhase + C-pre`
- `E_all = ARTERIAL + PORTAL + DELAY`

说明：

- `T2` 和 `T2WI` 被视为同一主平扫序列家族
- `P456` 是其余 plain MRI 的合并组
- 如果你要改分组，只改 [study_config.yaml](/d:/livermri_crossseq_nnunetv2/LiverMRI-CrossSeq-nnUNetv2/configs/study_config.yaml) 即可

## 实验矩阵

- `A`: 训练 `P1`，测试 `P1 / P2 / P3 / P456 / E_all`
- `M1`: 训练 `P1 + P2 + P3`
- `M2`: 训练 `P1 + P2 + P3 + P456`
  这就是 `All-Plain`
- `M3`: 训练 `M2 + E_all`
  这就是 `All-Seq`
- `U1`: 训练 `P2-only`
- `U2`: 训练 `P3-only`
- `U3`: 训练 `P456-only`
- `U4`: 训练 `E_all-only`

默认所有实验都测试：

- `P1`
- `P2`
- `P3`
- `P456`
- `E_all`

## 当前数据上的重要事实

基于你现在的数据统计：

- `P1` 在 `train/test` 都有
- `P2` 在 `train/test` 都有
- `P3=ADC` 只在 `train` 有，外部 `test` 没有
- `P456` 和 `E_all` 在 `train/test` 都有

所以：

- `P3` 只能做内部 CV，不能做外部测试
- 外部 `P1` 实际主要来自 `T2WI`

这些是数据决定的，不是脚本缺了什么。

## 环境准备

你不需要 clone `nnUNet` 源码。  
本机 `conda` 环境 `nnu` 已经装好了 `nnunetv2`。

进入环境：

```powershell
conda activate nnu
cd D:\livermri_crossseq_nnunetv2\LiverMRI-CrossSeq-nnUNetv2
pip install -r requirements.txt
```

设置 nnUNet 路径：

```powershell
$env:nnUNet_raw="D:\nnUNet_raw"
$env:nnUNet_preprocessed="D:\nnUNet_preprocessed"
$env:nnUNet_results="D:\nnUNet_results"

New-Item -ItemType Directory -Force -Path $env:nnUNet_raw | Out-Null
New-Item -ItemType Directory -Force -Path $env:nnUNet_preprocessed | Out-Null
New-Item -ItemType Directory -Force -Path $env:nnUNet_results | Out-Null
```

## 公共准备步骤

这些步骤所有实验只做一次。

```powershell
python scripts/propose_manifest.py `
  --root "D:\Dataset003_v2_LiverTumorSeg" `
  --out outputs/manifest_proposed.csv

python scripts/validate_manifest.py `
  --manifest outputs/manifest_proposed.csv `
  --study-config configs/study_config.yaml `
  --out-dir outputs/audit `
  --skip-label-values

python scripts/profile_dataset.py `
  --manifest outputs/manifest_proposed.csv `
  --study-config configs/study_config.yaml `
  --out-dir outputs/profile

python scripts/assign_folds.py `
  --manifest outputs/manifest_proposed.csv `
  --n-folds 5 `
  --seed 3407 `
  --out outputs/manifest_with_folds.csv

python scripts/self_audit.py `
  --manifest outputs/manifest_with_folds.csv `
  --study-config configs/study_config.yaml `
  --out-dir outputs/audit
```

关键输出：

- `outputs/manifest_with_folds.csv`
- `outputs/profile/DATASET_PROFILE.md`
- `outputs/profile/experiment_catalog.csv`

## 推荐入口

先运行顶层入口：

```powershell
.\run_full_study.ps1
```

这个脚本会：

1. 跑公共准备步骤
2. 自动生成每个实验的独立脚本

生成位置：

- `outputs/experiments/A/suite_commands/run_A.ps1`
- `outputs/experiments/M1/suite_commands/run_M1.ps1`
- `outputs/experiments/M2/suite_commands/run_M2.ps1`
- `outputs/experiments/M3/suite_commands/run_M3.ps1`
- `outputs/experiments/U1/suite_commands/run_U1.ps1`
- `outputs/experiments/U2/suite_commands/run_U2.ps1`
- `outputs/experiments/U3/suite_commands/run_U3.ps1`
- `outputs/experiments/U4/suite_commands/run_U4.ps1`

## 全部实验与优先级

这个工程支持的是完整 8 个实验，不是 3 个实验。

### 全部实验

- `A`: `P1 -> P1 / P2 / P3 / P456 / E_all`
- `M1`: `P1 + P2 + P3 -> P1 / P2 / P3 / P456 / E_all`
- `M2`: `P1 + P2 + P3 + P456 -> P1 / P2 / P3 / P456 / E_all`
- `M3`: `All-Plain + E_all -> P1 / P2 / P3 / P456 / E_all`
- `U1`: `P2-only -> P1 / P2 / P3 / P456 / E_all`
- `U2`: `P3-only -> P1 / P2 / P3 / P456 / E_all`
- `U3`: `P456-only -> P1 / P2 / P3 / P456 / E_all`
- `U4`: `E_all-only -> P1 / P2 / P3 / P456 / E_all`

### 为什么前面只重点提了 3 个

前面那一节说的不是“只跑 3 个”，而是“建议优先跑的 3 个核心实验”：

- `A`
  作为 `P1-only` 基线
- `M2`
  作为 `All-Plain` 主实验
- `M3`
  作为 `All-Seq` 上界实验

这 3 个最容易先形成论文主线。

### 完整推荐顺序

如果你要按完整设计跑，建议顺序是：

1. `A`
2. `M1`
3. `M2`
4. `M3`
5. `U1`
6. `U2`
7. `U3`
8. `U4`

其中：

- `A / M1 / M2 / M3` 是主线实验
- `U1 / U2 / U3 / U4` 更像补充对照和 ablation

## 手动单独跑某个实验

以 `A` 为例：

```powershell
python scripts/export_nnunet_source_dataset.py `
  --manifest outputs/manifest_with_folds.csv `
  --study-config configs/study_config.yaml `
  --experiment-id A `
  --nnunet-raw "D:\nnUNet_raw"

python scripts/generate_splits_json.py `
  --manifest outputs/manifest_with_folds.csv `
  --study-config configs/study_config.yaml `
  --experiment-id A `
  --nnunet-preprocessed "D:\nnUNet_preprocessed"

nnUNetv2_plan_and_preprocess -d 311 --verify_dataset_integrity
nnUNetv2_train 311 3d_fullres 0
nnUNetv2_train 311 3d_fullres 1
nnUNetv2_train 311 3d_fullres 2
nnUNetv2_train 311 3d_fullres 3
nnUNetv2_train 311 3d_fullres 4

python scripts/export_target_test_sets.py `
  --manifest outputs/manifest_with_folds.csv `
  --study-config configs/study_config.yaml `
  --experiment-id A `
  --out-dir outputs/experiments/A/targets

.\outputs\experiments\A\commands\infer_internal_cv.ps1
.\outputs\experiments\A\commands\infer_external_test.ps1

python scripts/evaluate_predictions.py `
  --evaluation-manifest outputs/experiments/A/evaluation_manifest.csv `
  --out-csv outputs/experiments/A/results/per_case_metrics.csv

python scripts/aggregate_results.py `
  --metrics outputs/experiments/A/results/per_case_metrics.csv `
  --audit outputs/audit/self_audit_summary.json `
  --out-dir outputs/experiments/A/paper_assets
```

`M2` 也是同样用法，只要把 `A` 换成 `M2`，`dataset_id` 会自动从配置里读取。

## 结果看哪里

以 `A` 为例：

- `outputs/experiments/A/evaluation_manifest.csv`
- `outputs/experiments/A/results/per_case_metrics.csv`
- `outputs/experiments/A/paper_assets/Table2_internal_source_cv.csv`
- `outputs/experiments/A/paper_assets/Table3_internal_cross_sequence.csv`
- `outputs/experiments/A/paper_assets/Table4_external_test.csv`
- `outputs/experiments/A/paper_assets/Table_bundle_raw_breakdown.csv`

其中：

- `Table2`：同域结果
- `Table3`：内部跨组结果
- `Table4`：外部测试结果
- `Table_bundle_raw_breakdown`：组内原始序列拆解

## 最重要的原则

内部评估不能直接用 `-f 0 1 2 3 4` ensemble。

原因是你的数据是同病人多序列结构，如果内部评估用 fold ensemble，会发生 patient leakage，结果虚高。

当前工程已经固定为：

- 内部评估：fold 对应 fold 模型
- 外部评估：完全未见 `test` 才允许 `0 1 2 3 4` ensemble
