# LiverMRI-CrossSeq-nnUNetv2

用于 liver MRI 肿瘤分割实验的 `nnUNetv2` 工程。

当前工程已经切换到新的主线设计：

1. `TotalSegmentator` 做 whole-liver ROI
2. 在 liver ROI 约束下完成所有单序列 `nnUNetv2 3d_fullres` 筛选
3. 比较各序列指标，选择后续配准的 anchor sequence
4. 为配准实验生成 patient-wise job manifest
5. 等配准方法确定后，再做配准后 ROI crop 训练与推理后肝外抑制

默认数据集：

```text
D:\Dataset003_v2_LiverTumorSeg
```

## 1. 当前推荐主线

当前最推荐使用的是 ROI + registration oriented workflow：

- 入口脚本：`run_roi_screening_study.ps1`
- 配置文件：`configs/roi_study_config.yaml`
- 详细中文说明：`docs/ROI_REGISTRATION_STUDY_CN.md`

如果你现在是要做新的 whole-liver ROI、单序列筛选和后续配准实验，就直接从这条主线开始。

## 2. ROI 流程总览

### 2.1 TotalSegmentator 环境

`TotalSegmentator` 单独放在新环境 `totalseg`：

```powershell
cd D:\livermri_crossseq_nnunetv2\LiverMRI-CrossSeq-nnUNetv2
& .\scripts\roi\create_totalseg_env.ps1
```

### 2.2 Liver ROI 输出位置

生成的 liver mask 默认写回原始数据集根目录下，而不是工程 `outputs`：

```text
D:\Dataset003_v2_LiverTumorSeg\liver_roi_totalseg\raw
D:\Dataset003_v2_LiverTumorSeg\liver_roi_totalseg\clean
D:\Dataset003_v2_LiverTumorSeg\liver_roi_totalseg\dilated_12mm
```

这样每个 ROI mask 都和原始 case 一一对应，同时又不污染 `imagesTr / labelsTr / imagesTs / labelsTs`。

### 2.3 一键准备

```powershell
& .\run_roi_screening_study.ps1
```

这一步会自动完成：

1. 为 `D:\Dataset003_v2_LiverTumorSeg` 生成 manifest
2. 做 patient-level 5-fold 划分
3. 调用 `TotalSegmentator` 生成 whole-liver ROI
4. 把 ROI 路径挂回 manifest
5. 生成所有单序列筛选实验的运行脚本

## 3. 单序列 ROI 筛选实验

当前筛选实验为：

| 实验 | 序列 |
| --- | --- |
| `RS01` | `T2_MAIN` |
| `RS02` | `DWI` |
| `RS03` | `ADC` |
| `RS04` | `T1` |
| `RS05` | `InPhase` |
| `RS06` | `OutPhase` |
| `RS07` | `C-pre` |
| `RS08` | `ARTERIAL` |
| `RS09` | `PORTAL` |
| `RS10` | `DELAY` |

当前策略：

- 训练输入保留原始空间，但将膨胀 liver mask 外体素置零
- 训练标签默认裁到 liver ROI 内
- 推理后再用膨胀 liver mask 去掉肝外预测

### 3.1 跑单个实验

例如先跑 `PORTAL`：

```powershell
& .\outputs\roi_study\screening\experiments\RS09\suite_commands\run_RS09.ps1
```

### 3.2 跑完整筛选

```powershell
& .\outputs\roi_study\screening\experiments\run_all_screening.ps1
```

### 3.3 汇总筛选结果并选 anchor

```powershell
python -m scripts.experiments.summarize_sequence_screening `
  --study-config .\configs\roi_study_config.yaml `
  --experiments-root .\outputs\roi_study\screening\experiments `
  --out-dir .\outputs\roi_study\screening\summary `
  --require-external-for-anchor
```

关键输出：

- `outputs/roi_study/screening/summary/sequence_screening_summary.csv`
- `outputs/roi_study/screening/summary/recommended_anchor.json`

## 4. 配准准备

当前工程已经把配准前的 patient-wise 配对关系准备好了，但还没有锁定具体配准算法。

在筛选出 anchor sequence 后运行：

```powershell
python -m scripts.experiments.prepare_registration_manifest `
  --manifest .\outputs\roi_study\manifests\manifest_with_folds_roi.csv `
  --study-config .\configs\roi_study_config.yaml `
  --anchor-json .\outputs\roi_study\screening\summary\recommended_anchor.json `
  --out-dir .\outputs\roi_study\registration
```

关键输出：

- `outputs/roi_study/registration/pairwise_registration_manifest.csv`
- `outputs/roi_study/registration/registration_patient_summary.csv`

## 5. 脚本目录

现在 `scripts` 已按功能整理为：

- `scripts/lib`: 公共工具与 ROI 工具
- `scripts/data`: manifest、fold、ROI 感知数据导出
- `scripts/roi`: TotalSegmentator 与 ROI 生成
- `scripts/experiments`: 实验脚本生成、筛选汇总、配准清单
- `scripts/evaluation`: 评估与推理后处理
- `scripts/quality`: 质控与自检

## 6. 关键脚本

- `scripts/roi/create_totalseg_env.ps1`
- `scripts/roi/run_totalsegmentator_liver_roi.py`
- `scripts/data/augment_manifest_with_roi.py`
- `scripts/data/export_nnunet_source_dataset_roi.py`
- `scripts/data/export_target_test_sets_roi.py`
- `scripts/evaluation/postprocess_predictions_by_liver_roi.py`
- `scripts/experiments/summarize_sequence_screening.py`
- `scripts/experiments/prepare_registration_manifest.py`
- `scripts/experiments/generate_roi_experiment_scripts.py`

## 7. 推荐执行顺序

1. 创建 `totalseg` 环境
2. 运行 `run_roi_screening_study.ps1`
3. 先跑 `T2_MAIN` 和增强期序列筛选
4. 汇总 ranking，确定 anchor
5. 生成配准 manifest
6. 后续再补具体配准算法与配准后多通道训练
