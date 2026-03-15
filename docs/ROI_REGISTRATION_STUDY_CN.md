# ROI + Registration Oriented Study

## 目标

这套新设计把工程拆成四个连续阶段：

1. `whole-liver ROI` 预处理：对 `D:\Dataset003_v2_LiverTumorSeg` 的所有序列先跑 `TotalSegmentator`，得到统一的肝脏 ROI。
2. `ROI 约束单序列筛选`：每个序列单独训练 `nnUNetv2 3d_fullres`，训练前用膨胀 liver mask 约束输入，推理后再用膨胀 liver mask 裁掉肝外预测，比较每个序列的真实可用性。
3. `anchor sequence` 选择与配准准备：根据筛选结果选一个目标序列作为 fixed image，给后续配准实验建立 pairwise job manifest。
4. `配准后 ROI 裁剪训练`：等配准方法定下来并跑完后，再把所有序列统一到 anchor 空间，按 liver ROI 裁剪，训练最终模型。

## 为什么改成这条主线

- 先做 liver ROI，可以把序列间 FOV 差异、背景组织噪声、肝外假阳性先压下来。
- 先做单序列筛选，再决定 anchor，更符合后续配准的实际需求。
- 配准阶段暂时不锁死算法，工程上先把 manifest、目录规范和输入输出接口铺好，后面可以平滑接入 ANTs、Elastix 或你最终想用的方法。

## 阶段 1：TotalSegmentator liver ROI

### 环境

项目主环境继续负责 `nnUNetv2`、评估和脚本调度。

`TotalSegmentator` 单独放到新环境 `totalseg`：

```powershell
cd D:\livermri_crossseq_nnunetv2\LiverMRI-CrossSeq-nnUNetv2
& .\scripts\roi\create_totalseg_env.ps1
```

脚本默认使用：

- `conda env`: `totalseg`
- `task`: `total_mr`
- `roi_subset`: `liver`

ROI 输出路径：

- `D:\Dataset003_v2_LiverTumorSeg\liver_roi_totalseg\raw/<subset>/<case_id>.nii.gz`
- `D:\Dataset003_v2_LiverTumorSeg\liver_roi_totalseg\clean/<subset>/<case_id>.nii.gz`
- `D:\Dataset003_v2_LiverTumorSeg\liver_roi_totalseg\dilated_12mm/<subset>/<case_id>.nii.gz`

其中：

- `raw`: TotalSegmentator 原始肝脏分割
- `clean`: 最大连通域 + 填洞后的 mask
- `dilated_12mm`: 后续训练约束和推理后处理使用的安全边界 mask

## 阶段 2：ROI 约束单序列筛选

### 当前落地策略

每个序列都做一个独立实验：

- `RS01`: `T2_MAIN`
- `RS02`: `DWI`
- `RS03`: `ADC`
- `RS04`: `T1`
- `RS05`: `InPhase`
- `RS06`: `OutPhase`
- `RS07`: `C-pre`
- `RS08`: `ARTERIAL`
- `RS09`: `PORTAL`
- `RS10`: `DELAY`

### 约束方式

- 训练输入：保留原始空间，不裁剪，只把膨胀 liver mask 外的体素置零。
- 训练标签：默认和同一个 liver mask 相交，避免肝外噪声标签进入训练。
- 推理后处理：预测结果与膨胀 liver mask 做交集，直接去掉肝外假阳性。

这样做的好处是：

- 第一阶段不引入配准误差。
- 各序列都仍在自己的原始空间里比较，结果更真实。
- 还保留了原体积上下文，避免过早 ROI crop 带来的边界截断风险。

### 一键准备

```powershell
& .\run_roi_screening_study.ps1
```

这一步会完成：

1. 生成 manifest
2. patient-level 5-fold 划分
3. 跑 whole-liver ROI
4. 把 ROI 路径挂回 manifest
5. 生成所有筛选实验的运行脚本

### 单个实验运行

例如只先跑 `PORTAL`：

```powershell
& .\outputs\roi_study\screening\experiments\RS09\suite_commands\run_RS09.ps1
```

全部跑完：

```powershell
& .\outputs\roi_study\screening\experiments\run_all_screening.ps1
```

### 序列筛选汇总

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

推荐 anchor 默认按下面顺序排序：

1. `external_dice_mean` 高
2. `internal_dice_mean` 高
3. `external_hd95_mean` 低

## 阶段 3：配准实验准备

目前工程里先把“配准输入是什么、输出应该放哪儿”固定下来，但不强行绑定具体方法。

### 生成配准任务清单

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

每条 pairwise job 都包含：

- fixed image: 选中的 anchor sequence
- moving image: 同一病人的其他序列
- fixed / moving tumor label
- fixed / moving liver ROI
- 预期输出目录
- 预期变换文件路径

### 建议的配准设计

后面真正做配准时，建议按这个顺序试：

1. liver ROI 内 rigid
2. liver ROI 内 affine
3. liver ROI 内 deformable

建议所有 moving 序列都只配准到同一个 anchor，避免链式误差传递。

## 阶段 4：配准后 ROI crop 训练

这一阶段的代码接口还没锁死，是因为你已经明确说“配准实验方法和设计后面再思考搭建”。所以我这里先定工程原则，不提前把错误假设写死：

- 所有 moving 序列先到 anchor 空间
- ROI crop 只基于 anchor 空间的 liver mask
- crop margin 建议从 `20 mm` 起
- 最终模型训练时，labels 以 anchor 空间标签为准
- 推理后依然保留 `dilated liver mask` 的肝外抑制

等你后面决定具体配准方法后，下一步最自然的扩展是：

1. 补一个 `registration runner`
2. 补一个 `registered multichannel exporter`
3. 补一个 `post-registration experiment generator`

## 当前已经新增的关键脚本

- `scripts/roi/create_totalseg_env.ps1`
- `scripts/roi/run_totalsegmentator_liver_roi.py`
- `scripts/data/augment_manifest_with_roi.py`
- `scripts/data/export_nnunet_source_dataset_roi.py`
- `scripts/data/export_target_test_sets_roi.py`
- `scripts/evaluation/postprocess_predictions_by_liver_roi.py`
- `scripts/experiments/summarize_sequence_screening.py`
- `scripts/experiments/prepare_registration_manifest.py`
- `scripts/experiments/generate_roi_experiment_scripts.py`
- `run_roi_screening_study.ps1`

## 现阶段最推荐的实际执行顺序

1. 创建 `totalseg` 环境
2. 运行 `run_roi_screening_study.ps1`
3. 先跑增强期序列和 `T2_MAIN` 的筛选实验
4. 汇总 ranking，确定 anchor
5. 生成配准 manifest
6. 再讨论并落地具体配准算法与 post-registration 多通道训练
