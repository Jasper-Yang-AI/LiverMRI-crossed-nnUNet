# livermri_crossseq

`livermri_crossseq` 是一个围绕肝脏 MRI 肿瘤分割实验重构的新工程。  
它解决的是一个很具体的问题:

- 你的数据只有 `tumor/cancer` 标签，没有完整的器官标签
- 直接训练分割模型时，模型容易在肝外产生假阳性
- 不同 MRI 序列之间既有互补信息，又存在配准误差和缺失序列问题

这个工程对应的整体实验逻辑是:

1. 先用 `TotalSegmentator` 得到肝脏 ROI
2. 用 ROI 约束做单序列筛选，找出最优 anchor 候选
3. 做跨序列配准，保留原始序列强度
4. 导出配准后的多分支数据集
5. 用自定义 `nnUNetTrainerCrossSeqFusion` 训练最终的多序列融合模型

这份 README 的目标不是只介绍目录，而是让一个不了解项目的人，按文档就能明白:

- 工程每一步在做什么
- 应该在终端里输入什么命令
- 每一步会产出什么文件
- 这些设计背后的策略是什么
- 当前工程已经自动化到哪一步，哪一步还需要你自己补

## 1. 先看全流程

这个工程分成两条线:

- `单序列筛选线`
  目标是先回答“哪条序列最适合作为 anchor 或主序列”
- `跨序列融合线`
  目标是把其他序列对齐到 anchor 空间后，导出多分支数据并训练最终模型

推荐按下面顺序走:

1. 安装环境
2. 检查数据目录
3. 跑单序列筛选
4. 汇总结果并选择 anchor
5. 生成配准任务清单
6. 执行配准
7. 导出最终多分支数据集
8. 规划、预处理并训练最终融合模型

## 2. 当前工程已经自动化到哪一步

已经内置并可直接运行的部分:

- manifest 生成
- patient-level folds 划分
- liver ROI 生成
- ROI 回写到 manifest
- 单序列筛选任务脚本生成
- 单序列结果汇总和 anchor 初筛
- 跨序列 workspace 初始化
- anchor shortlist pilot 规划
- 注册后多分支数据集导出
- 自定义 nnUNet 融合 trainer 与网络骨架

当前还没有内置“真正执行配准”的统一 runner:

- 工程会生成 `pairwise_registration_manifest.csv`
- 但你需要用自己的 `ANTs / Elastix / 其他配准脚本` 去消费这个 manifest
- 只要你把配准输出按约定文件名写回每个 `job_dir`，后续导出和训练都能接上

换句话说，这个工程现在已经把“实验组织层”和“训练层”打通了，唯一还需要你自己接的是“实际配准执行器”。

## 3. 工程目录怎么理解

```text
configs/
  dataset/      数据集级配置
  experiment/   跨序列实验级配置
  shared/       共享静态表
docs/           协议与命名规范
scripts/
  common/       公共函数
  dataset/      manifest/fold/export
  preprocess/   ROI 生成
  experiment/   单序列任务生成、anchor 汇总、registration manifest
  eval/         预测后处理、评估、汇总
  qc/           自检
  workflow/     高层入口包装
src/livermri_crossseq/
  cli/          跨序列 CLI
  data/         workspace/manifest 组织
  registration/ confidence 等逻辑
vendor/nnUNet/  vendored 官方 nnUNet 源码
```

最重要的配置文件是:

- `configs/dataset/livermri_crossseq_dataset.yaml`
- `configs/experiment/livermri_crossseq_fusion.yaml`

最重要的入口脚本是:

- `run_livermri_sequence_screening.ps1`
- `run_livermri_crossseq_pipeline.ps1`

## 4. 环境准备

## 4.1 主环境

在仓库根目录执行:

```powershell
conda create -n livermri_crossseq python=3.10 -y
conda activate livermri_crossseq
pip install -r requirements.txt
```

这一步做的事:

- 以 editable 方式安装本项目
- 以 editable 方式安装 `vendor/nnUNet`
- 安装项目依赖，例如 `nibabel`、`SimpleITK`、`pandas`、`scikit-learn`

产出:

- 当前 Python 环境里可以直接使用:
  - `python -m scripts.*`
  - `python -m scripts.workflow.*`
  - `livermri-bootstrap`
  - `livermri-plan-anchor`
  - `livermri-export-crossseq`
  - `nnUNetv2_*`

## 4.2 TotalSegmentator 环境

这个项目默认认为你有一个单独的 `totalseg` 环境。  
`configs/dataset/livermri_crossseq_dataset.yaml` 里现在写的是:

```yaml
roi:
  conda_env: totalseg
```

常见安装方式是:

```powershell
conda create -n totalseg python=3.10 -y
conda activate totalseg
pip install TotalSegmentator
```

说明:

- 这一步不是本仓库自己安装的，而是为了让 `scripts.preprocess.run_totalsegmentator_liver_roi` 能调起外部 `TotalSegmentator`
- 工程默认用 `--task total_mr --roi_subset liver`

## 4.3 进入仓库根目录

后面的命令默认都在仓库根目录执行:

```powershell
Set-Location -LiteralPath D:\livermri_crossseq_nnunetv2\LiverMRI-CrossSeq-nnUNetv2
```

## 5. 数据应该长什么样

默认数据根目录示例:

```powershell
D:\Dataset003_v2_LiverTumorSeg
```

脚本优先支持两类布局:

- `nnUNet 风格`
  例如 `imagesTr / labelsTr / imagesTs`
- `普通文件树`
  脚本会用启发式去猜 `patient_id / sequence / label`

推荐优先使用 nnUNet 风格，因为:

- case 和 label 对应更稳定
- `subset=train/test` 更容易识别
- 减少 `needs_manual_review`

manifest 里关键列包括:

- `patient_id`
- `case_stem`
- `seq_raw`
- `seq_group`
- `image_path`
- `label_path`
- `subset`
- `fold`
- `cohort_role`

## 6. 一键跑法

如果你只是想先把流程串起来，先用这两个入口。

## 6.1 单序列筛选一键入口

```powershell
.\run_livermri_sequence_screening.ps1
```

它会自动完成:

1. 生成 `manifest_proposed.csv`
2. 生成 `manifest_with_folds.csv`
3. 调 `TotalSegmentator` 生成 liver ROI
4. 生成 `manifest_with_folds_liver_roi.csv`
5. 生成所有单序列筛选任务脚本

注意:

- 这个脚本里数据根目录默认写死成 `D:\Dataset003_v2_LiverTumorSeg`
- ROI 生成默认写了 `--device gpu:1`
- 如果你的机器 GPU 编号不同，要先改这个脚本

## 6.2 跨序列工作区一键入口

```powershell
.\run_livermri_crossseq_pipeline.ps1
```

它会自动完成:

1. 初始化 `workspaces/livermri_crossseq`
2. 生成 workspace 版 manifest
3. 根据单序列 summary 规划 anchor pilot

注意:

- 它不会替你真的执行配准
- 它只是把后续配准和融合训练需要的 workspace 准备好

## 7. 推荐的手动逐步跑法

如果你要真正做研究实验，推荐下面这种“模块化逐步跑法”。  
这样你知道每一步在做什么，失败了也容易定位。

## 7.1 第一步：生成 manifest

命令:

```powershell
python -m scripts.dataset.propose_manifest `
  --root D:\Dataset003_v2_LiverTumorSeg `
  --study-config .\configs\dataset\livermri_crossseq_dataset.yaml `
  --out .\outputs\sequence_screening\manifests\manifest_proposed.csv
```

这一步干什么:

- 扫描数据根目录
- 识别图像和标签
- 根据 `sequence_aliases` 把原始序列名归一到 `T2_MAIN / DWI / ADC / PORTAL / DELAY` 等逻辑名
- 生成供后续实验使用的 case manifest

产出:

- `outputs/sequence_screening/manifests/manifest_proposed.csv`
- `outputs/sequence_screening/manifests/manifest_proposed_summary.json`

策略细节:

- 如果检测到 nnUNet 风格目录，优先用严格规则
- 如果不是，回退到启发式规则
- `needs_manual_review = 1` 的行需要你人工检查

## 7.2 第二步：按 patient 划分 folds

命令:

```powershell
python -m scripts.dataset.assign_folds `
  --manifest .\outputs\sequence_screening\manifests\manifest_proposed.csv `
  --n-folds 5 `
  --seed 3407 `
  --out .\outputs\sequence_screening\manifests\manifest_with_folds.csv
```

这一步干什么:

- 按 `patient_id` 做 5-fold 划分
- 防止同一个患者不同序列同时出现在 train 和 val 中

产出:

- `outputs/sequence_screening/manifests/manifest_with_folds.csv`

策略细节:

- 训练子集默认是 `train / imagesTr`
- 非训练子集会被标成 `external_test`

## 7.3 第三步：生成 liver ROI

命令:

```powershell
python -m scripts.preprocess.run_totalsegmentator_liver_roi `
  --manifest .\outputs\sequence_screening\manifests\manifest_with_folds.csv `
  --study-config .\configs\dataset\livermri_crossseq_dataset.yaml `
  --out-dir D:\Dataset003_v2_LiverTumorSeg\liver_roi_totalseg `
  --device gpu:0 `
  --skip-existing
```

这一步干什么:

- 对每个 case 调 `TotalSegmentator`
- 提取 `liver` mask
- 做最大连通域清理
- 做 hole filling
- 再做 12 mm 膨胀

产出目录:

```text
D:\Dataset003_v2_LiverTumorSeg\liver_roi_totalseg\
  raw\
  clean\
  dilated_12p0mm\
  roi_generation_summary.csv
```

策略细节:

- `raw` 是原始输出
- `clean` 是清理后的肝脏 mask
- `dilated_12p0mm` 用于筛选训练和肝外抑制
- 默认 `min_liver_voxels = 5000`，过小的肝 ROI 会被标记风险

为什么要先做 ROI:

- 你的标签只有 tumor，没有全器官负标签
- 直接训练时，模型会把肝外高信号结构也当成候选病灶
- ROI 先验能显著抑制肝外假阳性

## 7.4 第四步：把 ROI 路径回写到 manifest

命令:

```powershell
python -m scripts.dataset.attach_liver_roi_to_manifest `
  --manifest .\outputs\sequence_screening\manifests\manifest_with_folds.csv `
  --roi-dir D:\Dataset003_v2_LiverTumorSeg\liver_roi_totalseg `
  --out .\outputs\sequence_screening\manifests\manifest_with_folds_liver_roi.csv `
  --dilation-mm 12
```

这一步干什么:

- 给每一行 case 增加:
  - `roi_mask_raw_path`
  - `roi_mask_clean_path`
  - `roi_mask_dilated_path`
  - `roi_available`

产出:

- `outputs/sequence_screening/manifests/manifest_with_folds_liver_roi.csv`

策略细节:

- 后续训练默认用 `roi_mask_dilated_path`
- 因为被膜下病灶可能贴肝包膜，直接用 clean mask 太紧

## 7.5 第五步：生成单序列筛选任务

命令:

```powershell
python -m scripts.experiment.generate_sequence_screening_jobs `
  --manifest .\outputs\sequence_screening\manifests\manifest_with_folds_liver_roi.csv `
  --study-config .\configs\dataset\livermri_crossseq_dataset.yaml `
  --nnunet-raw .\nnUNet_raw_sequence_screening `
  --nnunet-preprocessed .\nnUNet_preprocessed_sequence_screening `
  --nnunet-results .\nnUNet_results_sequence_screening `
  --out-dir .\outputs\sequence_screening\jobs
```

这一步干什么:

- 根据 `configs/dataset/livermri_crossseq_dataset.yaml` 里的 `RS01 ~ RS10`
- 为每个单序列实验生成独立 PowerShell 任务

产出:

```text
outputs/sequence_screening/jobs/
  RS01/
  RS02/
  ...
  RS10/
  job_registry.csv
  run_all_jobs.ps1
```

每个 `RSxx` 任务会做:

1. 导出 ROI 约束训练集到 `nnUNet_raw`
2. 生成 `splits_final.json`
3. `nnUNetv2_plan_and_preprocess`
4. 训练 5 folds
5. 导出 internal/external target 测试集
6. 生成推理命令脚本
7. 对预测做 liver ROI 后处理
8. 计算指标
9. 汇总 paper-ready 表

## 7.6 第六步：运行单序列筛选

跑单个实验:

```powershell
& .\outputs\sequence_screening\jobs\RS08\commands\run_RS08.ps1
```

一次跑全部:

```powershell
& .\outputs\sequence_screening\jobs\run_all_jobs.ps1
```

建议:

- 先跑 1 个实验确认环境没问题
- 确认 `nnUNetv2_*` 命令可用后再批量跑

单个实验主要产出:

```text
outputs/sequence_screening/jobs/RS08/
  commands/
    run_RS08.ps1
    infer_internal_cv.ps1
    infer_external_test.ps1
  targets/
  predictions/
  predictions_postprocessed/
  results/
    per_case_metrics.csv
  reports/
    Table2_internal_source_cv.csv
    Table3_internal_cross_sequence.csv
    Table4_external_test.csv
  evaluation_manifest.csv
  evaluation_manifest_postprocessed.csv
```

策略细节:

- 训练集只用 `internal_cv`
- external 测试集不会泄漏进训练
- ROI 后处理会把肝外预测清掉
- 指标包括:
  - `dice`
  - `hd95`
  - `lesion_recall`
  - `lesion_precision`
  - `lesion_f1`
  - `volume_relative_error`

## 7.7 第七步：汇总单序列结果并做 anchor 初筛

命令:

```powershell
python -m scripts.experiment.summarize_sequence_screening `
  --study-config .\configs\dataset\livermri_crossseq_dataset.yaml `
  --experiments-root .\outputs\sequence_screening\jobs `
  --out-dir .\outputs\sequence_screening\summary `
  --require-external-for-anchor
```

这一步干什么:

- 读取每个 `RSxx/results/per_case_metrics.csv`
- 汇总 internal / external 指标
- 给出 `recommended_anchor.json`

产出:

- `outputs/sequence_screening/summary/sequence_screening_summary.csv`
- `outputs/sequence_screening/summary/recommended_anchor.json`

当前排序策略:

- `external_dice_mean` 越高越好
- `internal_dice_mean` 越高越好
- `external_hd95_mean` 越低越好

这一步的意义:

- 给出一个“直接可用”的 anchor 推荐
- 但它仍然是基于分割表现，不等于最终的最佳配准 anchor

## 7.8 第八步：生成正式 registration manifest

如果你已经决定 anchor，就直接生成配准任务:

```powershell
python -m scripts.experiment.prepare_registration_manifest `
  --manifest .\outputs\sequence_screening\manifests\manifest_with_folds_liver_roi.csv `
  --study-config .\configs\dataset\livermri_crossseq_dataset.yaml `
  --anchor-json .\outputs\sequence_screening\summary\recommended_anchor.json `
  --out-dir .\outputs\sequence_screening\registration
```

这一步干什么:

- 对每个患者找到 anchor 序列
- 为同一患者其他序列生成 `moving -> fixed(anchor)` 的任务清单

产出:

- `outputs/sequence_screening/registration/pairwise_registration_manifest.csv`
- `outputs/sequence_screening/registration/registration_patient_summary.csv`
- `outputs/sequence_screening/registration/registration_meta.json`

每个 pairwise job 里会约定这些输出文件名:

- `registered_image.nii.gz`
- `registered_liver_mask.nii.gz`
- `registered_label.nii.gz`
- `registration_confidence.nii.gz`
- `jacobian_determinant.nii.gz`
- `registration_metrics.json`
- `transform.tfm`

## 7.9 第九步：更严谨的 anchor pilot 规划

如果你不想直接采用 `recommended_anchor.json`，而是想更严谨地在 shortlist 中做 pilot benchmark，用这条线:

先初始化 workspace:

```powershell
python -m scripts.workflow.bootstrap_workspace `
  --dataset-root D:\Dataset003_v2_LiverTumorSeg `
  --study-config .\configs\dataset\livermri_crossseq_dataset.yaml `
  --crossseq-config .\configs\experiment\livermri_crossseq_fusion.yaml
```

再规划 anchor pilot:

```powershell
python -m scripts.workflow.plan_anchor_pilot `
  --screening-summary .\outputs\sequence_screening\summary\sequence_screening_summary.csv `
  --manifest .\workspaces\livermri_crossseq\manifests\manifest_with_folds_liver_roi.csv `
  --study-config .\configs\dataset\livermri_crossseq_dataset.yaml `
  --crossseq-config .\configs\experiment\livermri_crossseq_fusion.yaml
```

这两步干什么:

- 建立干净的跨序列实验工作区
- 根据 `shortlist + coverage + lesion_f1 + hd95` 生成 anchor pilot 表

产出:

```text
workspaces/livermri_crossseq/
  manifests/
  logs/bootstrap_summary.json
  env/workspace_env.ps1
  registration/anchor_pilot/
    anchor_candidate_table.csv
    pilot_patients.csv
    pilot_recommendation.json
```

接下来先加载 workspace 环境变量:

```powershell
. .\workspaces\livermri_crossseq\env\workspace_env.ps1
```

这一步会设置:

- `$env:nnUNet_raw`
- `$env:nnUNet_preprocessed`
- `$env:nnUNet_results`
- `$env:LIVERMRI_CROSSSEQ_WORKSPACE`

当前 `configs/experiment/livermri_crossseq_fusion.yaml` 里的默认 shortlist 是:

- `PORTAL`
- `DELAY`
- `T2_MAIN`

## 7.10 第十步：真正执行配准

这里是当前工程唯一没有统一 runner 的地方。

你要做的事:

- 读取 `pairwise_registration_manifest.csv`
- 逐行执行你的配准方法
- 把结果写回对应 `job_dir`

推荐方法:

- baseline:
  - `affine`
- stronger baseline:
  - `affine + Elastix B-spline`
- main method:
  - `affine + ANTs SyN`

推荐策略:

- fixed 选 anchor
- moving 选同患者其他序列
- 优先用 `roi_mask_clean_path` 做 liver-guided registration
- 把 confidence map 和 jacobian 也保存下来，后续融合会用

当前工程要求的最小契约是:

- `registered_image_path_expected` 必须存在
- 最好同时提供:
  - `registered_liver_mask_path_expected`
  - `registration_confidence_path_expected`

如果没有 `registration_confidence.nii.gz`:

- 导出阶段会退化成根据 anchor liver 和 warped liver overlap 来估计 confidence

## 7.11 第十一步：导出最终多分支数据集

在配准结果就位后执行:

```powershell
. .\workspaces\livermri_crossseq\env\workspace_env.ps1

python -m scripts.workflow.export_registered_multibranch_dataset `
  --manifest .\workspaces\livermri_crossseq\manifests\manifest_with_folds_liver_roi.csv `
  --pairwise-manifest .\outputs\sequence_screening\registration\pairwise_registration_manifest.csv `
  --study-config .\configs\dataset\livermri_crossseq_dataset.yaml `
  --crossseq-config .\configs\experiment\livermri_crossseq_fusion.yaml `
  --nnunet-raw .\workspaces\livermri_crossseq\nnUNet_raw `
  --dataset-id 451
```

这一步干什么:

- 以 anchor 空间为参考
- 读取注册后的各序列
- 生成多通道 nnUNet raw dataset
- 为每个序列附加 confidence channel
- 可选附加 liver mask channel

产出:

```text
workspaces/livermri_crossseq/nnUNet_raw/
  Dataset451_CrossSeqFusion/
    imagesTr/
    labelsTr/
    dataset.json
    case_manifest.csv
```

策略细节:

- 当前候选序列默认是:
  - `PORTAL`
  - `ARTERIAL`
  - `DELAY`
  - `T2_MAIN`
  - `DWI`
  - `ADC`
- 每个序列会占一个 image channel
- 如果开启 `include_confidence_channels: true`，每个序列再带一个 confidence channel
- 如果开启 `include_liver_mask_channel: true`，最后再追加一个 liver mask channel
- 缺失序列会被零填充，confidence 也会置零
- 少于 `minimum_present_sequences` 的患者会被跳过

## 7.12 第十二步：规划和预处理最终融合数据集

这一步要看你准备用哪种 plans。

如果你保持当前配置:

```yaml
model:
  plans_name: nnUNetResEncUNetPlans
```

那就用基础 residual encoder planner:

```powershell
nnUNetv2_plan_and_preprocess -d 451 -pl ResEncUNetPlanner --verify_dataset_integrity
```

如果你想切到官方推荐的 ResEnc preset:

- 24GB 显存附近常用 `nnUNetPlannerResEncL`
- 然后你要同步把 `configs/experiment/livermri_crossseq_fusion.yaml` 里的 `plans_name` 改成对应的 `nnUNetResEncUNetLPlans`

例如:

```powershell
nnUNetv2_plan_and_preprocess -d 451 -pl nnUNetPlannerResEncL --verify_dataset_integrity
```

这一步产出:

- `nnUNet_preprocessed/Dataset451_CrossSeqFusion/`
- dataset fingerprint
- 对应的 plans JSON
- 预处理好的 3D fullres 数据

## 7.13 第十三步：训练最终多分支模型

如果你保持当前配置里的:

- trainer: `nnUNetTrainerCrossSeqFusion`
- plans: `nnUNetResEncUNetPlans`

那么命令是:

```powershell
nnUNetv2_train 451 3d_fullres 0 -tr nnUNetTrainerCrossSeqFusion -p nnUNetResEncUNetPlans
nnUNetv2_train 451 3d_fullres 1 -tr nnUNetTrainerCrossSeqFusion -p nnUNetResEncUNetPlans
nnUNetv2_train 451 3d_fullres 2 -tr nnUNetTrainerCrossSeqFusion -p nnUNetResEncUNetPlans
nnUNetv2_train 451 3d_fullres 3 -tr nnUNetTrainerCrossSeqFusion -p nnUNetResEncUNetPlans
nnUNetv2_train 451 3d_fullres 4 -tr nnUNetTrainerCrossSeqFusion -p nnUNetResEncUNetPlans
```

如果你改成了 `nnUNetPlannerResEncL`，那训练时也要同步改 `-p`:

```powershell
nnUNetv2_train 451 3d_fullres 0 -tr nnUNetTrainerCrossSeqFusion -p nnUNetResEncUNetLPlans
```

这一步训练的不是普通 nnUNet，而是本项目自定义 trainer。  
它额外做了这些事:

- 多分支浅层 stem
- confidence-guided fusion
- warmup 阶段弱 sequence dropout
- 后期更强的 sequence dropout
- aux heads
- consistency loss
- outside-liver penalty

结果会写到:

```text
workspaces/livermri_crossseq/nnUNet_results/
  Dataset451_CrossSeqFusion/
    nnUNetTrainerCrossSeqFusion__<PlansName>__3d_fullres/
      fold_0/
      fold_1/
      ...
```

## 8. 策略层面的解释

## 8.1 为什么先做单序列筛选

因为你当前的问题不是“直接上多模态一定更好”，而是:

- 不同序列的信息密度不同
- 不同序列覆盖率不同
- 有些序列本身畸变大，不适合作 anchor

所以先做单序列筛选，可以回答两个问题:

1. 哪条序列本身最能分割 tumor
2. 哪条序列最值得被纳入最终多分支

## 8.2 为什么不直接把所有序列 GAN 成一个目标序列

当前工程的主线不是“图像翻译后再训”，而是:

- 配准只解决几何对齐
- 原始序列保留自己的物理/对比信息
- 融合发生在特征层

这样做的原因:

- DWI/ADC/T2/增强期并不是“同一图像的不同风格”
- 强行翻译到一个目标序列，容易抹掉有诊断价值的差异
- 审稿时也更容易论证“没有引入生成幻觉”

## 8.3 为什么要 registration-confidence-guided fusion

因为配准永远不是 100% 完美。

如果某个局部区域:

- 配准残差很大
- warped liver 和 anchor liver 不一致
- jacobian 不合理

那就不应该让该序列在那个区域有很大融合权重。  
这个工程的思路是:

- 把 confidence 当作显式输入
- 让融合权重自动下调不可靠序列

## 8.4 最终模型训练策略是什么

当前 trainer 的设计是:

1. 每个序列独立 stem
2. 用 confidence 做 gated fusion
3. 用 aux heads 防止某个分支“完全不学”
4. 前 25% epoch 只做很弱的 dropout
5. 后续提高 sequence dropout 强度
6. 用 consistency loss 保证缺失序列时预测稳定
7. 用 outside-liver penalty 抑制肝外假阳性

这对应的研究假设是:

- 多序列是互补的
- 但并非每个体素上每个序列都同样可信
- 缺失序列和局部错配必须在训练期就模拟

## 9. 关键配置文件要改哪里

## 9.1 `configs/dataset/livermri_crossseq_dataset.yaml`

你通常会改这些字段:

- `dataset_root`
- `sequence_aliases`
- `target_bundles`
- `roi.dilation_mm`
- `roi.crop_margin_mm`
- `experiments.RS01 ~ RS10`

这个文件决定:

- 你的数据怎么看
- 序列怎么归一
- 单序列筛选做哪些实验

## 9.2 `configs/experiment/livermri_crossseq_fusion.yaml`

你通常会改这些字段:

- `anchor.shortlist`
- `anchor.pilot_cases`
- `registration.methods`
- `model.anchor_target`
- `model.candidate_sequences`
- `model.minimum_present_sequences`
- `model.dropout.*`
- `model.loss.*`
- `model.plans_name`
- `export.dataset_id`

这个文件决定:

- anchor pilot 怎么做
- 最终导出哪些序列
- 最终 trainer 怎么训

## 10. 常用命令速查

查看某个模块帮助:

```powershell
python -m scripts.dataset.propose_manifest --help
python -m scripts.preprocess.run_totalsegmentator_liver_roi --help
python -m scripts.experiment.generate_sequence_screening_jobs --help
python -m scripts.workflow.bootstrap_workspace --help
python -m scripts.workflow.plan_anchor_pilot --help
python -m scripts.workflow.export_registered_multibranch_dataset --help
```

做 publication readiness 自检:

```powershell
python -m scripts.qc.self_audit `
  --manifest .\outputs\sequence_screening\manifests\manifest_with_folds_liver_roi.csv `
  --study-config .\configs\dataset\livermri_crossseq_dataset.yaml `
  --out-dir .\outputs\audit
```

## 11. 你第一次上手时最推荐的跑法

如果你是第一次接这个工程，按这个顺序最稳:

1. `pip install -r requirements.txt`
2. 先跑 `python -m scripts.dataset.propose_manifest`
3. 检查 `manifest_proposed.csv` 里是否有 `needs_manual_review = 1`
4. 再跑 fold、ROI、attach ROI
5. 只跑一个 `RS08` 或 `RS09` 单序列实验验证环境
6. 跑全部单序列筛选
7. 生成 `sequence_screening_summary.csv`
8. 再决定是直接 `prepare_registration_manifest`，还是先 `plan_anchor_pilot`
9. 配准结果就位后，再导出 `Dataset451_CrossSeqFusion`
10. 最后训练 `nnUNetTrainerCrossSeqFusion`

## 12. 当前工程最重要的结论

这个仓库不是一个“普通 nnUNet 工程”。

它的核心不是只训练一个分割模型，而是把下面这条研究路径工程化:

- 肝 ROI 约束
- 单序列筛选
- anchor 选择
- 跨序列配准
- confidence-guided 特征融合
- 缺失序列鲁棒训练

如果你只记住一件事，那就是:

先把单序列筛选和 registration manifest 跑通，再去碰最终融合训练。  
这样最符合这个工程当前的设计，也最不容易在中间卡住。
