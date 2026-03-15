# Scripts Layout

当前 `scripts` 已按实验阶段拆分：

- `lib`: 公共函数、ROI 工具
- `data`: manifest、fold、数据导出、ROI 挂载
- `roi`: `TotalSegmentator` 与 liver ROI 生成
- `experiments`: 实验脚本生成、序列筛选汇总、配准任务清单
- `evaluation`: 推理评估与 liver ROI 后处理
- `quality`: 自检与质控

统一调用方式：

```powershell
python -m scripts.<category>.<module> [args...]
```

例如：

```powershell
python -m scripts.roi.run_totalsegmentator_liver_roi --help
python -m scripts.experiments.generate_roi_experiment_scripts --help
```
