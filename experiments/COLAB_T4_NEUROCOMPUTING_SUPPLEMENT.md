# Colab 双 T4 补强实验运行指南

本指南用于在两个 Colab 账号上并行完成 Neurocomputing 投稿前补强实验。两个账号均为 T4 15G GPU。为避免两个账号同时写同一个 `task_plan.csv`，本版教程使用账号专属 stage，而不是手动 shard 同一个 stage。

## 0. 总体分工

账号 A 负责：

- `t4a_extension_amazon`：Amazon-Computers 主结果扩展。
- `t4a_5seed_core`：Cora、CiteSeer 追加 seeds 3 和 4。
- `t4a_diagnostics_light`：Cora、CiteSeer 机制诊断。
- `t4a_efficiency_light`：补 Cora、CiteSeer 的 FedAvg/FedProx 效率字段。

账号 B 负责：

- `t4b_extension_tolokers`：Tolokers 主结果扩展。
- `t4b_5seed_amazon_photo`：Amazon-Photo 追加 seeds 3 和 4。
- `t4b_diagnostics_pubmed`：PubMed 机制诊断。
- `t4b_efficiency_pubmed`：补 PubMed 的 FedAvg/FedProx 效率字段。

最低完成目标：先完成两个扩展数据集和 5-seed 复核。机制诊断与效率表是第二优先级。

## 1. 两个账号都先做环境准备

Colab 设置：

- Runtime type: Python 3
- Hardware accelerator: T4 GPU

挂载 Google Drive：

```python
from google.colab import drive
drive.mount('/content/drive')
```

进入项目目录：

```bash
%cd /content/drive/MyDrive/SPC-FedGNN
```

确认 GPU：

```bash
!nvidia-smi
```

安装依赖：

```bash
!pip -q install torch-geometric
```

如果两个账号不共用同一个 Google Drive，请分别上传最新代码包，并在最后把两个账号的 `results/colab_neuro_supp` 打包下载回本地合并。

## 2. 必须同步的最新文件

确保两个账号中的这些文件是最新版本：

```text
experiments/run_benchmark.py
experiments/summarize_results.py
experiments/aggregate_stage_tables.py
experiments/colab_full_experiment.py
experiments/merge_colab_supp_results.py
experiments/COLAB_T4_NEUROCOMPUTING_SUPPLEMENT.md
```

本版代码新增：

- 账号专属 stage：`t4a_*` 和 `t4b_*`。
- 自动生成 `stage_table.csv`。
- 效率字段：`train_seconds`、`estimated_upload_bytes`、`estimated_download_bytes`。
- 机制诊断字段：`candidate_label_conflict_rate`、`candidate_label_js_mean`、`guard_accept_rate`、`guard_pp_class_accept_rate`。

## 2.1 每个 stage 跑完后的通用检查模板

把下面代码中的 `STAGE` 改成刚跑完的 stage 名称即可。

```bash
STAGE=t4a_extension_amazon
ROOT=/content/drive/MyDrive/SPC-FedGNN/results/colab_neuro_supp

echo "==== done count"
cat "$ROOT/$STAGE/task_plan.csv" | tail -n +2 | cut -d, -f8 | sort | uniq -c

echo "==== unfinished tasks"
python - <<PY
import csv
from pathlib import Path
stage = "$STAGE"
root = Path("$ROOT")
p = root / stage / "task_plan.csv"
rows = list(csv.DictReader(open(p)))
todo = [r for r in rows if r.get("done") != "1"]
print("total", len(rows), "unfinished", len(todo))
for r in todo[:20]:
    print(r["dataset"], r["seed"], r["method"], r["tag"], r["output"])
PY

echo "==== result files"
find "$ROOT/$STAGE" -maxdepth 1 -name '*.csv' | wc -l
ls "$ROOT/$STAGE"/stage_table.csv "$ROOT/$STAGE"/enhanced_overview.csv "$ROOT/$STAGE"/aggregated_overview.csv

echo "==== failed logs"
grep -L "Wrote" "$ROOT/$STAGE"/*.log 2>/dev/null | head
```

如果 `unfinished` 不是 0，直接重复运行该 stage 命令即可断点续跑。

## 3. 账号 A：先跑 Amazon-Computers

```bash
!python experiments/colab_full_experiment.py \
  --stage t4a_extension_amazon \
  --output-dir /content/drive/MyDrive/SPC-FedGNN/results/colab_neuro_supp \
  --keep-going
```

检查完成度：

```bash
STAGE=t4a_extension_amazon
ROOT=/content/drive/MyDrive/SPC-FedGNN/results/colab_neuro_supp

cat "$ROOT/$STAGE/task_plan.csv" | tail -n +2 | cut -d, -f8 | sort | uniq -c
python - <<PY
import csv
from pathlib import Path
p=Path("$ROOT")/"$STAGE"/"task_plan.csv"
rows=list(csv.DictReader(open(p)))
todo=[r for r in rows if r.get("done")!="1"]
print("total", len(rows), "unfinished", len(todo))
for r in todo[:20]:
    print(r["dataset"], r["seed"], r["method"], r["output"])
PY
ls "$ROOT/$STAGE"/stage_table.csv "$ROOT/$STAGE"/enhanced_overview.csv
python - <<PY
import pandas as pd
p="$ROOT/$STAGE/stage_table.csv"
df=pd.read_csv(p)
print(df[["dataset","method","num_seeds","mean_macro_f1_mean","delta_mean_macro_f1_vs_fedath","delta_mean_macro_f1_vs_fedssp"]].to_string(index=False))
PY
```

## 4. 账号 B：同时跑 Tolokers

```bash
!python experiments/colab_full_experiment.py \
  --stage t4b_extension_tolokers \
  --output-dir /content/drive/MyDrive/SPC-FedGNN/results/colab_neuro_supp \
  --keep-going
```

如果 Tolokers 下载失败：

1. 先重新运行同一命令一次。
2. 如果仍失败，不要阻塞账号 B，先继续跑 `t4b_5seed_amazon_photo`。
3. 最终论文中可以把 Tolokers 标为未完成扩展，优先使用 Amazon-Computers 补强。

检查完成度：

```bash
STAGE=t4b_extension_tolokers
ROOT=/content/drive/MyDrive/SPC-FedGNN/results/colab_neuro_supp

cat "$ROOT/$STAGE/task_plan.csv" | tail -n +2 | cut -d, -f8 | sort | uniq -c
python - <<PY
import csv
from pathlib import Path
p=Path("$ROOT")/"$STAGE"/"task_plan.csv"
rows=list(csv.DictReader(open(p)))
todo=[r for r in rows if r.get("done")!="1"]
print("total", len(rows), "unfinished", len(todo))
for r in todo[:20]:
    print(r["dataset"], r["seed"], r["method"], r["output"])
PY
ls "$ROOT/$STAGE"/stage_table.csv "$ROOT/$STAGE"/enhanced_overview.csv
python - <<PY
import pandas as pd
p="$ROOT/$STAGE/stage_table.csv"
df=pd.read_csv(p)
print(df[["dataset","method","num_seeds","mean_macro_f1_mean","delta_mean_macro_f1_vs_fedath","delta_mean_macro_f1_vs_fedssp"]].to_string(index=False))
PY
```

## 5. 账号 A：5-seed Cora/CiteSeer

```bash
!python experiments/colab_full_experiment.py \
  --stage t4a_5seed_core \
  --output-dir /content/drive/MyDrive/SPC-FedGNN/results/colab_neuro_supp \
  --keep-going
```

该 stage 只跑 seeds 3 和 4。写论文时要和已有 seeds 0、1、2 合并成 5 seeds。

检查完成度：

```bash
STAGE=t4a_5seed_core
ROOT=/content/drive/MyDrive/SPC-FedGNN/results/colab_neuro_supp

cat "$ROOT/$STAGE/task_plan.csv" | tail -n +2 | cut -d, -f8 | sort | uniq -c
python - <<PY
import csv
from pathlib import Path
p=Path("$ROOT")/"$STAGE"/"task_plan.csv"
rows=list(csv.DictReader(open(p)))
todo=[r for r in rows if r.get("done")!="1"]
print("total", len(rows), "unfinished", len(todo))
for r in todo[:20]:
    print(r["dataset"], r["seed"], r["method"], r["output"])
PY
python - <<PY
import pandas as pd
p="$ROOT/$STAGE/stage_table.csv"
df=pd.read_csv(p)
print(df[["dataset","method","seeds","num_seeds","mean_macro_f1_mean","win_count_vs_fedath","win_count_vs_fedssp"]].to_string(index=False))
PY
```

## 6. 账号 B：5-seed Amazon-Photo

```bash
!python experiments/colab_full_experiment.py \
  --stage t4b_5seed_amazon_photo \
  --output-dir /content/drive/MyDrive/SPC-FedGNN/results/colab_neuro_supp \
  --keep-going
```

Amazon-Photo 原主结果提升较小，所以这个 5-seed 复核很重要。若结果不稳定，正文对 Amazon-Photo 要保守表述。

检查完成度：

```bash
STAGE=t4b_5seed_amazon_photo
ROOT=/content/drive/MyDrive/SPC-FedGNN/results/colab_neuro_supp

cat "$ROOT/$STAGE/task_plan.csv" | tail -n +2 | cut -d, -f8 | sort | uniq -c
python - <<PY
import csv
from pathlib import Path
p=Path("$ROOT")/"$STAGE"/"task_plan.csv"
rows=list(csv.DictReader(open(p)))
todo=[r for r in rows if r.get("done")!="1"]
print("total", len(rows), "unfinished", len(todo))
for r in todo[:20]:
    print(r["dataset"], r["seed"], r["method"], r["output"])
PY
python - <<PY
import pandas as pd
p="$ROOT/$STAGE/stage_table.csv"
df=pd.read_csv(p)
print(df[["dataset","method","seeds","num_seeds","mean_macro_f1_mean","win_count_vs_fedath","win_count_vs_fedssp"]].to_string(index=False))
PY
```

## 7. 账号 A：Cora/CiteSeer 机制诊断

```bash
!python experiments/colab_full_experiment.py \
  --stage t4a_diagnostics_light \
  --output-dir /content/drive/MyDrive/SPC-FedGNN/results/colab_neuro_supp \
  --keep-going
```

输出字段用于写机制分析：

- `candidate_label_conflict_rate`
- `candidate_label_js_mean`
- `guard_accept_rate`
- `guard_pp_class_accept_rate`

检查完成度和诊断字段：

```bash
STAGE=t4a_diagnostics_light
ROOT=/content/drive/MyDrive/SPC-FedGNN/results/colab_neuro_supp

cat "$ROOT/$STAGE/task_plan.csv" | tail -n +2 | cut -d, -f8 | sort | uniq -c
python - <<PY
import csv
from pathlib import Path
p=Path("$ROOT")/"$STAGE"/"task_plan.csv"
rows=list(csv.DictReader(open(p)))
todo=[r for r in rows if r.get("done")!="1"]
print("total", len(rows), "unfinished", len(todo))
for r in todo[:20]:
    print(r["dataset"], r["seed"], r["method"], r["output"])
PY
python - <<PY
import pandas as pd
p="$ROOT/$STAGE/stage_table.csv"
df=pd.read_csv(p)
cols=["dataset","method","num_seeds","mean_macro_f1_mean","candidate_label_conflict_rate_mean","candidate_label_js_mean_mean","guard_accept_rate_mean","guard_pp_class_accept_rate_mean"]
print(df[[c for c in cols if c in df.columns]].to_string(index=False))
PY
```

## 8. 账号 B：PubMed 机制诊断

```bash
!python experiments/colab_full_experiment.py \
  --stage t4b_diagnostics_pubmed \
  --output-dir /content/drive/MyDrive/SPC-FedGNN/results/colab_neuro_supp \
  --keep-going
```

PubMed 是当前优势最明显的数据集之一，适合放入机制诊断表。

检查完成度和诊断字段：

```bash
STAGE=t4b_diagnostics_pubmed
ROOT=/content/drive/MyDrive/SPC-FedGNN/results/colab_neuro_supp

cat "$ROOT/$STAGE/task_plan.csv" | tail -n +2 | cut -d, -f8 | sort | uniq -c
python - <<PY
import csv
from pathlib import Path
p=Path("$ROOT")/"$STAGE"/"task_plan.csv"
rows=list(csv.DictReader(open(p)))
todo=[r for r in rows if r.get("done")!="1"]
print("total", len(rows), "unfinished", len(todo))
for r in todo[:20]:
    print(r["dataset"], r["seed"], r["method"], r["output"])
PY
python - <<PY
import pandas as pd
p="$ROOT/$STAGE/stage_table.csv"
df=pd.read_csv(p)
cols=["dataset","method","num_seeds","mean_macro_f1_mean","candidate_label_conflict_rate_mean","candidate_label_js_mean_mean","guard_accept_rate_mean","guard_pp_class_accept_rate_mean"]
print(df[[c for c in cols if c in df.columns]].to_string(index=False))
PY
```

## 9. 账号 A：Cora/CiteSeer 效率补充

```bash
!python experiments/colab_full_experiment.py \
  --stage t4a_efficiency_light \
  --output-dir /content/drive/MyDrive/SPC-FedGNN/results/colab_neuro_supp \
  --keep-going
```

这个 stage 只补 FedAvg-GCN 和 FedProx-GCN。SPC-FedGNN、SPC-Guard、FedATH、FedSSP 的 `train_seconds` 已由诊断 stage 输出，后续在本地合并表中一起使用。

检查完成度和效率字段：

```bash
STAGE=t4a_efficiency_light
ROOT=/content/drive/MyDrive/SPC-FedGNN/results/colab_neuro_supp

cat "$ROOT/$STAGE/task_plan.csv" | tail -n +2 | cut -d, -f8 | sort | uniq -c
python - <<PY
import csv
from pathlib import Path
p=Path("$ROOT")/"$STAGE"/"task_plan.csv"
rows=list(csv.DictReader(open(p)))
todo=[r for r in rows if r.get("done")!="1"]
print("total", len(rows), "unfinished", len(todo))
for r in todo[:20]:
    print(r["dataset"], r["seed"], r["method"], r["output"])
PY
python - <<PY
import pandas as pd
p="$ROOT/$STAGE/stage_table.csv"
df=pd.read_csv(p)
cols=["dataset","method","train_seconds_mean","estimated_upload_bytes_mean","estimated_download_bytes_mean","estimated_model_upload_bytes_mean","estimated_model_download_bytes_mean"]
print(df[[c for c in cols if c in df.columns]].to_string(index=False))
PY
```

## 10. 账号 B：PubMed 效率补充

```bash
!python experiments/colab_full_experiment.py \
  --stage t4b_efficiency_pubmed \
  --output-dir /content/drive/MyDrive/SPC-FedGNN/results/colab_neuro_supp \
  --keep-going
```

这个 stage 同样只补 FedAvg-GCN 和 FedProx-GCN。

检查完成度和效率字段：

```bash
STAGE=t4b_efficiency_pubmed
ROOT=/content/drive/MyDrive/SPC-FedGNN/results/colab_neuro_supp

cat "$ROOT/$STAGE/task_plan.csv" | tail -n +2 | cut -d, -f8 | sort | uniq -c
python - <<PY
import csv
from pathlib import Path
p=Path("$ROOT")/"$STAGE"/"task_plan.csv"
rows=list(csv.DictReader(open(p)))
todo=[r for r in rows if r.get("done")!="1"]
print("total", len(rows), "unfinished", len(todo))
for r in todo[:20]:
    print(r["dataset"], r["seed"], r["method"], r["output"])
PY
python - <<PY
import pandas as pd
p="$ROOT/$STAGE/stage_table.csv"
df=pd.read_csv(p)
cols=["dataset","method","train_seconds_mean","estimated_upload_bytes_mean","estimated_download_bytes_mean","estimated_model_upload_bytes_mean","estimated_model_download_bytes_mean"]
print(df[[c for c in cols if c in df.columns]].to_string(index=False))
PY
```

查看效率字段：

```bash
!python - <<'PY'
import pandas as pd
from pathlib import Path
root=Path('/content/drive/MyDrive/SPC-FedGNN/results/colab_neuro_supp')
for stage in ['t4a_efficiency_light','t4b_efficiency_pubmed']:
    p=root/stage/'stage_table.csv'
    if p.exists():
        df=pd.read_csv(p)
        cols=[
            'dataset','method','mean_macro_f1_mean','train_seconds_mean',
            'estimated_upload_bytes_mean','estimated_download_bytes_mean',
            'estimated_descriptor_upload_bytes_mean','estimated_candidate_download_bytes_mean'
        ]
        print('\\n====', stage)
        print(df[cols].to_string(index=False))
PY
```

## 11. 断点续跑

所有 stage 默认跳过已有 raw CSV 和 aggregated CSV。运行中断后重复同一命令即可。

只在确实要重跑时使用 `--force`。

查看失败日志：

```bash
!for f in /content/drive/MyDrive/SPC-FedGNN/results/colab_neuro_supp/*/*.log; do echo "==== $f"; tail -n 12 "$f"; done
```

## 12. 不推荐两个账号同时跑同一个 stage

代码仍支持：

```bash
--num-shards 2 --shard-index 0
--num-shards 2 --shard-index 1
```

但如果两个账号写同一个 stage 目录，会覆盖 `task_plan.csv`，不利于检查完成度。除非你给两个账号设置不同 `--output-dir`，否则优先使用本教程的 `t4a_*` 与 `t4b_*` stage。

## 13. 打包结果

如果两个账号共用同一 Drive：

```bash
%cd /content/drive/MyDrive/SPC-FedGNN
!zip -r spc_fedgnn_neuro_supp_dual_t4_results.zip results/colab_neuro_supp
```

如果两个账号使用不同 Drive，请分别打包：

```bash
%cd /content/drive/MyDrive/SPC-FedGNN
!zip -r spc_fedgnn_neuro_supp_account_a.zip results/colab_neuro_supp
```

```bash
%cd /content/drive/MyDrive/SPC-FedGNN
!zip -r spc_fedgnn_neuro_supp_account_b.zip results/colab_neuro_supp
```

放回本地项目后，保留两个 zip 即可，我会再帮你合并。

## 14. 完成后重点检查

```text
results/colab_neuro_supp/t4a_extension_amazon/stage_table.csv
results/colab_neuro_supp/t4b_extension_tolokers/stage_table.csv
results/colab_neuro_supp/t4a_5seed_core/stage_table.csv
results/colab_neuro_supp/t4b_5seed_amazon_photo/stage_table.csv
results/colab_neuro_supp/t4a_diagnostics_light/stage_table.csv
results/colab_neuro_supp/t4b_diagnostics_pubmed/stage_table.csv
results/colab_neuro_supp/t4a_efficiency_light/stage_table.csv
results/colab_neuro_supp/t4b_efficiency_pubmed/stage_table.csv
```

## 15. 本地合并两个账号结果

把两个账号的 zip 放回本地项目根目录并解压后，运行：

```bash
python3 experiments/merge_colab_supp_results.py \
  --inputs results/colab_neuro_supp \
  --output-dir results/colab_neuro_supp_merged
```

如果两个账号解压在不同目录，例如：

```text
results/account_a/colab_neuro_supp
results/account_b/colab_neuro_supp
```

则运行：

```bash
python3 experiments/merge_colab_supp_results.py \
  --inputs results/account_a/colab_neuro_supp results/account_b/colab_neuro_supp \
  --output-dir results/colab_neuro_supp_merged
```

输出重点看：

```text
results/colab_neuro_supp_merged/combined_enhanced_overview.csv
results/colab_neuro_supp_merged/combined_stage_table.csv
results/colab_neuro_supp_merged/combined_stage_tables_raw.csv
```

`combined_stage_table.csv` 会按 dataset、partition、clients、method、run_tag 汇总，并计算相对 FedATH、FedSSP、FedPub-style 和 SPC-Guard 的差值、win count、paired delta 与双侧 sign-test p-value。效率表建议从这个合并表读取，因为诊断 stage 和效率补充 stage 的时间/通信字段会在这里汇合。

## 16. 论文使用方式

- Amazon-Computers/Tolokers 若 SPC-FedGNN 继续领先，可加入主结果表或附录扩展表。
- Cora/CiteSeer/Amazon-Photo 的 5-seed 结果用于增强统计稳健性。
- `candidate_label_conflict_rate` 与 `candidate_label_js_mean` 用于解释 direct ProtoAgg 的负迁移。
- `guard_accept_rate` 与 `guard_pp_class_accept_rate` 用于解释 guarded collaboration 如何筛选候选知识。
- `train_seconds` 和通信估计字段用于效率/通信表。

## 17. 新增 FedPub-style baseline 补强实验

这一步用于降低“缺少 personalized FGL 强 baseline”的审稿风险。代码中的方法名是 `fedpub_gcn`，论文中建议写为 **FedPub-style GCN under our unified protocol**，不要写成官方 FedPub 结果。

账号 A 跑 citation datasets：

```bash
%cd /content/drive/MyDrive/SPC-FedGNN
!python3 experiments/colab_full_experiment.py \
  --stage t4a_fedpub_citation \
  --output-dir /content/drive/MyDrive/SPC-FedGNN/results/colab_neuro_supp \
  --data-root data \
  --keep-going
```

账号 A 跑完后一行检查：

```bash
!cat /content/drive/MyDrive/SPC-FedGNN/results/colab_neuro_supp/t4a_fedpub_citation/task_plan.csv | tail -n +2 | cut -d, -f8 | sort | uniq -c
```

账号 B 跑扩展数据集：

```bash
%cd /content/drive/MyDrive/SPC-FedGNN
!python3 experiments/colab_full_experiment.py \
  --stage t4b_fedpub_extended \
  --output-dir /content/drive/MyDrive/SPC-FedGNN/results/colab_neuro_supp \
  --data-root data \
  --keep-going
```

账号 B 跑完后一行检查：

```bash
!cat /content/drive/MyDrive/SPC-FedGNN/results/colab_neuro_supp/t4b_fedpub_extended/task_plan.csv | tail -n +2 | cut -d, -f8 | sort | uniq -c
```

两个检查都应只返回 `9 1` 或 `12 1` 这类格式，其中左侧数字分别对应该 stage 的任务数，右侧 `1` 表示完成。若出现 `0`，说明还有任务未完成，可以重复运行同一命令续跑。

补强实验完成后重新打包：

```bash
%cd /content/drive/MyDrive/SPC-FedGNN
!zip -r spc_fedgnn_neuro_supp_with_fedpub_results.zip results/colab_neuro_supp
```
