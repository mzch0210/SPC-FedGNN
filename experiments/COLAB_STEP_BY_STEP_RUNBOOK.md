# SPC-FedGNN Colab 双账号 Step-by-step 运行指导

本指南按实际操作顺序写，适用于：

- 免费 Colab 账号：T4 GPU 15G，跑轻量任务。
- Colab Pro 账号：A100/L4/T4 等更强 GPU，跑重任务。

建议两个账号都使用同一个 Google Drive 目录保存结果：

```text
/content/drive/MyDrive/SPC-FedGNN/results/colab_full
```

这样两个账号跑出的结果会自动汇合，runner 会跳过已经完成的任务。

---

## Step 0：准备本地上传文件

在本地项目中确认有最新版：

```text
spc_fedgnn_colab_full.zip
```

这个 zip 已包含：

```text
experiments/run_benchmark.py
experiments/colab_full_experiment.py
experiments/summarize_results.py
experiments/aggregate_stage_tables.py
experiments/COLAB_PRO_FULL_EXPERIMENT.md
experiments/COLAB_STEP_BY_STEP_RUNBOOK.md
```

---

## Step 1：两个 Colab 账号都先设置 Runtime

在免费账号：

```text
Runtime -> Change runtime type -> GPU
```

通常会得到 T4。

在 Pro 账号：

```text
Runtime -> Change runtime type -> GPU
```

优先选择 A100 或 L4；如果只有 T4 也可以跑，但重任务会慢。

检查 GPU：

```bash
!nvidia-smi
```

如果没有 GPU，重新切换 runtime。

---

## Step 2：两个账号都挂载 Google Drive

在两个账号中都运行：

```python
from google.colab import drive
drive.mount('/content/drive')
```

创建统一结果目录：

```bash
!mkdir -p /content/drive/MyDrive/SPC-FedGNN/results/colab_full
```

检查目录：

```bash
!ls /content/drive/MyDrive/SPC-FedGNN/results
```

如果能看到 `colab_full`，说明路径正确。

---

## Step 3：两个账号都上传并解压代码包

上传 zip：

```python
from google.colab import files
uploaded = files.upload()
```

选择本地的：

```text
spc_fedgnn_colab_full.zip
```

解压：

```bash
!unzip -o spc_fedgnn_colab_full.zip
```

检查文件：

```bash
!ls experiments
```

应该能看到：

```text
run_benchmark.py
colab_full_experiment.py
aggregate_stage_tables.py
summarize_results.py
```

---

## Step 4：两个账号都安装依赖

先运行：

```python
import torch
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())
```

安装 PyG：

```bash
!pip install -q torch_geometric ogb
```

测试：

```python
import torch
import torch_geometric
print("torch:", torch.__version__)
print("cuda:", torch.version.cuda)
print("gpu:", torch.cuda.is_available())
```

如果导入 `torch_geometric` 报错，再运行：

```bash
!pip install -q pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f "$(python - <<'PY'
import torch
print(f'https://data.pyg.org/whl/torch-{torch.__version__}.html')
PY
)"
```

然后重新测试导入。

---

## Step 5：先只在一个账号跑 smoke

建议先在免费 T4 账号跑 smoke。运行：

```bash
!python experiments/colab_full_experiment.py \
  --stage smoke \
  --output-dir /content/drive/MyDrive/SPC-FedGNN/results/colab_full \
  --keep-going
```

检查 smoke 是否完成：

```bash
!cat /content/drive/MyDrive/SPC-FedGNN/results/colab_full/smoke/task_plan.csv | tail -n +2 | cut -d, -f8 | sort | uniq -c
```

正常应看到：

```text
4 1
```

含义：4 个任务完成。

如果看到：

```text
4 0
```

说明没完成。检查日志：

```bash
!for f in /content/drive/MyDrive/SPC-FedGNN/results/colab_full/smoke/*.log; do echo "==== $f"; tail -n 30 "$f"; done
```

---

## Step 6：免费 T4 账号跑轻量主结果

在免费 T4 账号运行：

```bash
!python experiments/colab_full_experiment.py \
  --stage free_t4_main \
  --output-dir /content/drive/MyDrive/SPC-FedGNN/results/colab_full \
  --keep-going
```

这个 stage 会跑：

```text
Cora, CiteSeer
methods = spc_guard_pp, fedath, fedssp, fedavg_gcn, fedprox_gcn
seeds = 0,1,2
total tasks = 30
```

运行中断也没关系，重新运行同一条命令会自动跳过已完成任务。

检查完成情况：

```bash
!cat /content/drive/MyDrive/SPC-FedGNN/results/colab_full/free_t4_main/task_plan.csv | tail -n +2 | cut -d, -f8 | sort | uniq -c
```

正常全部完成：

```text
30 1
```

如果返回：

```text
30 0
```

说明 `task_plan.csv` 认为 30 个任务都未完成。按下面检查。

### Step 6.1：检查结果是否保存到了正确目录

检查 Drive 目录：

```bash
!find /content/drive/MyDrive/SPC-FedGNN/results/colab_full/free_t4_main -maxdepth 1 -type f | head -50
```

检查本地默认目录：

```bash
!find results/colab_full/free_t4_main -maxdepth 1 -type f 2>/dev/null | head -50
```

如果本地目录有结果，Drive 目录没有，说明运行时漏了 `--output-dir`。复制到 Drive：

```bash
!mkdir -p /content/drive/MyDrive/SPC-FedGNN/results/colab_full/free_t4_main
!cp -n results/colab_full/free_t4_main/* /content/drive/MyDrive/SPC-FedGNN/results/colab_full/free_t4_main/
```

然后刷新计划：

```bash
!python experiments/colab_full_experiment.py \
  --stage free_t4_main \
  --output-dir /content/drive/MyDrive/SPC-FedGNN/results/colab_full \
  --dry-run
```

再检查：

```bash
!cat /content/drive/MyDrive/SPC-FedGNN/results/colab_full/free_t4_main/task_plan.csv | tail -n +2 | cut -d, -f8 | sort | uniq -c
```

### Step 6.2：检查结果文件数量

正常应该有 30 个 aggregated 文件：

```bash
!ls /content/drive/MyDrive/SPC-FedGNN/results/colab_full/free_t4_main/*_aggregated.csv 2>/dev/null | wc -l
```

正常输出：

```text
30
```

如果是 0 或小于 30，继续运行：

```bash
!python experiments/colab_full_experiment.py \
  --stage free_t4_main \
  --output-dir /content/drive/MyDrive/SPC-FedGNN/results/colab_full \
  --keep-going
```

如果想先测试 5 个任务：

```bash
!python experiments/colab_full_experiment.py \
  --stage free_t4_main \
  --output-dir /content/drive/MyDrive/SPC-FedGNN/results/colab_full \
  --max-tasks 5 \
  --keep-going
```

---

## Step 7：免费 T4 账号生成 free_t4_main 表格

跑完 `free_t4_main` 后运行：

```bash
!python experiments/aggregate_stage_tables.py \
  --input /content/drive/MyDrive/SPC-FedGNN/results/colab_full/free_t4_main/enhanced_overview.csv \
  --output /content/drive/MyDrive/SPC-FedGNN/results/colab_full/free_t4_main/stage_table.csv
```

查看表格：

```python
import pandas as pd
pd.read_csv("/content/drive/MyDrive/SPC-FedGNN/results/colab_full/free_t4_main/stage_table.csv")
```

备份下载：

```bash
!zip -r free_t4_main_results.zip /content/drive/MyDrive/SPC-FedGNN/results/colab_full/free_t4_main
```

```python
from google.colab import files
files.download("free_t4_main_results.zip")
```

---

## Step 8：Pro 账号跑重主结果

在 Pro 账号运行：

```bash
!python experiments/colab_full_experiment.py \
  --stage pro_main_heavy \
  --output-dir /content/drive/MyDrive/SPC-FedGNN/results/colab_full \
  --keep-going
```

这个 stage 会跑：

```text
PubMed, Amazon-Photo, Roman-empire
methods = spc_guard_pp, fedath, fedssp, fedavg_gcn, fedprox_gcn
seeds = 0,1,2
total tasks = 45
```

如果想分数据集跑，推荐：

```bash
!python experiments/colab_full_experiment.py \
  --stage pro_main_heavy \
  --datasets pubmed \
  --output-dir /content/drive/MyDrive/SPC-FedGNN/results/colab_full \
  --keep-going
```

```bash
!python experiments/colab_full_experiment.py \
  --stage pro_main_heavy \
  --datasets amazon_photo \
  --output-dir /content/drive/MyDrive/SPC-FedGNN/results/colab_full \
  --keep-going
```

```bash
!python experiments/colab_full_experiment.py \
  --stage pro_main_heavy \
  --datasets roman_empire \
  --output-dir /content/drive/MyDrive/SPC-FedGNN/results/colab_full \
  --keep-going
```

检查完成：

```bash
!cat /content/drive/MyDrive/SPC-FedGNN/results/colab_full/pro_main_heavy/task_plan.csv | tail -n +2 | cut -d, -f8 | sort | uniq -c
```

全部完成应为：

```text
45 1
```

生成表格：

```bash
!python experiments/aggregate_stage_tables.py \
  --input /content/drive/MyDrive/SPC-FedGNN/results/colab_full/pro_main_heavy/enhanced_overview.csv \
  --output /content/drive/MyDrive/SPC-FedGNN/results/colab_full/pro_main_heavy/stage_table.csv
```

---

## Step 9：免费 T4 账号继续跑轻量消融

在免费 T4 账号运行：

```bash
!python experiments/colab_full_experiment.py \
  --stage free_t4_ablation \
  --output-dir /content/drive/MyDrive/SPC-FedGNN/results/colab_full \
  --keep-going
```

这个 stage 会跑 Cora/CiteSeer 的关键消融：

```text
spc_guard_pp
spc_guard
spc_no_protoagg
spc_protoagg
spc_guard_no_gate
spc_guard_no_distill
spc_guard_pp_no_coverage
spc_guard_pp_no_class_balance
```

检查：

```bash
!cat /content/drive/MyDrive/SPC-FedGNN/results/colab_full/free_t4_ablation/task_plan.csv | tail -n +2 | cut -d, -f8 | sort | uniq -c
```

全部完成应为：

```text
48 1
```

生成表格：

```bash
!python experiments/aggregate_stage_tables.py \
  --input /content/drive/MyDrive/SPC-FedGNN/results/colab_full/free_t4_ablation/enhanced_overview.csv \
  --output /content/drive/MyDrive/SPC-FedGNN/results/colab_full/free_t4_ablation/stage_table.csv
```

---

## Step 10：Pro 账号跑 PubMed 消融

在 Pro 账号运行：

```bash
!python experiments/colab_full_experiment.py \
  --stage pro_ablation_heavy \
  --output-dir /content/drive/MyDrive/SPC-FedGNN/results/colab_full \
  --keep-going
```

检查：

```bash
!cat /content/drive/MyDrive/SPC-FedGNN/results/colab_full/pro_ablation_heavy/task_plan.csv | tail -n +2 | cut -d, -f8 | sort | uniq -c
```

全部完成应为：

```text
27 1
```

生成表格：

```bash
!python experiments/aggregate_stage_tables.py \
  --input /content/drive/MyDrive/SPC-FedGNN/results/colab_full/pro_ablation_heavy/enhanced_overview.csv \
  --output /content/drive/MyDrive/SPC-FedGNN/results/colab_full/pro_ablation_heavy/stage_table.csv
```

---

## Step 11：Pro 账号跑异质性实验

在 Pro 账号运行：

```bash
!python experiments/colab_full_experiment.py \
  --stage pro_heterogeneity \
  --output-dir /content/drive/MyDrive/SPC-FedGNN/results/colab_full \
  --keep-going
```

如果太慢，分数据集跑：

```bash
!python experiments/colab_full_experiment.py \
  --stage pro_heterogeneity \
  --datasets cora,citeseer \
  --output-dir /content/drive/MyDrive/SPC-FedGNN/results/colab_full \
  --keep-going
```

```bash
!python experiments/colab_full_experiment.py \
  --stage pro_heterogeneity \
  --datasets pubmed \
  --output-dir /content/drive/MyDrive/SPC-FedGNN/results/colab_full \
  --keep-going
```

生成表格：

```bash
!python experiments/aggregate_stage_tables.py \
  --input /content/drive/MyDrive/SPC-FedGNN/results/colab_full/pro_heterogeneity/enhanced_overview.csv \
  --output /content/drive/MyDrive/SPC-FedGNN/results/colab_full/pro_heterogeneity/stage_table.csv
```

---

## Step 12：免费 T4 账号跑轻量敏感性分析

在免费 T4 账号运行：

```bash
!python experiments/colab_full_experiment.py \
  --stage free_t4_sensitivity \
  --output-dir /content/drive/MyDrive/SPC-FedGNN/results/colab_full \
  --keep-going
```

检查：

```bash
!cat /content/drive/MyDrive/SPC-FedGNN/results/colab_full/free_t4_sensitivity/task_plan.csv | tail -n +2 | cut -d, -f8 | sort | uniq -c
```

全部完成应为：

```text
18 1
```

生成表格：

```bash
!python experiments/aggregate_stage_tables.py \
  --input /content/drive/MyDrive/SPC-FedGNN/results/colab_full/free_t4_sensitivity/enhanced_overview.csv \
  --output /content/drive/MyDrive/SPC-FedGNN/results/colab_full/free_t4_sensitivity/stage_table.csv
```

---

## Step 13：合并免费账号和 Pro 账号结果

如果两个账号使用同一个 Drive 目录，不需要手动合并。

最终目录应类似：

```text
/content/drive/MyDrive/SPC-FedGNN/results/colab_full/
  smoke/
  free_t4_main/
  free_t4_ablation/
  free_t4_sensitivity/
  pro_main_heavy/
  pro_ablation_heavy/
  pro_heterogeneity/
```

检查：

```bash
!find /content/drive/MyDrive/SPC-FedGNN/results/colab_full -maxdepth 2 -name "stage_table.csv"
```

如果两个账号不是同一个 Drive：

1. 在免费账号下载 zip。
2. 在 Pro 账号上传 zip。
3. 解压到根目录 `/`，保持原路径。

免费账号打包：

```bash
!zip -r free_account_results.zip /content/drive/MyDrive/SPC-FedGNN/results/colab_full/free_t4_main /content/drive/MyDrive/SPC-FedGNN/results/colab_full/free_t4_ablation /content/drive/MyDrive/SPC-FedGNN/results/colab_full/free_t4_sensitivity
```

Pro 账号上传：

```python
from google.colab import files
uploaded = files.upload()
```

解压：

```bash
!unzip -o free_account_results.zip -d /
```

---

## Step 14：生成总备份包

在任意一个账号运行：

```bash
!zip -r spc_fedgnn_all_colab_results.zip /content/drive/MyDrive/SPC-FedGNN/results/colab_full
```

下载：

```python
from google.colab import files
files.download("spc_fedgnn_all_colab_results.zip")
```

---

## Step 15：常见问题排查

### 问题 A：`task_plan.csv` 显示全是 0

先看日志里的输出文件名。如果日志出现类似：

```text
--output .../cora_dirichlet_0.csv
```

而不是：

```text
--output .../cora_dirichlet_0.1_k20_spc_guard_pp_s0.csv
```

说明使用了旧版 runner，遇到 `dirichlet_0.1` 的小数点时错误截断文件名。请上传最新版 `spc_fedgnn_colab_full.zip`，重新解压后再运行。旧的 `cora_dirichlet_0.csv` 和 `cora_dirichlet_0_aggregated.csv` 是被多个任务覆盖后的无效文件，不能用于正式汇总。

检查 Drive 目录是否有结果：

```bash
!find /content/drive/MyDrive/SPC-FedGNN/results/colab_full/free_t4_main -maxdepth 1 -type f | head
```

检查是否跑到了本地：

```bash
!find results/colab_full/free_t4_main -maxdepth 1 -type f 2>/dev/null | head
```

如果本地有、Drive 没有，复制：

```bash
!cp -n results/colab_full/free_t4_main/* /content/drive/MyDrive/SPC-FedGNN/results/colab_full/free_t4_main/
```

刷新计划：

```bash
!python experiments/colab_full_experiment.py \
  --stage free_t4_main \
  --output-dir /content/drive/MyDrive/SPC-FedGNN/results/colab_full \
  --dry-run
```

### 问题 B：某个数据集下载失败

查看日志：

```bash
!for f in /content/drive/MyDrive/SPC-FedGNN/results/colab_full/pro_main_heavy/*.log; do echo "==== $f"; tail -n 20 "$f"; done
```

如果 Roman-empire 失败，可换：

```bash
--datasets amazon_ratings
```

或：

```bash
--datasets tolokers
```

### 问题 C：Colab 断线

重新运行同一个 stage 命令即可。已完成任务会自动跳过。

### 问题 D：想强制重跑某个任务

例如强制重跑 Cora seed 0 的 SPC-FedGNN：

```bash
!python experiments/colab_full_experiment.py \
  --stage free_t4_main \
  --datasets cora \
  --seeds 0 \
  --methods spc_guard_pp \
  --force \
  --output-dir /content/drive/MyDrive/SPC-FedGNN/results/colab_full \
  --keep-going
```

### 问题 E：想两个账号拆同一个 stage

账号 A：

```bash
!python experiments/colab_full_experiment.py \
  --stage free_t4_ablation \
  --num-shards 2 \
  --shard-index 0 \
  --output-dir /content/drive/MyDrive/SPC-FedGNN/results/colab_full \
  --keep-going
```

账号 B：

```bash
!python experiments/colab_full_experiment.py \
  --stage free_t4_ablation \
  --num-shards 2 \
  --shard-index 1 \
  --output-dir /content/drive/MyDrive/SPC-FedGNN/results/colab_full \
  --keep-going
```

---

## 推荐总执行顺序

1. 两个账号：挂载 Drive、上传 zip、安装依赖。
2. 免费账号：跑 `smoke`。
3. 免费账号：跑 `free_t4_main`。
4. Pro 账号：跑 `pro_main_heavy`。
5. 免费账号：跑 `free_t4_ablation`。
6. Pro 账号：跑 `pro_ablation_heavy`。
7. Pro 账号：跑 `pro_heterogeneity`。
8. 免费账号：跑 `free_t4_sensitivity`。
9. 每个 stage 完成后生成 `stage_table.csv`。
10. 下载 `spc_fedgnn_all_colab_results.zip` 放回本地项目。
