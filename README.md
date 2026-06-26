# SSL Representation ASR

本仓库用于完成“基于语音自监督模型的 ASR”课程作业，实验方向为冻结 SSL 表征下的低资源语音识别。

## 环境

推荐使用 Conda 环境 `py310`，Python 版本固定为 `3.10.20`：

```powershell
conda create -n py310 python=3.10.20 -y
conda activate py310
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

验证环境：

```powershell
conda run -n py310 python --version
conda run -n py310 python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

如果需要指定 CUDA 版本，请先按 PyTorch 官方说明安装匹配的 `torch` 和 `torchaudio`，再安装其余依赖：

```powershell
conda activate py310
python -m pip install -r requirements.txt
```

如果 PyPI 访问较慢，可以使用公开镜像：

```powershell
python -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

Windows 下可选安装 tmux，用于后台运行完整实验流水线：

```powershell
winget install arndawg.tmux-windows
tmux -V
```

## 下载数据与模型

下载脚本默认直连。数据和模型都会写入被 `.gitignore` 排除的目录，不会提交到 Git。

```powershell
conda activate py310
python scripts/download_assets.py
```

默认行为：

- 下载 Libri-Light `librispeech_finetuning.tgz`，用于官方 1h/10h 标注划分。
- 下载 LibriSpeech `train-clean-100`、`dev-clean`、`test-clean`、`test-other`。
- 下载并固定 WavLM Base+、HuBERT Base、wav2vec 2.0 Base 三个 SSL checkpoint revision。
- 将压缩包保存到 `data/downloads/`，解压到 `data/raw/`，模型保存到 `models/`。
- 已存在且校验通过的文件会自动复用。

常用选项：

```powershell
# 只下载数据，不下载模型
python scripts/download_assets.py --data-only

# 只下载模型
python scripts/download_assets.py --models-only

# 只准备某些划分
python scripts/download_assets.py --data dev-clean test-clean --data-only

# 只准备主模型
python scripts/download_assets.py --models wavlm-base-plus --models-only

# 下载但不解压
python scripts/download_assets.py --data-only --skip-extract
```

如果网络环境必须使用代理，请显式打开环境代理模式，并设置标准代理变量：

```powershell
$env:HTTPS_PROXY="http://<proxy-host>:<proxy-port>"
$env:HTTP_PROXY="http://<proxy-host>:<proxy-port>"
python scripts/download_assets.py --use-env-proxy
```

## 手动下载

自动下载失败时，可以手动下载以下文件并放到 `data/downloads/`，然后重新运行脚本完成校验和解压：

- `https://dl.fbaipublicfiles.com/librilight/data/librispeech_finetuning.tgz`
- `https://www.openslr.org/resources/12/dev-clean.tar.gz`
- `https://www.openslr.org/resources/12/test-clean.tar.gz`
- `https://www.openslr.org/resources/12/test-other.tar.gz`
- `https://www.openslr.org/resources/12/train-clean-100.tar.gz`

模型可从 Hugging Face 页面手动下载 `config.json`、`preprocessor_config.json` 和 `pytorch_model.bin`，分别放入：

- `models/wavlm-base-plus/`：`microsoft/wavlm-base-plus`
- `models/hubert-base-ls960/`：`facebook/hubert-base-ls960`
- `models/wav2vec2-base/`：`facebook/wav2vec2-base`

## 构建 Manifest

下载和解压完成后，可以生成训练、验证和测试用的 JSONL manifest：

```powershell
python scripts/build_manifests.py
```

默认会读取 `data/raw/` 下已存在的划分，并将结果写入 `data/manifests/`。每行包含相对音频路径、原始文本、规范化文本、说话人、章节和 split 名称。

常用选项：

```powershell
# 只构建 dev-clean
python scripts/build_manifests.py --split dev-clean

# 为任意 LibriSpeech 风格目录构建 manifest
python scripts/build_manifests.py --split-dir data/raw/LibriSpeech/dev-clean --split dev-clean --output data/manifests/dev-clean.jsonl
```

## 提取冻结 SSL 特征

Manifest 准备好后，可以从本地 checkpoint 提取指定 hidden layer，并缓存为 FP16：

```powershell
python scripts/extract_features.py --manifest data/manifests/dev-clean.jsonl --model wavlm-base-plus --layer 9 --device cpu
```

默认输出目录为 `features/<model>/layer<layer>/<split>/`。如果机器有可用 CUDA，可以把 `--device cpu` 改为 `--device cuda`，或省略该参数让脚本自动选择。

常用选项：

```powershell
# 只 smoke test 前 1 条
python scripts/extract_features.py --manifest data/manifests/dev-clean.jsonl --model wavlm-base-plus --layer 9 --limit 1

# 比较 HuBERT 第 9 层
python scripts/extract_features.py --manifest data/manifests/dev-clean.jsonl --model hubert-base-ls960 --layer 9
```

## 离散单元

连续特征准备好后，可以拟合 K-means 码本并量化为离散 unit：

```powershell
python scripts/fit_kmeans.py --manifest data/manifests/libri-light-10h.jsonl --model wavlm-base-plus --layer 9 --codebook-size 100
python scripts/quantize_units.py --manifest data/manifests/libri-light-10h.jsonl --model wavlm-base-plus --layer 9 --codebook-size 100
```

默认输出目录为 `units/`，其中包含 codebook、逐帧 unit、相邻去重 unit、run length 和 duration bucket。

## 训练、评测与汇总

生成完整 19 个实验配置：

```powershell
python scripts/generate_experiment_configs.py --output-dir configs/experiments
```

准备所有正式实验需要的 manifest、SSL 特征、K-means codebook 和离散 unit cache：

```powershell
python scripts/prepare_experiment_assets.py --device cuda
```

运行全部 19 个训练配置：

```powershell
python scripts/run_experiments.py --config-dir configs/experiments --device cuda --skip-existing
```

也可以用 Windows tmux 在后台启动完整流水线：

```powershell
python scripts/start_tmux_experiments.py --device cuda
tmux attach -t ssl_asr_full
```

对所有已训练 checkpoint 进行测试集评测并汇总：

```powershell
python scripts/evaluate_experiments.py --config-dir configs/experiments --device cuda --skip-missing
python scripts/summarize.py --outputs-dir outputs --output results/summary.csv
```

快速 smoke test 可以使用 `configs/smoke/` 下的小配置：

```powershell
python scripts/train.py --config configs/smoke/wavlm_continuous_ctc_smoke.yaml --device cuda --limit-train 2 --limit-dev 2
python scripts/train.py --config configs/smoke/wavlm_units_ctc_smoke.yaml --device cuda --limit-train 2 --limit-dev 2
python scripts/train.py --config configs/smoke/wavlm_continuous_transformer_smoke.yaml --device cuda --limit-train 2 --limit-dev 2
```

## 实验结果

完整实验包含 19 个训练配置。训练完成后，每个实验会保存：

- `outputs/<experiment>/metrics.json`：dev 集最优 checkpoint 指标。
- `outputs/<experiment>/eval/test-clean.json`：test-clean 指标。
- `outputs/<experiment>/eval/test-other.json`：test-other 指标。
- `results/summary.csv`：dev、test-clean、test-other 的总表。

仓库只跟踪上述轻量结果文件。数据、SSL 特征缓存、离散 unit、模型 checkpoint 和日志不会提交到 Git。

## 仓库内容

仓库包含项目方案、下载脚本、manifest 构建脚本、冻结 SSL 特征提取脚本、K-means/unit 脚本、训练脚本、评测脚本、汇总脚本、完整实验配置和 smoke 配置。
