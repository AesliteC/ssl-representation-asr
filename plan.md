# 冻结语音自监督表征的低资源 ASR 方案

## 1. 总体方案

- 使用官方 Libri-Light `10h` 标注集训练，`dev-clean` 用于模型选择，`test-clean` 和 `test-other` 用于最终评测。
- 使用官方 Libri-Light `1h` 子集进行标注数据规模消融。
- 主上游为冻结的 `microsoft/wavlm-base-plus`，不微调、不反向传播，仅离线提取 hidden states。
- 研究主线是连续表征与 K-means 离散单元在识别准确率、推理成本和信息率上的权衡。
- 下游比较 `2-layer BiLSTM + CTC` 与 `Transformer Encoder-Decoder`。
- 不使用外部语言模型或数据增强，避免掩盖语音表征本身带来的差异。

## 2. 运行环境

- Conda 环境：`py310`
- Python：`3.10.20`
- 核心依赖：PyTorch、Transformers、torchaudio、scikit-learn、jiwer、TensorBoard
- 所有数据准备、特征提取、训练和评测命令均在 `py310` 环境中运行。

## 3. 数据集与文本处理

- 训练集：Libri-Light 官方 `10h` 有标注划分，音频来自 LibriSpeech。
- 数据规模消融：Libri-Light 官方 `1h` 有标注划分。
- 验证集：`dev-clean`，仅用于早停、选层和超参数选择。
- 测试集：`test-clean`、`test-other`，只在配置锁定后进行最终评测。
- 音频统一为单声道、16 kHz。
- 文本转换为大写并移除无关标点，字符表包括 `A-Z`、撇号和空格。
- CTC 额外使用 blank，Transformer 额外使用 BOS、EOS 和 PAD。

### 3.1 数据与模型下载

- 项目脚本负责将 LibriSpeech 音频和 Libri-Light 官方 1h/10h manifest 下载到 `data/`。
- 项目脚本负责将 WavLM Base+、HuBERT Base 和 wav2vec 2.0 Base 下载到 `models/` 或项目指定的 Hugging Face 缓存目录。
- 下载过程支持通过 `HTTPS_PROXY=http://127.0.0.1:7892` 访问外部资源，并支持断点续传和已下载文件复用。
- 下载完成后校验所需划分、文件数量、可读取性及模型 checkpoint revision。
- 若自动下载因授权、网络或镜像限制失败，项目文档必须给出官方下载地址、可直接运行的命令以及文件应放置的本地目录。
- 数据集、模型权重和缓存均不得提交到 Git 仓库。

## 4. 语音自监督表征

### 4.1 主模型

- 使用 `microsoft/wavlm-base-plus`。
- 上游始终处于 `eval()` 状态，所有参数设置为 `requires_grad=False`。
- 特征提取使用 `torch.inference_mode()`，缓存为 FP16。
- 比较 Transformer 第 3、6、9、12 层 hidden states，以 `dev-clean WER` 选择最佳层。
- 连续与离散主实验使用同一最佳层，确保比较公平。

### 4.2 SSL 模型消融

- `microsoft/wavlm-base-plus`
- `facebook/hubert-base-ls960`
- `facebook/wav2vec2-base`
- 三种模型统一比较第 9 层连续表征，并使用相同的 BiLSTM-CTC 下游。

## 5. 下游模型

### 5.1 BiLSTM-CTC

- 连续输入：`LayerNorm -> Linear(768, 256)`。
- 离散输入：`Embedding(codebook_size, 256)`。
- 两层双向 LSTM，每个方向 hidden size 为 256，dropout 为 0.2。
- 输出为字符级 CTC logits，使用 greedy CTC 解码。

### 5.2 Transformer Encoder-Decoder

- 输入适配层后使用一层 stride-2 Conv1d 降低时间分辨率。
- Encoder：4 层，Decoder：3 层。
- `d_model=256`，4 个 attention heads，FFN 维度 1024，dropout 0.1。
- 使用字符级自回归交叉熵和 0.1 label smoothing。
- 推理使用 beam size 5，不融合外部语言模型。

### 5.3 非 SSL 基线

- 提取 80 维 log-Mel 特征，hop length 为 20 ms，使帧率与 WavLM 接近。
- log-Mel 分别接入 BiLSTM-CTC 和 Transformer Encoder-Decoder。

## 6. 离散语音单元

- 从选定的 WavLM 层提取训练集帧，并均匀采样 20 万帧。
- 特征先进行 L2 归一化，再拟合 `MiniBatchKMeans`。
- K-means 使用 batch size 4096、3 次初始化、最多 100 iterations、随机种子 42。
- 比较 `K=50/100/200`，主码本使用 `K=100`。
- 主离散表示保留逐帧 unit ID，维持约 50 token/s 的时间轴。
- 压缩消融包括相邻重复 unit 去重，以及去重后加入 8 档时长 embedding。
- 时长档定义为 `min(7, floor(log2(run_length)))`。
- 固定长度编码 bitrate 定义为 `token_rate * ceil(log2(K))`；时长表示额外计入每个 token 3 bit。

## 7. 实验矩阵

计划完成 17 个训练配置：

| 实验组 | 配置数 | 内容 |
| --- | ---: | --- |
| 非 SSL 基线 | 2 | log-Mel + CTC/Transformer |
| WavLM 层选择 | 4 | 第 3/6/9/12 层连续表征 + CTC |
| SSL 模型比较 | 2 | HuBERT Base、wav2vec 2.0 Base 第 9 层 + CTC |
| 码本大小 | 3 | 最佳 WavLM 层，K=50/100/200 + CTC |
| 序列压缩 | 2 | K=100 去重、去重+时长 + CTC |
| 标注数据规模 | 2 | 1h 连续与 K=100 离散 + CTC |
| Transformer 主系统 | 2 | 10h 连续与 K=100 逐帧离散 |

主结果表形成连续/离散表征与 CTC/Transformer 下游的 2x2 对比。

## 8. 训练策略

- 优化器：AdamW，weight decay 为 `1e-2`。
- CTC 初始学习率：`1e-3`。
- Transformer 初始学习率：`3e-4`。
- 使用自动混合精度，梯度裁剪阈值为 5.0。
- 最多训练 50 epochs，以 `dev-clean WER` 早停，patience 为 8。
- 所有主实验固定随机种子 42。
- 测试集不参与模型、层或超参数选择。

## 9. 评价与分析

- 核心指标：WER、CER。
- 错误分析：替换、删除、插入错误数量及典型案例。
- 系统效率：总参数量、可训练参数量、端到端 RTF、下游 RTF。
- 端到端 RTF 从原始 waveform 开始计时，包含 SSL 特征提取，batch size 为 1。
- 离散系统额外报告 token rate、bitrate、码本大小和实际激活单元数。
- 2x2 主结果通过 utterance bootstrap 给出 95% WER 置信区间。
- 绘制 hidden layer/WER、codebook size/WER 和 WER/bitrate 权衡图。

## 10. 实现接口

代码按数据准备、特征缓存、离散化、模型、训练、评测和结果汇总拆分，并由 YAML 配置驱动。统一提供以下命令：

- `prepare-data`
- `extract-features`
- `fit-kmeans`
- `train`
- `evaluate`
- `summarize`

配置需明确记录数据划分、上游 checkpoint 及 revision、hidden layer、表示类型、码本、下游模型和随机种子。评测产物包含逐句参考和预测、指标 JSON、汇总 CSV、实验配置及 checkpoint revision。

## 11. 测试与验收

- 单元测试覆盖文本规范化、数据集无交叉、padding mask、CTC 解码、K-means ID 范围、去重、时长恢复、bitrate 和 WER/CER。
- 验证 SSL 模型参数无梯度、始终处于评估模式，并且相同音频的缓存特征可重复。
- 使用少量语音完成“数据 -> 特征 -> 聚类 -> 两种下游 -> 评测”的端到端 smoke test。
- 全部 17 个配置必须能通过配置文件复现。
- 主表、消融表和分析图能够从实验结果自动重建。

## 12. 范围边界与研究假设

- 本项目不进行 SSL 上游微调、语言模型融合、多语言实验或 100h 训练。
- 假设连续表征具有更低 WER，离散表征能够显著降低 bitrate。
- 假设时长 embedding 能够部分恢复去重造成的识别性能损失。
- Transformer 是否优于 CTC 由实验确定，不预设结果。
