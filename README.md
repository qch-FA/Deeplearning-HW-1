# Deeplearning-HW-1
Codebase for Homework 1 of the 2026 Spring Course CS60003: Deep Learning and Spatial Intelligence at FDU.

## 1. 环境依赖

### 1.1 Python 版本

建议使用：

- `Python 3.10+`

本项目在 `Python 3.13` 环境下已实际跑通。

### 1.2 依赖包

训练与测试所需的核心依赖如下：

- `numpy`
- `Pillow`
- `matplotlib`

### 1.3 安装命令

在项目根目录下执行：

```bash
python -m pip install numpy Pillow matplotlib
```

---

## 2. 目录结构

项目默认要求数据集目录结构如下：

```text
hw1/
├─ EuroSAT_RGB/
│  ├─ AnnualCrop/
│  ├─ Forest/
│  ├─ ...
├─ hw1_mlp/
├─ run_hw1.py
├─ evaluate_hw1.py
└─ README.md
```

## 3. 如何运行训练脚本

训练入口脚本为：

- `run_hw1.py`

该脚本会自动完成以下流程：

1. 读取并预处理 `EuroSAT_RGB`
2. 划分训练集 / 验证集 / 测试集
3. 执行超参数搜索
4. 用最佳超参数重新训练最终模型
5. 保存最优权重
6. 在测试集上评估
7. 生成混淆矩阵、训练曲线、权重可视化、错例分析等结果

### 3.1 最简单的运行方式

```bash
python run_hw1.py
```

默认参数包括：

- `data-root=EuroSAT_RGB`
- `output-dir=outputs`
- `image-size=32`
- `batch-size=256`
- `search-trials=4`
- `search-epochs=4`
- `final-epochs=12`

### 3.2 复现实验报告结果的训练命令

如果想复现当前项目里最终采用的正式实验结果，可以运行：

```bash
python run_hw1.py --image-size 32 --search-trials 6 --search-epochs 6 --final-epochs 24 --batch-size 256 --hidden-space 192x96 256x128 320x160 --lr-space 0.03 0.05 --weight-decay-space 0.0001 0.0005 --activation-space relu tanh --lr-decay 0.95 --output-dir outputs
```

这条命令会在 `outputs/` 下生成正式实验结果，并保存最佳模型权重。

### 3.3 常用可调参数

可以通过命令行修改以下超参数：

- `--data-root`：数据集目录
- `--output-dir`：输出目录
- `--image-size`：输入图像缩放大小
- `--train-ratio`：训练集比例
- `--val-ratio`：验证集比例
- `--seed`：随机种子
- `--batch-size`：批大小
- `--search-trials`：超参数搜索次数
- `--search-epochs`：每次搜索训练轮数
- `--final-epochs`：最终模型训练轮数
- `--lr-decay`：学习率衰减系数
- `--hidden-space`：隐藏层候选规模
- `--lr-space`：学习率候选集合
- `--weight-decay-space`：L2 正则候选集合
- `--activation-space`：激活函数候选集合

例如：

```bash
python run_hw1.py --output-dir outputs_try --search-trials 2 --search-epochs 2 --final-epochs 5
```

---

## 4. 如何运行测试脚本

测试入口脚本为：

- `evaluate_hw1.py`

该脚本用于：

- 加载已经训练好的模型权重
- 在测试集上重新评估
- 输出测试集 Loss、Accuracy 和混淆矩阵

### 4.1 使用默认最佳模型测试

如果权重文件存在：

```bash
python evaluate_hw1.py
```

默认会读取：

- 权重文件：`outputs/final/best_model.npz`
- 元信息文件：`outputs/final/best_model.json`


### 4.2 指定其他模型权重测试

例如测试某个搜索试验保存的权重：

```bash
python evaluate_hw1.py --checkpoint outputs/search/trial_03/best_weights.npz
```

### 4.3 如果没有对应 JSON 元信息

你也可以手动指定模型结构参数：

```bash
python evaluate_hw1.py --checkpoint outputs/final/best_model.npz --image-size 32 --train-ratio 0.7 --val-ratio 0.15 --seed 42 --hidden-dims 320x160 --activation relu
```

---

## 5. 训练结果

默认输出目录为 `outputs/`，其中重要文件包括：

### 5.1 最终模型

- `outputs/final/best_model.npz`
  - 最终采用的最佳模型权重

- `outputs/final/best_model.json`
  - 与该权重对应的元信息，包括最佳 epoch、验证集准确率、模型配置等


### 5.2 报告图表与分析结果

- `outputs/reports/training_curves.png`
- `outputs/reports/search_results.png`
- `outputs/reports/confusion_matrix.png`
- `outputs/reports/first_layer_weights.png`
- `outputs/reports/error_cases.png`
- ...

---

## 6. 项目主要文件说明

### 6.1 核心代码

- `hw1_mlp/autodiff.py`
  - 自定义 `Tensor`、自动微分与反向传播

- `hw1_mlp/data.py`
  - 数据读取、缩放、标准化、数据划分、批处理

- `hw1_mlp/model.py`
  - 三层 MLP 定义

- `hw1_mlp/losses.py`
  - 交叉熵损失与 L2 正则

- `hw1_mlp/optim.py`
  - SGD 与学习率衰减

- `hw1_mlp/trainer.py`
  - 单轮训练、验证评估、最优模型保存与加载

- `hw1_mlp/search.py`
  - 超参数随机搜索

- `hw1_mlp/reporting.py`
  - 训练曲线、混淆矩阵、权重可视化、错例分析

### 6.2 脚本入口

- `run_hw1.py`
  - 一键训练、搜索、评估、生成报告素材

- `evaluate_hw1.py`
  - 加载指定权重并在测试集上评估

---


## 7. 说明

1. 本项目默认在 CPU 上运行，不依赖 GPU。
2. 本项目没有使用任何现成自动微分深度学习框架。

---
