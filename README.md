---
title: MedicalSeg
emoji: 🏥
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
short_description: Medical image lesion segmentation with U-Net (PyTorch)
---

# 🏥 医疗影像病灶分割系统

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange)
![License](https://img.shields.io/badge/License-MIT-green)

基于 PyTorch 实现的完整医疗影像病灶分割系统，支持 U-Net 和 UNet++ 两种经典分割网络，内置 Gradio 交互界面，开箱即用。适合 AI 学习、计算机视觉毕设项目。

---

## ✨ 功能特性

| 模块 | 功能 |
|------|------|
| **数据预处理** | 支持 DICOM、NIfTI、JPG/PNG 格式；CLAHE 灰度矫正；高斯去噪；归一化；数据增强（翻转/旋转）；自动数据集划分 7:2:1 |
| **模型架构** | U-Net（经典编码器-解码器）、UNet++（嵌套跳跃连接）；一键切换；支持迁移学习 |
| **损失函数** | Dice Loss + Focal Loss 混合损失，解决医疗影像正负样本不均衡 |
| **训练优化** | AdamW 优化器；Cosine 学习率动态衰减；Early Stopping 早停防过拟合 |
| **评估指标** | Dice 系数、IoU 交并比、Precision、Recall、Accuracy |
| **可视化** | 原始影像 / 病灶掩码 / 分割叠加 三图对比；自动绘制训练损失/Dice 曲线 |
| **部署兼容** | GPU/CPU 自动检测；Windows/Linux/macOS 跨平台；FastAPI 接口预留 |

---

## 🚀 在线演示

直接在本页面上传医疗影像（灰度图），点击「开始分割」即可体验！

> ℹ️ **当前演示模式说明**：由于 HF Spaces 存储限制，在线版本使用传统 CV 方法（Otsu 阈值 + 形态学操作）演示界面功能。如需深度学习模型的精准分割效果，请下载代码到本地训练。

---

## 🛠️ 本地使用

### 1. 安装依赖

```bash
git clone https://github.com/changxuelee-gif/medical_seg.git
cd medical_seg
pip install -r requirements.txt
```

### 2. 快速体验（无需准备数据）

```bash
# 生成100张模拟医疗影像数据
python tools/generate_sim_data.py

# 快速训练演示（10轮，CPU可运行）
python tools/train.py --config configs/demo.yaml

# 启动Web界面
python app.py
```

浏览器访问 http://localhost:7860 即可使用。

### 3. 训练自定义数据集

1. 将你的数据组织为以下结构：
```
data/your_dataset/
├── images/          # 原始影像 (.png/.jpg/.dcm/.nii)
│   ├── 001.png
│   ├── 002.png
│   └── ...
└── masks/           # 标注掩码（与images同名对应）
    ├── 001.png
    ├── 002.png
    └── ...
```

2. 修改配置文件 `configs/default.yaml` 中的 `data_dir` 指向你的数据路径

3. 开始训练：
```bash
python tools/train.py --config configs/default.yaml
```

---

## 📁 项目结构

```
medical_seg/
├── app.py                     # Gradio Web界面（本文件启动入口）
├── api.py                     # FastAPI推理接口预留
├── requirements.txt           # 依赖清单
├── configs/                   # 配置文件
│   ├── default.yaml           # 默认训练配置（50轮）
│   └── demo.yaml              # 快速演示配置（10轮）
├── medicalseg/                # 核心代码包
│   ├── io/                    # 多格式影像读取（DICOM/NIfTI/JPG/PNG）
│   ├── datasets/              # 数据预处理、数据集类、数据增强
│   ├── models/                # U-Net、UNet++模型实现
│   ├── training/              # 损失函数、评估指标、优化器、早停、训练器
│   ├── inference/             # 推理预测器
│   ├── visualization/         # 可视化（训练曲线、分割结果展示）
│   └── utils/                 # 配置加载、设备检测、日志、随机种子
└── tools/                     # 命令行脚本
    ├── generate_sim_data.py   # 生成模拟数据
    ├── train.py               # 训练入口
    ├── predict.py             # 批量推理
    └── plot_curves.py         # 绘制训练指标曲线
```

---

## 📊 评估指标说明

| 指标 | 含义 | 医疗分割意义 |
|------|------|-------------|
| **Dice** | 相似度系数 | 衡量预测掩码与真实掩码的重叠程度，最重要指标 |
| **IoU** | 交并比 (Jaccard) | 交集除以并集，与Dice正相关 |
| **Precision** | 精确率 | 预测为病灶的像素中真正是病灶的比例 |
| **Recall** | 召回率 | 真实病灶像素中被正确检出的比例（不漏诊） |
| **Accuracy** | 准确率 | 整体像素分类正确率（类别不平衡时参考价值有限） |

---

## 📝 开源协议

本项目采用 [MIT License](https://opensource.org/licenses/MIT) 开源协议，可自由用于学习、研究和商业项目。

---

## ⭐ 相关链接

- **GitHub 仓库**: [changxuelee-gif/medical_seg](https://github.com/changxuelee-gif/medical_seg)

如果这个项目对你有帮助，欢迎在 GitHub 上给个 Star ⭐！
