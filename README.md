# 🚀 YOLOv5 Object Detection

> **Real-time Object Detection System based on YOLOv5**

---

## 📌 项目简介

本项目基于 YOLOv5 实现目标检测系统，支持图片、视频以及摄像头实时检测。

YOLOv5 是当前工业界广泛应用的目标检测模型之一，具备：

* ⚡ 高速度（Real-time）
* 🎯 高精度
* 🧩 易部署

---

## 🎯 项目目标

* ✅ 搭建完整目标检测流程（训练 + 推理）
* ✅ 支持自定义数据集训练
* ✅ 实现实时检测（Webcam）
* ✅ 掌握目标检测核心原理

---

## 🧠 模型原理（How YOLO Works）

YOLO（You Only Look Once）将目标检测转化为**回归问题**：

👉 一次前向传播同时输出：

* 目标位置（Bounding Box）
* 类别（Class）
* 置信度（Confidence）

---

## 🏗️ 模型结构（Architecture）

YOLOv5 主要由三部分组成：

### 🔹 1. Backbone（特征提取）

* CSPDarknet
* 提取图像特征

### 🔹 2. Neck（特征融合）

* PANet + FPN
* 多尺度特征融合

### 🔹 3. Head（检测头）

* 输出目标框 + 分类结果

---

## 💡 核心特点（Key Features）

* ⚡ 实时检测（FPS高）
* 📦 支持多目标检测
* 📊 自动Anchor机制
* 🧠 数据增强（Mosaic、MixUp）
* 🔄 多尺度训练

---

## 🖼️ 效果展示（Demo）

![val_batch2_labels](https://github.com/user-attachments/assets/10a2851f-611c-4fc6-9d46-4f9bc96afe33)
![val_batch2_pred](https://github.com/user-attachments/assets/bab70c9f-b3ba-4e13-b80b-537aee2eee74)


---

## 📊 实验结果（Results）

| 模型      | mAP  |
| ------- | ---- |
| YOLOv5s | 0.72 |

👉（你可以替换成你自己的结果）

---

## 📦 项目结构（Project Structure）

```bash
YOLOv5-Project/
├── data/            # 数据集配置
├── models/          # 模型结构
├── utils/           # 工具函数
├── runs/            # 训练结果
├── train.py         # 训练
├── detect.py        # 推理
├── val.py           # 验证
└── README.md
```

---

## ⚡ 快速开始（Quick Start）

### 1️⃣ 安装依赖

```bash
pip install -r requirements.txt
```

---

### 2️⃣ 训练模型

```bash
python train.py --img 640 --batch 16 --epochs 100 --data data.yaml --weights yolov5s.pt
```

---

### 3️⃣ 图片检测

```bash
python detect.py --weights runs/train/exp/weights/best.pt --source imgs/
```

---

### 4️⃣ 摄像头实时检测

```bash
python detect.py --weights best.pt --source 0
```

---

## 🧩 数据集格式（Dataset Format）

YOLO格式：

```text
image.jpg
image.txt
```

txt内容：

```text
class x_center y_center width height
```

（归一化坐标）

---

### ❓ Q1：YOLO和Faster R-CNN区别？

👉 A：

* YOLO：单阶段，速度快
* Faster R-CNN：两阶段，精度高

---

### ❓ Q2：YOLOv5为什么快？

👉 A：

* 单阶段检测
* 网络结构优化
* 多尺度预测

---

### ❓ Q3：mAP是什么？

👉 A：

* 平均精度（Mean Average Precision）
* 衡量检测模型性能

---

## 🔥 可扩展方向（Advanced）

* ✅ 加入目标跟踪（DeepSORT）
* ✅ 部署到 Web（Flask）
* ✅ ONNX / TensorRT 加速
* ✅ 训练自定义数据集

---

## 🌍 应用场景

* 🚗 自动驾驶
* 🏭 工业检测
* 🛍️ 零售分析
* 🎥 安防监控
