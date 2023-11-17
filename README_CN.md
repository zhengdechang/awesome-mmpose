## Introduction

[English](./README.md) | 简体中文

---

本项目基于 [MMPOSE](https://github.com/open-mmlab/mmpose.git) 开发，更多的样例请参考 [MMPOSE demos](https://mmpose.readthedocs.io/en/latest/demos.html)。

## 安装步骤

以下是安装步骤，需要注意的是，这些步骤可能会因为你的环境而有所不同。

### Step 0: 安装 PyTorch

```bash
pip install torch==1.10.0+cu111 torchvision==0.11.1+cu111 torchaudio==0.10.0+cu111 -f https://download.pytorch.org/whl/cu111/torch_stable.html 
```

### Step 1: 创建和激活conda环境

```bash
conda create --name openmmlab python=3.8 -y
conda activate openmmlab
```

### Step 2: 安装 OpenMIM

```bash
pip install -U openmim
```

### Step 3: 安装 MMCV 和 MMDetection

```bash
mim install mmengine
mim install "mmcv>=2.0.1"
mim install "mmdet>=3.1.0" 
```

### Step 4: 安装项目依赖

```bash
pip install -r requirements.txt
```

### Step 5: 安装项目

```bash
pip install -v -e .  
```

### Step 6: 安装 MMPOSE

```bash
mim install "mmpose>=1.1.0"
```

## 测试

以下是一个测试命令，它会将原图（`demo/test.jpg`）和效果图（`vis_results/test.jpg`）进行对比。

```bash
python demo/topdown_demo_with_mmdet.py \
    demo/mmdetection_cfg/rtmdet_m_640-8xb32_coco-person.py \
    https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth \
    configs/wholebody_2d_keypoint/topdown_heatmap/coco-wholebody/td-hm_hrnet-w48_dark-8xb32-210e_coco-wholebody-384x288.py \
    https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_wholebody_384x288_dark-f5726563_20200918.pth \
    --input demo/test.jpg \
    --output-root vis_results/ --save-predictions --black-background
```

## 结果展示

在运行测试命令后，你可以在 `vis_results/` 目录下找到结果图像。

原图：

![原图](demo/test.jpg)

效果图：

![效果图](vis_results/test.jpg)

## 贡献

如果你在使用过程中发现任何问题，或者有任何建议，欢迎提交 Issue 或者 Pull Request。