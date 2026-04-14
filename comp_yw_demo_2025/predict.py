# !/usr/bin/env python3 
# -*- coding: utf-8 -*-
"""
Copyright 2024 Baidu Inc. All Rights Reserved.
2024/1/8, by zhangyi82@baidu.com, create

DESCRIPTION
【选手编写】预测脚本 - 专门检测烟雾，过滤火焰标签
"""
import os
import sys
import time
import logging
import cv2
import torch
import numpy as np
import pandas as pd
from PIL import Image

# 添加YOLOv5相关导入
from models.common import DetectMultiBackend
from utils.general import (check_img_size, non_max_suppression, scale_boxes, 
                          xyxy2xywh, xywh2xyxy, strip_optimizer, colorstr)
from utils.torch_utils import select_device, time_sync
from utils.augmentations import letterbox

# 系统默认配置
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
#此处设置您的队伍名称
os.environ['MATCH_TEAM_NAME'] = "LYGongcheng"

class Predictor(object):
    """
    InitModel函数  模型初始化参数，注意不能自行增加删除函数入参
    ret            是否正常: 正常True,异常False
    err_message    错误信息: 默认normal
    return ret,err_message
    备注说明:比赛使用上传模型的方式，模型路径等请使用相对路径
    """

    def __init__(self):
        # 模型参数
        self.model = None
        self.device = None
        self.stride = None
        self.names = None
        self.imgsz = 640
        self.conf_thres = 0.25
        self.iou_thres = 0.45
        self.max_det = 300
        # 类别过滤设置：只保留smoke类别（class_id=1），过滤fire类别（class_id=0）
        self.filter_classes = [1]  # 只保留smoke类别
        logging.info("初始化预测器 - 配置为只检测烟雾，过滤火焰")
    
    def InitModel(self):
        ret = True
        err_message = None
        '''
        模型初始化,由用户自行编写
        加载出错时,给ret和err_message赋值相应的错误
        例如
        ret=False
        err_message = "[Error] model_path: [{}] init failed".format(model_path)
        '''
        try:
            # 设备选择
            self.device = select_device('')  # 自动选择最佳设备
            logging.info(f'Using device: {self.device}')
            
            # 模型路径 - 使用相对路径
            model_path = "model/best_v8.onnx"
            
            # 检查模型文件是否存在
            if not os.path.exists(model_path):
                ret = False
                err_message = f"[Error] model_path: [{model_path}] not found"
                return ret, err_message
            
            # 加载模型
            self.model = DetectMultiBackend(model_path, device=self.device, dnn=False, fp16=False)
            self.stride = self.model.stride
            self.names = self.model.names
            
            # 检查图像尺寸
            self.imgsz = check_img_size(self.imgsz, s=self.stride)
            
            # 模型预热
            self.model.warmup(imgsz=(1, 3, self.imgsz, self.imgsz))
            
            logging.info(f"Model loaded successfully from {model_path}")
            logging.info(f"Model names: {self.names}")
            logging.info(f"Filter settings: Only detect classes {self.filter_classes} (smoke only)")
            
        except Exception as e:
            ret, err_message = False, f"loading model error: {str(e)}"
            logging.error(f"Model initialization failed: {str(e)}")
            
        return ret, err_message
    
    @torch.no_grad()
    def Detect(self, input_data_file):
        """
        模型预测函数，注意调用该函数时会进行计时，后续将用来计算到模型性能得分
        input_data_file: 输入数据的绝/相对路径，举例如下:
        dataset.json的data_path的字段值 + "/input_data/images/"
        选手本地测试时，也需要按照上面的目录结构进行存放
        
        return:(可参考下面的示例代码)
        detect_result: path   为推理结果的存放路径，是选手自己定义的临时目录（多为程序运行时自动生成）
        推理结果为txt文件，文件前缀为推理的图片名称前缀
        txt文件中的内容为:
        1 595 554 1009 735 0.78 
        第1位'1'为检测的类别代码，只输出smoke类别(1)，过滤fire类别(0)
        第2-5为检测目标的坐标框，VOC格式为Xmin,Ymin,Xmax,Ymax
        第6位检测目标的置信度

        err_message: 模型预测错误信息
        """
        err_message = None
        detect_result = None
        
        try:
            # 检查模型是否已初始化
            if self.model is None:
                err_message = "Model not initialized"
                return detect_result, err_message
            
            # 检查输入目录是否存在
            if not os.path.exists(input_data_file):
                err_message = f"Input directory not found: {input_data_file}"
                return detect_result, err_message
            
            # 生成临时结果存放文件夹
            if "A" in os.path.dirname(input_data_file):
                txtpath = './dataset/pre_res/A'
            else:
                txtpath = './dataset/pre_res/B'
            
            if not os.path.exists(txtpath):
                os.makedirs(txtpath)
            detect_result = txtpath
            
            # 获取所有图像文件
            image_files = [f for f in os.listdir(input_data_file) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'))]
            
            if not image_files:
                err_message = "No image files found in input directory"
                return detect_result, err_message
            
            logging.info(f"Found {len(image_files)} images to process")
            logging.info(f"Filter mode: Only detecting smoke (class_id=1), filtering fire (class_id=0)")
            
            # 处理每张图像
            for file in image_files:
                img_path = os.path.join(input_data_file, file)
                base_name = os.path.splitext(file)[0]  # 获取文件名前缀
                txt_path = f"{base_name}.txt"  # 生成txt文件名
                txtfilepath = os.path.join(txtpath, txt_path)
                
                try:
                    # 读取图像
                    im0 = cv2.imread(img_path)  # BGR
                    if im0 is None:
                        logging.warning(f"Failed to load image: {img_path}")
                        continue
                    
                    # 图像预处理
                    im = letterbox(im0, self.imgsz, stride=self.stride, auto=False)[0]  # padded resize
                    im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
                    im = np.ascontiguousarray(im)  # contiguous
                    
                    # 转换为tensor
                    im = torch.from_numpy(im).to(self.device)
                    im = im.half() if self.model.fp16 else im.float()  # uint8 to fp16/32
                    im /= 255  # 0 - 255 to 0.0 - 1.0
                    if len(im.shape) == 3:
                        im = im[None]  # expand for batch dim
                    
                    # 推理
                    pred = self.model(im, augment=False, visualize=False)
                    
                    # NMS后处理
                    pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, 
                                             classes=None, agnostic=False, max_det=self.max_det)
                    
                    # 处理检测结果
                    det = pred[0]  # 第一张图像的检测结果
                    
                    # 清空或创建txt文件
                    with open(txtfilepath, 'w') as f:
                        if len(det):
                            # 将坐标从模型输入尺寸缩放回原始图像尺寸
                            det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
                            
                            # 过滤检测结果：只保留smoke类别
                            filtered_detections = []
                            for *xyxy, conf, cls in reversed(det):
                                class_id = int(cls)
                                # 只保留smoke类别（class_id=1），过滤fire类别（class_id=0）
                                if class_id in self.filter_classes:
                                    filtered_detections.append((*xyxy, conf, cls))
                            
                            # 写入过滤后的检测结果
                            for *xyxy, conf, cls in filtered_detections:
                                # 转换为VOC格式 (xmin, ymin, xmax, ymax)
                                x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                                class_id = 0
                                confidence = float(conf)
                                
                                # 写入txt文件: class_id x1 y1 x2 y2 confidence
                                f.write(f"{class_id} {x1} {y1} {x2} {y2} {confidence:.4f}\n")
                        
                        # 如果没有检测到目标或所有检测都被过滤，文件保持为空
                    
                    logging.debug(f"Processed {file}")
                    
                except Exception as e:
                    logging.error(f"Error processing {file}: {str(e)}")
                    # 创建空的txt文件
                    with open(txtfilepath, 'w') as f:
                        pass
                    continue
            
            logging.info(f"Detection completed. Results saved to {detect_result}")
            
        except Exception as e:
            err_message = f"Detection error: {str(e)}"
            logging.error(f"Detection failed: {str(e)}")
        
        return detect_result, err_message

    def data_preprocess(self, input_data_file):
        """
        数据预处理
        param input_data_file: 输入数据的路径
        return df: pandas.DataFrame格式，预处理好的数据
               err_message: 处理中途报错信息，如果没有就是None
        """
        err_message = None
        logging.info(f"Predictor.Detect函数的input_data_file is: {input_data_file}")
        df = None
        return df, err_message


if __name__ == '__main__':

    # 备注说明:main函数提供给用户内测,修改后[不影响]评估
    predictor = Predictor()
    # 初始化模型
    ret, err_message = predictor.InitModel()

    # 模型预测
    if ret:
        detect_result, detect_err_message = predictor.Detect("./dataset/input_data/A/images")
        if detect_err_message is None:
            logging.info(f"Detect_result:\n{detect_result}")
        else:
            logging.error(f"[Error] Detect failed. {detect_err_message}")
    else:
        logging.error(f"[Error] InitModel failed. ret is {ret}, err_message is {err_message}")
