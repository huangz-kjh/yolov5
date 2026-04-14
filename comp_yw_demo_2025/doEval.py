# !/usr/bin/env python3 
# -*- coding: utf-8 -*-
"""
Copyright 2024 Baidu Inc. All Rights Reserved.
2024/1/10, by zhangyi82@baidu.com, create

DESCRIPTION
【选手】本地测试脚本
"""
import localEvaluate
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--data-info', type=str, help='Path to dataset json', required=True)
    parser.add_argument('--data-info', type=str, default="dataset.json", help='Path to dataset json', required=False)
    parser.add_argument('--draw-bbox', type=str, default=False, help='Draw Gt and Pred Bbox on Images', required=False)
    args = parser.parse_args()
    # data_info = "dataset.json"
    # 初始化模型评估组件
    print(args.data_info)
    evaluator = localEvaluate.LYEval(args)
    evaluator.pipeline()