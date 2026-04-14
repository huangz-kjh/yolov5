# 使用YOLOv5原版的模型加载方式
import torch
from models.yolo import Model

def load_custom_model():
    """加载自定义DP模型"""
    try:
        # 加载模型配置
        model = Model("./models/yolov5s.yaml")
        
        # 打印模型信息
        print("模型加载成功！")
        print(f"模型结构: {model}")
        
        # 测试前向传播
        dummy_input = torch.randn(1, 3, 640, 640)
        with torch.no_grad():
            output = model(dummy_input)
        print(f"输出形状: {[x.shape for x in output]}")
        
        # 打印模型参数数量
        total_params = sum(p.numel() for p in model.parameters())
        print(f"总参数数量: {total_params:,}")
        
        return model
        
    except Exception as e:
        print(f"模型加载失败: {e}")
        return None

if __name__ == "__main__":
    model = load_custom_model()
# 如果需要加载预训练权重
# model.load_state_dict(torch.load('path_to_weights.pt')['model'].state_dict())