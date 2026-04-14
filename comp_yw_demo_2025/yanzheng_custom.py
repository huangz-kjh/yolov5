import sys
import os
from predict import Predictor
import cv2
import numpy as np

def test_custom_model():
    """测试自定义火灾烟雾检测模型"""
    print("=== 自定义火灾烟雾检测模型验证 ===")
    
    # 初始化预测器
    predictor = Predictor()
    
    # 初始化模型
    print("正在加载自定义模型...")
    ret, err_message = predictor.InitModel()
    
    if not ret:
        print(f"❌ 模型加载失败: {err_message}")
        return False
    
    print("✅ 模型加载成功")
    print(f"模型类别: {predictor.names}")
    
    # 测试单张图片
    test_image = "dataset/input_data/A/images/hy_test_A_0011.jpg"
    
    if not os.path.exists(test_image):
        print(f"❌ 测试图片不存在: {test_image}")
        return False
    
    print(f"正在检测图片: {test_image}")
    
    # 读取图片
    img = cv2.imread(test_image)
    if img is None:
        print("❌ 无法读取图片")
        return False
    
    print(f"图片尺寸: {img.shape}")
    
    # 进行检测
    try:
        # 使用predict.py中的检测方法
        input_dir = "dataset/input_data/A/images"
        
        # 执行检测 (Detect方法只接受一个参数)
        detect_result, detect_err_message = predictor.Detect(input_dir)
        
        if detect_err_message:
            print(f"❌ 检测过程出错: {detect_err_message}")
            return False
        
        print(f"检测结果保存在: {detect_result}")
        
        # 检查结果文件
        result_file = os.path.join(detect_result, "hy_test_A_0011.txt")
        if os.path.exists(result_file):
            with open(result_file, 'r') as f:
                detections = f.readlines()
            
            print(f"✅ 检测完成，发现 {len(detections)} 个目标")
            
            if detections:
                print("检测结果:")
                for i, detection in enumerate(detections):
                    parts = detection.strip().split()
                    if len(parts) >= 6:
                        class_id = int(parts[0])
                        x1, y1, x2, y2 = map(float, parts[1:5])
                        conf = float(parts[5])
                        class_name = predictor.names[class_id] if predictor.names else f"class_{class_id}"
                        print(f"  目标 {i+1}: {class_name} (置信度: {conf:.3f}) 位置: ({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f})")
            else:
                print("未检测到火灾或烟雾")
        else:
            print("❌ 结果文件未生成")
            return False
            
    except Exception as e:
        print(f"❌ 检测过程出错: {e}")
        return False
    
    return True

def compare_models():
    """比较通用模型和自定义模型的性能"""
    print("\n=== 模型性能对比 ===")
    
    print("通用YOLOv5模型:")
    print("  - 类别数: 80 (COCO数据集)")
    print("  - 适用场景: 通用目标检测")
    print("  - 对火灾烟雾检测: 可能无法识别或误识别")
    
    print("\n自定义火灾烟雾检测模型:")
    print("  - 类别数: 2 (fire, smoke)")
    print("  - 适用场景: 专门的火灾烟雾检测")
    print("  - 性能指标:")
    print("    * 整体 mAP50: 0.742")
    print("    * 火灾 mAP50: 0.706") 
    print("    * 烟雾 mAP50: 0.778")
    print("  - 推荐用于生产环境的火灾烟雾检测任务")

def test_ultralytics_model():
    """测试Ultralytics YOLOv5模型"""
    print("\n=== Ultralytics YOLOv5模型测试 ===")
    
    try:
        from ultralytics import YOLO
        
        # 测试改进的yolov5su模型
        print("正在加载 YOLOv5su 模型...")
        model = YOLO('yolov5su.pt')
        
        # 显示模型信息
        model.info()
        
        # 测试图片
        test_image = "dataset/input_data/A/images/hy_test_A_0011.jpg"
        
        if os.path.exists(test_image):
            print(f"正在使用 YOLOv5su 检测图片: {test_image}")
            
            # 进行推理
            results = model(test_image, save=True, conf=0.25, iou=0.45)
            
            # 打印结果
            for r in results:
                print(f"检测到 {len(r.boxes)} 个目标")
                if len(r.boxes) > 0:
                    for i, box in enumerate(r.boxes):
                        cls = int(box.cls)
                        conf = float(box.conf)
                        xyxy = box.xyxy[0].tolist()
                        class_name = model.names[cls]
                        print(f"目标 {i+1}: {class_name} (置信度: {conf:.3f}) 坐标: {xyxy}")
                print(f"结果已保存到: {r.save_dir}")
        else:
            print(f"测试图片不存在: {test_image}")
            
    except ImportError:
        print("❌ Ultralytics 库未安装")
    except Exception as e:
        print(f"❌ Ultralytics 模型测试失败: {e}")

if __name__ == "__main__":
    # 测试自定义模型
    success = test_custom_model()
    
    # 显示模型对比
    compare_models()
    
    # 测试Ultralytics模型
    test_ultralytics_model()
    
    if success:
        print("\n✅ 验证完成！自定义模型工作正常")
    else:
        print("\n❌ 验证失败，请检查模型和数据")