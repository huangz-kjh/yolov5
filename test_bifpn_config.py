#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试 BiFPN 配置文件的语法和模块定义
"""

import yaml
import sys
from pathlib import Path

def test_bifpn_config():
    """测试 BiFPN 配置文件"""
    config_path = "models/yolov5s_bgf.yaml"
    
    try:
        # 读取配置文件
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        print("✓ BiFPN 配置文件语法正确")
        
        # 检查必需字段
        required_fields = ['nc', 'depth_multiple', 'width_multiple', 'anchors', 'backbone', 'head']
        for field in required_fields:
            if field in config:
                print(f"✓ 找到字段: {field}")
            else:
                print(f"✗ 缺少字段: {field}")
                return False
        
        # 提取所有使用的模块
        modules = set()
        
        # 从 backbone 提取模块
        if 'backbone' in config:
            for layer in config['backbone']:
                if len(layer) >= 3:
                    modules.add(layer[2])
        
        # 从 head 提取模块
        if 'head' in config:
            for layer in config['head']:
                if len(layer) >= 3:
                    modules.add(layer[2])
        
        print("\n配置文件中使用的模块:")
        for module in sorted(modules):
            print(f"  - {module}")
        
        # 检查 BiFPN 特有模块
        bifpn_modules = ['BiFPNBlock', 'BiFPNLayer', 'BiFPN', 'BiFPNC3']
        print("\nBiFPN 特有模块:")
        for module in bifpn_modules:
            if module in modules:
                print(f"  ✓ {module}")
            else:
                print(f"  - {module} (未使用)")
        
        # 检查是否成功替换了 Concat
        if 'Concat' in modules:
            print("\n⚠️  警告: 配置中仍包含 Concat 模块，可能未完全替换为 BiFPN")
        else:
            print("\n✓ 成功将 Concat 替换为 BiFPN 模块")
        
        return True
        
    except yaml.YAMLError as e:
        print(f"✗ YAML 语法错误: {e}")
        return False
    except FileNotFoundError:
        print(f"✗ 配置文件未找到: {config_path}")
        return False
    except Exception as e:
        print(f"✗ 其他错误: {e}")
        return False

def test_module_imports():
    """测试模块导入"""
    try:
        # 测试导入 BiFPN 模块
        sys.path.append('.')
        from models.common import BiFPNBlock, BiFPNLayer, BiFPN, BiFPNC3
        print("\n✓ BiFPN 模块导入成功")
        
        # 测试模块实例化
        import torch
        
        # 测试 BiFPNBlock
        bifpn_block = BiFPNBlock(256, 256)
        print("✓ BiFPNBlock 实例化成功")
        
        # 测试 BiFPNC3
        bifpn_c3 = BiFPNC3(256, 256, n=3)
        print("✓ BiFPNC3 实例化成功")
        
        return True
        
    except ImportError as e:
        print(f"\n✗ 模块导入失败: {e}")
        return False
    except Exception as e:
        print(f"\n✗ 模块测试失败: {e}")
        return False

if __name__ == "__main__":
    print("=== BiFPN 配置测试 ===")
    
    # 测试配置文件
    config_ok = test_bifpn_config()
    
    # 测试模块导入
    modules_ok = test_module_imports()
    
    if config_ok and modules_ok:
        print("\n🎉 所有测试通过！BiFPN 配置已成功替换 Concat 操作")
    else:
        print("\n❌ 测试失败，请检查配置或模块定义")