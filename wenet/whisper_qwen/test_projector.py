import sys
import os

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_dir))))
sys.path.insert(0, project_root)

import torch
from projector import TransformerProjector

def test_transformer_projector():
    # 测试参数
    batch_size = 2
    seq_len = 100
    idim = 2560  # whisper(1280) + qwen-audio(1280)
    odim = 3584  # qwen hidden size
    
    # 创建模型
    print("\n1. 初始化TransformerProjector...")
    projector = TransformerProjector(
        idim=idim,
        odim=odim,
        dropout_rate=0.1,
        num_layers=2,
        num_heads=8,
        intermediate_size=4096
    )
    print(projector)
    
    # 创建输入数据
    print("\n2. 创建测试数据...")
    x = torch.randn(batch_size, seq_len, idim)
    mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
    # 模拟部分序列的padding
    mask[:, 80:] = False
    
    print(f"输入形状: {x.shape}")
    print(f"mask形状: {mask.shape}")
    
    # 前向传播
    print("\n3. 执行前向传播...")
    try:
        with torch.no_grad():
            output, output_mask = projector(x, mask)
            
        print("\n4. 输出信息:")
        print(f"输出张量形状: {output.shape}")
        print(f"输出mask形状: {output_mask.shape}")
        
        # 检查输出维度
        assert output.shape == (batch_size, seq_len, odim), \
            f"输出形状不符合预期: {output.shape} != {(batch_size, seq_len, odim)}"
        
        # 检查数值范围
        print("\n5. 数值统计:")
        print(f"输出均值: {output.mean().item():.4f}")
        print(f"输出标准差: {output.std().item():.4f}")
        print(f"输出最小值: {output.min().item():.4f}")
        print(f"输出最大值: {output.max().item():.4f}")
        
        # 检查mask
        print("\n6. Mask统计:")
        print(f"有效token比例: {output_mask.float().mean().item():.2%}")
        
        print("\n测试成功完成!")
        return True
        
    except Exception as e:
        print(f"\n测试失败: {str(e)}")
        return False

if __name__ == "__main__":
    test_transformer_projector() 