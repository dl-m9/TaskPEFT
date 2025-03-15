import torch
import numpy as np
import torch.nn as nn

# 存储结果的字典
magnitude_products = {}

def forward_hook(module, input, output):
    """前向传播钩子函数"""
    # 获取模块参数的幅度
    param_magnitude = {}
    for name, param in module.named_parameters(recurse=False):
        if param.requires_grad:
            # 保存每个参数的幅度，保持原始形状
            param_magnitude[name] = torch.abs(param.data)
    
    # 获取输入的幅度，保持原始形状
    input_magnitude = torch.abs(input[0])
    
    # 计算乘积并存储，为每个参数创建对应的敏感度掩码
    for name, param_mag in param_magnitude.items():
        # 确保输入幅度的维度与参数匹配
        if "conv" in name or "weight" in name and len(param_mag.shape) == 4:
            # 对于卷积层，需要调整输入维度以匹配参数维度
            # 假设参数形状为 [out_channels, in_channels, kernel_h, kernel_w]
            # 输入形状为 [batch, in_channels, height, width]
            B, C, H, W = input_magnitude.shape
            # 扩展输入以匹配参数维度
            expanded_input = input_magnitude.mean(dim=(0, 2, 3)).view(1, -1, 1, 1)
            expanded_input = expanded_input.expand(param_mag.shape[0], -1, param_mag.shape[2], param_mag.shape[3])
            product = param_mag * expanded_input
        else:
            # 对于线性层等，直接计算乘积
            if len(param_mag.shape) == 2 and len(input_magnitude.shape) > 2:
                # 如果是线性层且输入是多维的，将输入平均到适当的维度
                input_flat = input_magnitude.mean(dim=tuple(range(len(input_magnitude.shape)-1)))
                product = param_mag * input_flat.view(-1, 1)
            else:
                # 其他情况，尝试直接计算乘积
                try:
                    product = param_mag * input_magnitude.mean(dim=0)
                except:
                    # 如果维度不匹配，使用标量乘法
                    product = param_mag * torch.norm(input_magnitude).item()
        
        # 存储结果
        key = module._get_name() + '_' + str(id(module)) + '_' + name
        magnitude_products[key] = {
            'param_name': name,
            'module_name': module._get_name(),
            'product': product,
            'param_shape': param_mag.shape
        }

# 注册钩子到模型的每一层
def register_magnitude_hooks(model):
    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear, nn.BatchNorm2d)):  # 可以根据需要添加更多层类型
            hook = module.register_forward_hook(forward_hook)
            hooks.append(hook)
    return hooks

# 使用示例
def measure_magnitudes(model, sample_input):
    # 注册钩子
    hooks = register_magnitude_hooks(model)
    
    # 执行前向传播
    model.eval()  # 设置为评估模式
    with torch.no_grad():
        _ = model(sample_input)
    
    # 移除钩子
    for hook in hooks:
        hook.remove()
    
    return magnitude_products

def create_sensitive_mask(model, dataloader, percentile=90):
    """
    通过数据加载器中的样本创建敏感度掩码
    
    Args:
        model: 要分析的模型
        dataloader: 包含样本数据的数据加载器
        percentile: 用于确定掩码的百分位数阈值
    
    Returns:
        sensitive_mask: 包含每层敏感度的字典，格式与pruning.py中的new_masks兼容
    """
    global magnitude_products
    magnitude_products = {}  # 重置字典
    
    model.eval()  # 设置为评估模式
    hooks = register_magnitude_hooks(model)
    
    # 使用数据加载器中的多个样本来获取更稳定的结果
    with torch.no_grad():
        for batch_idx, (data, _) in enumerate(dataloader):
            if isinstance(data, list):
                data = data[0]  # 如果数据是列表，取第一个元素
            
            if torch.cuda.is_available():
                data = data.cuda()
                
            _ = model(data)
            
            # 只使用少量批次以提高效率
            if batch_idx >= 5:  # 可以根据需要调整批次数量
                break
    
    # 移除钩子
    for hook in hooks:
        hook.remove()
    
    # 创建敏感度掩码，格式与pruning.py中的new_masks兼容
    sensitive_mask = {}
    param_to_sensitivity = {}
    
    # 将敏感度信息映射到模型参数名称
    for key, value in magnitude_products.items():
        param_name = value['param_name']
        module_name = value['module_name']
        
        # 查找完整的参数名称
        for name, _ in model.named_parameters():
            if param_name in name and module_name.lower() in name.lower():
                param_to_sensitivity[name] = value['product']
                break
    
    # 为每个参数创建掩码
    for name, param in model.named_parameters():
        if name in param_to_sensitivity:
            # 获取敏感度
            sensitivity = param_to_sensitivity[name]
            
            # 创建掩码：保留敏感度高的参数，剪枝敏感度低的参数
            if "norm" in name or "pos_embed" in name or "cls_token" in name:
                # 如果是norm、pos_embed或cls_token，不剪枝
                mask = torch.ones_like(param.data)
            elif 'head' in name or "bias" in name or "gamma" in name:
                # 如果是head、bias或gamma，全部剪枝
                mask = torch.zeros_like(param.data)
            else:
                # 根据敏感度创建掩码
                threshold = torch.quantile(sensitivity.flatten(), 1 - percentile/100)
                mask = (sensitivity >= threshold).float()
            
            sensitive_mask[name] = mask.cuda() if torch.cuda.is_available() else mask
    
    return sensitive_mask