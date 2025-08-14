import matplotlib.pyplot as plt
import torch
import numpy as np

def plot_distributions(tensor_list:list, output_filename, 
                          labels=[],
                          colors=[],
                          title="", 
                          xlabel="Value distribution", 
                          ylabel="Density",
                          xlim=None,
                          ylim=None,
                          log_y=False,
                          bins=[],
                          alpha=0.6):
    """
    在同一图上绘制多个张量的分布
    
    参数:
        tensor_list: PyTorch张量列表
        output_filename: 输出文件名(不含.png)
        labels: 分布的图例标签
        colors: 分布的颜色
        title: 图表标题
        xlabel: x轴标签
        ylabel: y轴标签
        log_y: 是否对数坐标
        bins: 直方图分箱设置
        alpha: 透明度(0-1)
    """
    # 将张量转换为numpy数组并展平
    def process_tensor(t):
        if isinstance(t, torch.Tensor):
            if t.requires_grad:
                t = t.detach()
            return t.cpu().numpy().flatten()
        return np.array(t).flatten()
    
    data=[process_tensor(_tensor) for _tensor in tensor_list]
    
    # 创建图形
    plt.figure(figsize=(12, 7))

    # 绘制两个直方图
    for _data,_color,_lable,_bin in zip(data,colors,labels,bins):
        plt.hist(_data, bins=_bin, color=_color, edgecolor=_color,
                 alpha=alpha, label=_lable, density=True)

    
    # 坐标轴标签
    plt.tick_params(labelsize=14)
    
    # # 添加统计信息
    # stats_text = (f"{labels[0]}:\n"
    #              f"mean: {np.mean(data1):.2f}\n"
    #              f"std: {np.std(data1):.2f}\n\n"
    #              f"{labels[1]}:\n"
    #              f"mean: {np.mean(data2):.2f}\n"
    #              f"std: {np.std(data2):.2f}")
    
    # plt.text(0.95, 0.95, stats_text, transform=plt.gca().transAxes,
    #          fontsize=15, verticalalignment='top', horizontalalignment='right',
    #          bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 添加标题和标签
    plt.title(title, fontsize=25, pad=20)
    plt.xlabel(xlabel, fontsize=25)
    plt.ylabel(ylabel, fontsize=25)

    if xlim is not None: plt.xlim(*xlim)
    if ylim is not None: plt.ylim(*ylim)

    if log_y: plt.yscale('log')
    # 添加图例
    plt.legend(fontsize=25)
    
    # 添加网格线
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存为PNG图片
    plt.savefig(f"{output_filename}.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"对比分布图已保存为 {output_filename}.png")

def plot_tensor_distribution(tensor, output_filename,  bins='auto'):
    """
    绘制PyTorch张量的数值分布图并保存为PNG图片
    
    参数:
        tensor: 要绘制的PyTorch张量(任意形状)
        output_filename: 要保存的文件名(不需要.png后缀)
        title: 图表标题(可选)
        xlabel: x轴标签(可选)
        ylabel: y轴标签(可选)
        bins: 直方图分箱设置(可选)
    """
    # 将张量转换为numpy数组并展平(处理任意形状)
    if tensor.requires_grad:
        tensor = tensor.detach()
    data = tensor.cpu().numpy().flatten()
    
    # 创建图形
    plt.figure(figsize=(10, 6))
    
    # 绘制直方图
    plt.hist(data, bins=bins, color='skyblue', edgecolor='black', alpha=0.7)

    # 添加网格线
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存为PNG图片
    plt.savefig(f"{output_filename}.png", dpi=300, bbox_inches='tight')
    plt.close()
