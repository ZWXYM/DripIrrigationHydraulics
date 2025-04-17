"""
极简版修复CSV读取问题的程序 - 只修改必要部分
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from scipy.spatial.distance import cdist, pdist
from matplotlib import rcParams
import platform

# 创建结果目录
RESULTS_DIR = 'pareto_comparison_results'
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

# ====================== 帕累托解集读取函数 ======================

def load_latest_pareto_front(algorithm_name, results_dir=RESULTS_DIR):
    """
    读取最新的帕累托前沿CSV文件

    参数:
    algorithm_name: 算法名称
    results_dir: 结果目录

    返回:
    帕累托前沿解集，形式为numpy数组
    """
    # 获取所有匹配的文件
    files = [f for f in os.listdir(results_dir) if f.startswith(f"{algorithm_name}_pareto_front_") and f.endswith(".csv")]

    if not files:
        print(f"未找到{algorithm_name}的帕累托前沿文件")
        return None

    # 按文件修改时间排序
    files.sort(key=lambda x: os.path.getmtime(os.path.join(results_dir, x)), reverse=True)

    # 读取最新的文件
    latest_file = os.path.join(results_dir, files[0])

    try:
        # 先尝试简单直接的方式读取
        print(f"读取文件: {latest_file}")
        data = np.loadtxt(latest_file, delimiter=',', skiprows=1)
        print(f"文件读取成功，形状: {data.shape}")
        return data
    except Exception as e:
        print(f"使用numpy读取失败: {str(e)}")

        try:
            # 再尝试pandas读取
            df = pd.read_csv(latest_file)
            print(f"pandas读取成功，列名: {df.columns.tolist()}")
            return df.values
        except Exception as e:
            print(f"pandas读取失败: {str(e)}")

            # 最后尝试手动读取
            try:
                print("尝试手动读取...")
                data = []
                with open(latest_file, 'r') as f:
                    lines = f.readlines()
                    # 跳过第一行（标题行）
                    for line in lines[1:]:
                        parts = line.strip().split(',')
                        if len(parts) >= 2:
                            data.append([float(parts[0]), float(parts[1])])

                if data:
                    result = np.array(data)
                    print(f"手动读取成功，形状: {result.shape}")
                    return result
            except Exception as e:
                print(f"所有读取方式都失败: {str(e)}")
                return None

def load_all_pareto_fronts(algorithm_names, results_dir=RESULTS_DIR):
    """
    读取多个算法的最新帕累托前沿

    参数:
    algorithm_names: 算法名称列表
    results_dir: 结果目录

    返回:
    字典，键为算法名称，值为帕累托前沿
    """
    fronts = {}
    for alg_name in algorithm_names:
        front = load_latest_pareto_front(alg_name, results_dir)
        if front is not None:
            fronts[alg_name] = front

    if not fronts:
        print("未找到帕累托前沿数据，生成模拟数据用于演示...")
        # 生成模拟数据
        fronts = {
            "OriginalPSO": np.array([
                [285000, 0.45],
                [305000, 0.40],
                [325000, 0.35],
                [355000, 0.28],
                [375000, 0.25],
                [405000, 0.22],
                [435000, 0.19],
                [465000, 0.17],
                [500000, 0.15]
            ]),
            "EnhancedMOPSO": np.array([
                [260000, 0.32],
                [280000, 0.28],
                [310000, 0.22],
                [330000, 0.18],
                [350000, 0.15],
                [380000, 0.12],
                [450000, 0.08]
            ]),
            "MOPSO": np.array([
                [300000, 0.18],
                [320000, 0.14],
                [340000, 0.11],
                [370000, 0.09],
                [410000, 0.07],
                [470000, 0.05],
                [520000, 0.04]
            ])
        }

    return fronts

# ====================== 性能指标计算函数 ======================

def calculate_sp(front):
    """
    计算Pareto前沿的均匀性指标SP
    值越小表示分布越均匀
    """
    if len(front) < 2:
        return 0

    # 计算每对解之间的欧几里得距离
    distances = pdist(front, 'euclidean')

    # 计算平均距离
    d_mean = np.mean(distances)

    # 计算标准差
    sp = np.sqrt(np.sum((distances - d_mean) ** 2) / (len(distances) - 1))

    return sp

def calculate_igd(approximation_front, true_front):
    """
    计算反向代际距离(IGD)
    从真实Pareto前沿到近似前沿的平均距离
    值越小表示质量越高
    """
    if len(approximation_front) == 0 or len(true_front) == 0:
        return float('inf')

    # 计算每个点到前沿的最小距离
    distances = cdist(true_front, approximation_front, 'euclidean')
    min_distances = np.min(distances, axis=1)

    # 返回平均距离
    return np.mean(min_distances)

def calculate_hypervolume(front, reference_point):
    """
    计算超体积指标(HV)
    前沿与参考点构成的超体积
    值越大表示质量越高
    注意：这是一个简化版本，只适用于二维问题
    """
    # 对于高维问题应使用专业库如pygmo或pymoo
    if len(front) == 0:
        return 0

    # 确保前沿是按照第一个目标升序排序的
    front_sorted = front[front[:, 0].argsort()]

    # 计算超体积（二维情况下是面积）
    hypervolume = 0
    for i in range(len(front_sorted)):
        if i == 0:
            # 第一个点
            height = reference_point[1] - front_sorted[i, 1]
            width = front_sorted[i, 0] - reference_point[0]
        else:
            # 其他点
            height = reference_point[1] - front_sorted[i, 1]
            width = front_sorted[i, 0] - front_sorted[i - 1, 0]

        # 只累加正面积
        area = height * width
        if area > 0:
            hypervolume += area

    # 确保返回非负值
    return max(0, hypervolume)

def calculate_all_metrics(fronts, ref_point=None):
    """
    计算多个帕累托前沿的所有性能指标

    参数:
    fronts: 字典，键为算法名称，值为帕累托前沿

    返回:
    字典，包含各算法的性能指标
    """
    metrics = {}

    # 确定参考点
    if ref_point is None:
        # 找到所有前沿中的最大值作为参考点
        max_cost = max([np.max(front[:, 0]) for front in fronts.values()])
        max_var = max([np.max(front[:, 1]) for front in fronts.values()])
        # 参考点为(0, 0)，因为是最小化问题
        ref_point = np.array([0, 0])

    # 计算每个前沿的指标
    for alg_name, front in fronts.items():
        metrics[alg_name] = {
            'sp': calculate_sp(front),
            'hv': calculate_hypervolume(front, ref_point)
        }

        # IGD需要一个参考前沿，这里我们可以将所有前沿合并作为参考
        # 或者使用其中一个作为参考
        # 这里简单示例，仅计算与其他算法前沿的IGD
        metrics[alg_name]['igd'] = {}
        for other_alg, other_front in fronts.items():
            if other_alg != alg_name:
                metrics[alg_name]['igd'][other_alg] = calculate_igd(front, other_front)

    return metrics

# ====================== 可视化函数 ======================

def configure_fonts():
    """配置全局图表字体设置"""
    # 检测操作系统类型
    system = platform.system()

    # 配置中文字体
    if system == 'Windows':
        chinese_font = 'SimSun'  # Windows系统宋体
    elif system == 'Darwin':
        chinese_font = 'Songti SC'  # macOS系统宋体
    else:
        chinese_font = 'SimSun'  # Linux系统尝试使用宋体

    # 配置英文字体
    english_font = 'Times New Roman'

    # 设置字体
    font_list = [chinese_font, english_font, 'DejaVu Sans']

    # 设置字体大小
    chinese_size = 12
    english_size = 10

    # 配置matplotlib字体
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = font_list
    plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号

    # 设置不同元素的字体
    rcParams['font.size'] = english_size  # 默认英文字体大小
    rcParams['axes.titlesize'] = chinese_size  # 标题字体大小
    rcParams['axes.labelsize'] = english_size  # 轴标签字体大小
    rcParams['xtick.labelsize'] = english_size  # x轴刻度标签字体大小
    rcParams['ytick.labelsize'] = english_size  # y轴刻度标签字体大小
    rcParams['legend.fontsize'] = english_size  # 图例字体大小

    # 设置DPI和图表大小
    rcParams['figure.dpi'] = 100
    rcParams['savefig.dpi'] = 300

def normalize_fronts(fronts, max_variance=None):
    """
    统一水头均方差截取标准

    参数:
    fronts: 字典，键为算法名称，值为帕累托前沿
    max_variance: 最大水头均方差值，若为None则使用所有前沿中的最小最大值

    返回:
    处理后的前沿字典
    """
    # 如果未指定最大均方差，找到所有前沿中的最大值的最小值
    if max_variance is None:
        max_variance = min([np.max(front[:, 1]) for front in fronts.values()])

    # 截取每个前沿
    normalized_fronts = {}
    for alg_name, front in fronts.items():
        # 筛选出均方差不超过max_variance的解
        valid_indices = front[:, 1] <= max_variance
        normalized_fronts[alg_name] = front[valid_indices]

    return normalized_fronts

def visualize_pareto_fronts(fronts, save_path=None, title="算法Pareto前沿对比", show_plot=True):
    """
    可视化多个帕累托前沿对比

    参数:
    fronts: 字典，键为算法名称，值为帕累托前沿
    save_path: 保存路径，若为None则不保存
    title: 图表标题
    show_plot: 是否显示图表
    """
    # 配置字体
    configure_fonts()

    plt.figure(figsize=(12, 9))

    # 配置颜色和标记
    colors = ['green', 'blue', 'red']
    markers = ['s', 'o', '^']

    # 绘制每个前沿
    for (alg_name, front), color, marker in zip(fronts.items(), colors, markers):
        plt.scatter(front[:, 0], front[:, 1],
                  s=70, color=color, label=alg_name, alpha=0.8, marker=marker)

        # 连线
        sorted_indices = np.argsort(front[:, 0])
        sorted_front = front[sorted_indices]
        plt.plot(sorted_front[:, 0], sorted_front[:, 1], '--', color=color, alpha=0.5)

    # 设置坐标轴和标题
    plt.title(title, fontsize=16)
    plt.xlabel("系统成本 (元)", fontsize=14)
    plt.ylabel("水头均方差", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)

    # 优化图表样式
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.ticklabel_format(style='plain', axis='x')

    # 保存图表
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图表已保存到: {save_path}")

    if show_plot:
        plt.show()
    else:
        plt.close()

def generate_metrics_table(metrics, save_path=None):
    """
    生成性能指标表格

    参数:
    metrics: 字典，包含各算法的性能指标
    save_path: 保存路径，若为None则不保存
    """
    # 创建DataFrame
    data = []
    for alg_name, metric_values in metrics.items():
        row = {
            'Algorithm': alg_name,
            'SP': metric_values['sp'],
            'HV': metric_values['hv']
        }
        # 添加IGD指标
        for other_alg, igd_value in metric_values['igd'].items():
            row[f'IGD vs {other_alg}'] = igd_value

        data.append(row)

    df = pd.DataFrame(data)

    # 打印表格
    print("\n性能指标比较:")
    print(df.to_string(index=False))

    # 保存表格
    if save_path:
        df.to_csv(save_path, index=False)
        print(f"指标表格已保存到: {save_path}")

    return df

def visualize_metrics(metrics, save_path=None, show_plot=True):
    """
    可视化性能指标比较

    参数:
    metrics: 字典，包含各算法的性能指标
    save_path: 保存路径，若为None则不保存
    show_plot: 是否显示图表
    """
    # 配置字体
    configure_fonts()

    # 提取数据
    algorithms = list(metrics.keys())
    sp_values = [metrics[alg]['sp'] for alg in algorithms]
    hv_values = [metrics[alg]['hv'] for alg in algorithms]

    # 创建图表
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))

    # SP指标（越小越好）
    ax1.bar(algorithms, sp_values, color=['green', 'blue', 'red'])
    ax1.set_title('SP 指标比较 (均匀性，越小越好)', fontsize=14)
    ax1.set_ylabel('SP 值', fontsize=14)
    ax1.grid(True, linestyle='--', alpha=0.7, axis='y')

    # HV指标（越大越好）
    ax2.bar(algorithms, hv_values, color=['green', 'blue', 'red'])
    ax2.set_title('HV 指标比较 (超体积，越大越好)', fontsize=14)
    ax2.set_ylabel('HV 值', fontsize=14)
    ax2.grid(True, linestyle='--', alpha=0.7, axis='y')

    # 优化图表样式
    for ax in [ax1, ax2]:
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

    plt.tight_layout()

    # 保存图表
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"指标对比图已保存到: {save_path}")

    if show_plot:
        plt.show()
    else:
        plt.close()

# ====================== 主函数 ======================

def main():
    """主函数，演示完整的比较流程"""
    # 配置字体
    configure_fonts()

    # 指定算法名称
    algorithm_names = ["OriginalPSO", "EnhancedMOPSO", "MOPSO"]
    
    # 加载所有帕累托前沿
    print("尝试加载帕累托前沿文件...")
    fronts = load_all_pareto_fronts(algorithm_names)

    if fronts:
        print("成功加载以下算法的帕累托前沿:")
        for alg, front in fronts.items():
            print(f"- {alg}: 包含 {len(front)} 个解")

    # 统一水头均方差截取标准
    max_variance = 0.45  # 可以根据需要调整
    normalized_fronts = normalize_fronts(fronts, max_variance)

    # 可视化帕累托前沿对比
    pareto_path = os.path.join(RESULTS_DIR, "pareto_fronts_comparison.png")
    visualize_pareto_fronts(normalized_fronts, pareto_path, title="灌溉系统优化算法Pareto前沿对比")

    # 设置参考点计算性能指标
    max_cost = max([np.max(front[:, 0]) for front in fronts.values()])
    max_var = max([np.max(front[:, 1]) for front in fronts.values()])
    ref_point = np.array([max_cost * 1.1, max_var * 1.1])  # 参考点设置为最大值的1.1倍

    print(f"使用参考点: [{ref_point[0]}, {ref_point[1]}]")
    metrics = calculate_all_metrics(normalized_fronts, ref_point)

    # 生成性能指标表格
    metrics_table_path = os.path.join(RESULTS_DIR, "performance_metrics.csv")
    generate_metrics_table(metrics, metrics_table_path)

    # 可视化性能指标
    metrics_path = os.path.join(RESULTS_DIR, "metrics_comparison.png")
    visualize_metrics(metrics, metrics_path)

    print(f"\n分析完成！所有结果已保存至 {RESULTS_DIR} 目录")

if __name__ == "__main__":
    main()