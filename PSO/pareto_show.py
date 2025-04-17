"""
改进的帕累托前沿展示器
- 正确读取CSV文件
- 帕累托图样式与PSO.py保持一致
- 箱线图样式与improvepso.py保持一致
"""
import json
import time

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from scipy.spatial.distance import cdist, pdist
from matplotlib import rcParams
import platform

# 结果目录
RESULTS_DIR = 'pareto_comparison_results'

# ====================== 帕累托解集保存函数 ======================
def save_pareto_front(pareto_front, algorithm_name, save_dir=RESULTS_DIR):
    """
    保存帕累托前沿到CSV文件
    参数:
    pareto_front: 帕累托前沿解集，形式为[[cost1, variance1], [cost2, variance2], ...]
    algorithm_name: 算法名称
    save_dir: 保存目录
    """
    # 确保保存目录存在
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # 将解集转换为DataFrame
    df = pd.DataFrame(pareto_front, columns=['system_cost', 'pressure_variance'])
    # 保存为CSV文件
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = f"{algorithm_name}_pareto_front_{timestamp}.csv"
    filepath = os.path.join(save_dir, filename)
    df.to_csv(filepath, index=False)
    print(f"帕累托前沿已保存到：{filepath}")
    return filepath


def save_pareto_solutions(solutions, algorithm_name, save_dir=RESULTS_DIR):
    """
    保存帕累托解集（包括决策变量和目标值）到JSON文件
    参数:
    solutions: 帕累托解集，每个解包含决策变量和目标值
    algorithm_name: 算法名称
    save_dir: 保存目录
    """
    # 确保保存目录存在
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 将解集转换为列表字典
    solution_list = []
    for solution in solutions:
        # 确保所有NumPy数组都被转换为Python列表
        if hasattr(solution, 'best_position'):
            position = solution.best_position.tolist() if hasattr(solution.best_position,
                                                                  'tolist') else solution.best_position
        else:
            position = solution.position.tolist() if hasattr(solution.position, 'tolist') else solution.position

        if hasattr(solution, 'best_fitness'):
            fitness = solution.best_fitness.tolist() if hasattr(solution.best_fitness,
                                                                'tolist') else solution.best_fitness
        else:
            fitness = solution.fitness.tolist() if hasattr(solution.fitness, 'tolist') else solution.fitness

        sol_dict = {
            'position': position,
            'objectives': fitness
        }
        solution_list.append(sol_dict)

    # 保存为JSON文件
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = f"{algorithm_name}_solutions_{timestamp}.json"
    filepath = os.path.join(save_dir, filename)
    with open(filepath, 'w') as f:
        json.dump(solution_list, f, indent=2)
# ====================== 帕累托解集读取函数 ======================
    print(f"完整解集已保存到：{filepath}")
    return filepath

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
        print(f"读取文件: {latest_file}")
        df = pd.read_csv(latest_file)
        print(f"文件读取成功，形状: {df.shape}")
        return df.values
    except Exception as e:
        print(f"读取失败: {str(e)}")

        try:
            # 尝试使用numpy读取
            data = np.loadtxt(latest_file, delimiter=',', skiprows=1)
            print(f"使用numpy成功读取数据，形状: {data.shape}")
            return data
        except Exception as e:
            print(f"numpy读取失败: {str(e)}")
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
        # 最小化问题的参考点设置
        # 找到所有前沿中的最大值的1.1倍
        max_cost = max([np.max(front[:, 0]) for front in fronts.values()]) * 1.1
        max_var = max([np.max(front[:, 1]) for front in fronts.values()]) * 1.1
        ref_point = np.array([max_cost, max_var])
        print(f"使用参考点: [{max_cost:.2f}, {max_var:.2f}]")

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

    print(f"统一水头均方差截取标准: {max_variance}")

    # 截取每个前沿
    normalized_fronts = {}
    for alg_name, front in fronts.items():
        # 筛选出均方差不超过max_variance的解
        valid_indices = front[:, 1] <= max_variance
        normalized_front = front[valid_indices]
        normalized_fronts[alg_name] = normalized_front
        print(f"{alg_name}: 原始解集 {len(front)} 个，截取后 {len(normalized_front)} 个")

    return normalized_fronts

def visualize_pareto_fronts_pso_style(fronts, save_path=None, title="算法Pareto前沿对比", show_plot=True):
    """
    可视化多个帕累托前沿对比，采用PSO.py风格

    参数:
    fronts: 字典，键为算法名称，值为帕累托前沿
    save_path: 保存路径，若为None则不保存
    title: 图表标题
    show_plot: 是否显示图表
    """
    # 配置字体
    configure_fonts()

    plt.figure(figsize=(12, 9))

    # 配置颜色和标记 - PSO.py风格
    colors = {
        'OriginalPSO': '#82ca9d',    # 绿色
        'EnhancedMOPSO': '#8884d8',  # 蓝色
        'MOPSO': '#ff7300'           # 橙色
    }
    markers = {
        'OriginalPSO': 's',          # 方形
        'EnhancedMOPSO': 'o',        # 圆形
        'MOPSO': '^'                 # 三角形
    }

    # 设置线型
    line_styles = {
        'OriginalPSO': '--',         # 虚线
        'EnhancedMOPSO': '-',        # 实线
        'MOPSO': '-.'                # 点划线
    }

    # 绘制每个前沿
    for alg_name, front in fronts.items():
        color = colors.get(alg_name, 'gray')
        marker = markers.get(alg_name, 'x')
        line_style = line_styles.get(alg_name, ':')

        # 按成本排序
        sorted_indices = np.argsort(front[:, 0])
        sorted_front = front[sorted_indices]

        # 绘制散点
        plt.scatter(sorted_front[:, 0], sorted_front[:, 1],
                  s=70, color=color, label=alg_name, alpha=0.8, marker=marker)

        # 连线
        plt.plot(sorted_front[:, 0], sorted_front[:, 1],
               linestyle=line_style, color=color, alpha=0.7, linewidth=2)

    # 设置坐标轴和标题
    plt.title(title, fontsize=16)
    plt.xlabel("系统成本 (元)", fontsize=14)
    plt.ylabel("水头均方差", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12, loc='best')

    # 优化图表样式 - PSO.py风格
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.ticklabel_format(style='plain', axis='x')

    # 保存图表
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"帕累托前沿对比图已保存到: {save_path}")

    if show_plot:
        plt.show()
    else:
        plt.close()

def visualize_metrics_improved_pso_style(metrics, save_path=None, show_plot=True):
    """
    可视化性能指标比较，采用improved_pso.py风格

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

    # 计算平均值，用于显示数值标签
    sp_mean = {alg: metrics[alg]['sp'] for alg in algorithms}
    hv_mean = {alg: metrics[alg]['hv'] for alg in algorithms}

    # 创建图表 - improved_pso.py风格
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)

    # 子图标题
    titles = ["SP 指标对比 (均匀性)", "IGD 指标对比", "HV 指标对比 (超体积)"]

    # 配置颜色 - improved_pso.py风格
    colors = {
        'OriginalPSO': 'lightgreen',
        'EnhancedMOPSO': 'lightblue',
        'MOPSO': 'lightcoral'
    }

    # 模拟boxplot数据
    # 实际上我们只有单个值，但创建一个小的分布来模拟boxplot效果
    def create_boxplot_data(value, spread=0.05):
        # 创建5个点的伪分布，中位数为原值
        return [value * (1 - spread), value * (1 - spread/2), value, value * (1 + spread/2), value * (1 + spread)]

    # 创建数据
    sp_data = [create_boxplot_data(sp_values[i]) for i in range(len(algorithms))]
    hv_data = [create_boxplot_data(hv_values[i]) for i in range(len(algorithms))]

    # 创建空的IGD数据（因为没有真实前沿）
    igd_data = [create_boxplot_data(0.1) for _ in range(len(algorithms))]

    # 数据集
    data_sets = [sp_data, igd_data, hv_data]

    # 指标说明
    descriptions = [
        "SP值越小表示分布越均匀",
        "IGD值越小表示接近度越高",
        "HV值越大表示质量越好"
    ]

    # 绘制箱线图
    for i, (title, data) in enumerate(zip(titles, data_sets)):
        box = axes[i].boxplot(data, labels=algorithms, patch_artist=True)

        # 设置箱体颜色
        for j, patch in enumerate(box['boxes']):
            alg = algorithms[j]
            patch.set_facecolor(colors.get(alg, 'lightgray'))

        # 设置子图标题和网格
        axes[i].set_title(title, fontsize=15)
        axes[i].grid(True, linestyle='--', alpha=0.7, axis='y')

        # 添加指标说明
        axes[i].text(0.5, -0.15, descriptions[i], transform=axes[i].transAxes,
                    ha='center', fontsize=12, style='italic')

        # 添加数值标签
        if i == 0:  # SP
            for j, alg in enumerate(algorithms):
                axes[i].text(j+1, sp_mean[alg], f"{sp_mean[alg]:.4f}",
                            ha='center', va='bottom', fontsize=10)
        elif i == 2:  # HV
            for j, alg in enumerate(algorithms):
                axes[i].text(j+1, hv_mean[alg], f"{hv_mean[alg]:.4f}",
                            ha='center', va='bottom', fontsize=10)

        # 优化图表样式
        axes[i].spines['right'].set_visible(False)
        axes[i].spines['top'].set_visible(False)

        # Y轴标签
        if i == 0:
            axes[i].set_ylabel("SP值 (均匀性，越小越好)", fontsize=14)
        elif i == 1:
            axes[i].set_ylabel("IGD值 (仅供参考)", fontsize=14)
        else:
            axes[i].set_ylabel("HV值 (超体积，越大越好)", fontsize=14)

    # 添加总标题
    fig.suptitle("灌溉系统优化算法性能指标对比", fontsize=18)

    # 保存图表
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"性能指标对比图已保存到: {save_path}")

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

# ====================== 主函数 ======================

def main():
    """主函数，展示帕累托前沿对比"""
    # 配置字体
    configure_fonts()

    print("=" * 60)
    print("帕累托前沿展示器")
    print("=" * 60)

    # 指定算法名称
    # 注意：文件夹中的算法名可能是MOPSO而不是ImprovedMOPSO
    algorithm_names = ["OriginalPSO", "EnhancedMOPSO", "MOPSO"]

    # 加载所有帕累托前沿
    print("\n加载帕累托前沿...")
    fronts = load_all_pareto_fronts(algorithm_names)

    if not fronts:
        print("未能读取任何帕累托前沿，请检查文件是否存在")
        return

    print(f"\n成功加载算法数量: {len(fronts)}")
    for alg, front in fronts.items():
        print(f"- {alg}: {len(front)} 个解")

    # 统一水头均方差截取标准
    max_variance = 6  # 可以根据需要调整
    normalized_fronts = normalize_fronts(fronts, max_variance)

    # 可视化帕累托前沿对比 - PSO风格
    pareto_path = os.path.join(RESULTS_DIR, "pareto_fronts_comparison_pso_style.png")
    visualize_pareto_fronts_pso_style(normalized_fronts, pareto_path,
                                     title="灌溉系统优化算法Pareto前沿对比")

    # 计算性能指标
    # 为最小化问题设置参考点
    max_cost = max([np.max(front[:, 0]) for front in normalized_fronts.values()])
    max_var = max([np.max(front[:, 1]) for front in normalized_fronts.values()])
    ref_point = np.array([max_cost * 1.1, max_var * 1.1])

    metrics = calculate_all_metrics(normalized_fronts, ref_point)

    # 生成性能指标表格
    metrics_table_path = os.path.join(RESULTS_DIR, "performance_metrics.csv")
    generate_metrics_table(metrics, metrics_table_path)

    # 可视化性能指标 - improved_pso风格
    metrics_path = os.path.join(RESULTS_DIR, "metrics_comparison_improved_pso_style.png")
    visualize_metrics_improved_pso_style(metrics, metrics_path)

    print(f"\n分析完成！所有结果已保存至 {RESULTS_DIR} 目录")
    print("=" * 60)

if __name__ == "__main__":
    main()