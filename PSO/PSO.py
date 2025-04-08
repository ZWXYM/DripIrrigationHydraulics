import math
import random
import logging
import time
import numpy as np
import matplotlib.pyplot as plt
from deap import tools
from matplotlib import rcParams
import platform

# 系统常量定义
DRIPPER_SPACING = 0.3  # 滴灌孔间隔（米）
DEFAULT_NODE_SPACING = 400  # 默认节点间距（米）
DEFAULT_FIRST_SEGMENT_LENGTH = 200  # 第一个管段的默认长度（米）
DEFAULT_SUBMAIN_LENGTH = 800  # 默认斗管长度（米）
DEFAULT_LATERAL_LENGTH = 200  # 默认农管长度（米）
DEFAULT_AUXILIARY_LENGTH = 50  # 默认辅助农管长度（米）
DEFAULT_DRIP_LINE_LENGTH = 50  # 默认滴灌带长度（米）
DEFAULT_DRIP_LINE_SPACING = 1  # 默认滴灌带间隔（米）
DEFAULT_DRIPPER_FLOW_RATE = 2.1  # 默认滴灌孔流量（L/h）
DEFAULT_DRIP_LINE_INLET_PRESSURE = 10  # 默认滴灌带入口水头压力（米）
PRESSURE_BASELINE = 23.8  # 基准压力值

# 日志和图表配置
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']

# 管道规格和价格数据
PIPE_SPECS = {
    "submain": {
        "material": "UPVC",
        "pressure": "0.63MPa",
        "diameters": [140, 160],
        "prices": [29.99, 36.95]
    },
    "main": {
        "material": "玻璃钢管",
        "pressure": "0.63MPa",
        "diameters": [300, 350, 400, 450, 500, 600],
        "prices": [145.20, 180.00, 208.80, 234.00, 276.00, 375.60]
    },
    "lateral": {
        "material": "PE",
        "pressure": "0.4MPa",
        "diameters": [75, 90],
        "prices": [11.78, 15.60]
    },
    "drip_line": {
        "material": "滴灌带",
        "pressure": "-",
        "diameters": [16],
        "prices": [0.42]
    }
}


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

    # 返回字体配置，以便在特定函数中使用
    return {
        'chinese_font': chinese_font,
        'english_font': english_font,
        'chinese_size': chinese_size,
        'english_size': english_size
    }


# 水力学计算函数
def water_speed(diameter, flow_rate):
    """计算流速"""
    d = diameter / 1000
    speed = flow_rate / ((d / 2) ** 2 * math.pi)
    return speed


def friction_factor(diameter, flow_rate, pipe_roughness=1.5e-6):
    """计算摩阻系数"""
    d = diameter / 1000
    if d <= 0 or flow_rate <= 0:
        return 0, 0

    v = water_speed(diameter, flow_rate)
    Re = 1000 * v * d / 1.004e-3

    if Re == 0:
        return 0, 0

    relative_roughness = pipe_roughness / d

    if Re < 2300:
        return 64 / Re, Re
    elif Re > 4000:
        A = (relative_roughness / 3.7) ** 1.11 + (5.74 / Re) ** 0.9
        f = 0.25 / (math.log10(A) ** 2)
        return f, Re
    else:
        f_2300 = 64 / 2300
        A_4000 = relative_roughness / 3.7 + 5.74 / 4000 ** 0.9
        f_4000 = 0.25 / (math.log10(A_4000) ** 2)
        f = f_2300 + (f_4000 - f_2300) * (Re - 2300) / (4000 - 2300)
        return f, Re


def pressure_loss(diameter, length, flow_rate):
    """计算压力损失"""
    f, Re = friction_factor(diameter, flow_rate)
    d = diameter / 1000
    v = water_speed(diameter, flow_rate)
    h_f = f * (length / d) * (v ** 2 / (2 * 9.81))
    return h_f


class IrrigationSystem:
    def __init__(self, node_count, rukoushuitou, node_spacing=DEFAULT_NODE_SPACING,
                 first_segment_length=DEFAULT_FIRST_SEGMENT_LENGTH,
                 submain_length=DEFAULT_SUBMAIN_LENGTH,
                 lateral_layout="single"):
        """初始化灌溉系统"""
        self.node_count = node_count
        self.node_spacing = node_spacing
        self.first_segment_length = first_segment_length
        self.submain_length = submain_length
        self.lateral_layout = lateral_layout
        self.rukoushuitou = rukoushuitou

        # 初始化管网结构
        self.main_pipe = self._create_main_pipe()
        self.submains = self._create_submains()
        self.laterals = self._create_laterals()
        self.drip_lines = self._create_drip_lines()

        # 轮灌组相关属性
        self.lgz1 = None
        self.lgz2 = None
        self.irrigation_groups = []
        self.current_group_index = None

    def initialize_irrigation_groups(self, lgz1, lgz2):
        """初始化轮灌组配置"""
        self.lgz1 = lgz1
        self.lgz2 = lgz2
        self.irrigation_groups = self._generate_irrigation_groups()
        return len(self.irrigation_groups)

    def _generate_irrigation_groups(self):
        """生成轮灌组分配方案"""
        groups = []
        start_idx = 1
        end_idx = self.node_count

        while start_idx <= end_idx:
            current_group = []
            nodes_needed = min(self.lgz1, end_idx - start_idx + 1)

            for i in range(nodes_needed // 2):
                if start_idx <= end_idx:
                    current_group.extend([start_idx, end_idx])
                    start_idx += 1
                    end_idx -= 1

            if len(current_group) < nodes_needed and start_idx <= end_idx:
                current_group.append(start_idx)
                start_idx += 1

            if current_group:
                groups.append(sorted(current_group))

        return groups

    def evaluate_group(self, group_index):
        """评估指定轮灌组的性能"""
        if not 0 <= group_index < len(self.irrigation_groups):
            raise ValueError("无效的轮灌组索引")

        active_nodes = self.irrigation_groups[group_index]
        self._update_flow_rates(active_nodes)
        self._calculate_hydraulics()
        self._calculate_pressures()

        return {
            'head_loss': self._get_total_head_loss(),
            'pressure_variance': self.calculate_pressure_variance(active_nodes),  # 修改这里
            'cost': self.get_system_cost(),
            'pressure_satisfaction': self._check_pressure_requirements()
        }

    def _update_flow_rates(self, active_nodes):
        """精确更新管网流量分配"""
        # 清空所有流量
        for pipe in self.main_pipe:
            pipe["flow_rate"] = 0
        for submain in self.submains:
            submain["flow_rate"] = 0
        for lateral in self.laterals:
            lateral["flow_rate"] = 0

        # 计算基础流量
        drippers_per_line = math.ceil(DEFAULT_DRIP_LINE_LENGTH / DRIPPER_SPACING)
        single_dripper_flow = DEFAULT_DRIPPER_FLOW_RATE / 3600000  # 转换为m³/s
        lateral_flow = drippers_per_line * single_dripper_flow * 100  # 一条农管的流量
        submain_flow = lateral_flow * self.lgz2  # 一条斗管的流量

        # 启用的斗管设置流量
        for node in active_nodes:
            if node <= len(self.submains):
                self.submains[node - 1]["flow_rate"] = submain_flow

        # 干管流量计算（包括管段0）
        active_nodes_sorted = sorted(active_nodes)

        # 首先计算管段0的流量（等于系统总流量）
        self.main_pipe[0]["flow_rate"] = len([node for node in active_nodes_sorted
                                              if node <= len(self.submains)]) * submain_flow

        # 计算其他管段的流量
        for i in range(1, len(self.main_pipe)):
            downstream_flow = sum(submain_flow
                                  for node in active_nodes_sorted
                                  if node >= i and node <= len(self.submains))
            self.main_pipe[i]["flow_rate"] = downstream_flow

    def _calculate_hydraulics(self):
        """计算水力特性"""
        # 计算干管水力特性
        for segment in self.main_pipe:
            if segment["flow_rate"] > 0:
                segment["velocity"] = water_speed(segment["diameter"], segment["flow_rate"])
                segment["head_loss"] = pressure_loss(segment["diameter"],
                                                     segment["length"],
                                                     segment["flow_rate"])
            else:
                segment["velocity"] = 0
                segment["head_loss"] = 0

        # 计算斗管水力特性
        for submain in self.submains:
            if submain["flow_rate"] > 0:
                first_loss = pressure_loss(submain["diameter_first_half"],
                                           submain["length"] / 2,
                                           submain["flow_rate"])
                second_loss = pressure_loss(submain["diameter_second_half"],
                                            submain["length"] / 2,
                                            submain["flow_rate"])
                submain["head_loss"] = first_loss + second_loss
            else:
                submain["head_loss"] = 0

        # 计算农管水力特性
        for lateral in self.laterals:
            if lateral["flow_rate"] > 0:
                lateral["head_loss"] = pressure_loss(lateral["diameter"],
                                                     lateral["length"],
                                                     lateral["flow_rate"])
            else:
                lateral["head_loss"] = 0

    def _calculate_pressures(self):
        """计算所有节点的压力"""
        # 干管压力计算
        cumulative_loss = 0
        for i, segment in enumerate(self.main_pipe):
            if i == 0:
                segment["pressure"] = self.rukoushuitou
            else:
                previous_pressure = self.main_pipe[i - 1]["pressure"]
                current_loss = self.main_pipe[i - 1]["head_loss"]
                segment["pressure"] = previous_pressure - current_loss

            # 计算对应斗管压力
            if i > 0 and i <= len(self.submains):
                submain = self.submains[i - 1]
                submain["inlet_pressure"] = segment["pressure"]
                submain["outlet_pressure"] = submain["inlet_pressure"] - submain["head_loss"]

                # 计算农管压力
                submain_laterals = [lat for lat in self.laterals if lat["submain_index"] == i - 1]
                for lateral in submain_laterals:
                    if lateral["flow_rate"] > 0:
                        lateral["inlet_pressure"] = submain["outlet_pressure"]
                        lateral["outlet_pressure"] = lateral["inlet_pressure"] - lateral["head_loss"]

    def get_system_cost(self):
        """计算系统总成本"""
        cost = 0

        # 创建价格查找表
        price_lookup = {
            "main": {d: p for d, p in zip(PIPE_SPECS["main"]["diameters"],
                                          PIPE_SPECS["main"]["prices"])},
            "submain": {d: p for d, p in zip(PIPE_SPECS["submain"]["diameters"],
                                             PIPE_SPECS["submain"]["prices"])},
            "lateral": {d: p for d, p in zip(PIPE_SPECS["lateral"]["diameters"],
                                             PIPE_SPECS["lateral"]["prices"])}
        }

        # 计算干管成本（从管段1开始）
        for segment in self.main_pipe[1:]:
            if segment["diameter"] > 0:
                cost += segment["length"] * price_lookup["main"][segment["diameter"]]

        # 计算斗管成本
        for submain in self.submains:
            if submain["diameter_first_half"] > 0 and submain["diameter_second_half"] > 0:
                cost += (submain["length"] / 2) * price_lookup["submain"][submain["diameter_first_half"]]
                cost += ((submain["length"] / 2) - 50) * price_lookup["submain"][submain["diameter_second_half"]]

        # 计算农管成本
        lateral_configs = {}
        for lateral in self.laterals:
            if lateral["diameter"] > 0:
                key = (lateral["length"], lateral["diameter"])
                lateral_configs[key] = lateral_configs.get(key, 0) + 1

        for (length, diameter), count in lateral_configs.items():
            cost += length * price_lookup["lateral"][diameter] * count

        # 计算滴灌带成本
        total_drip_line_length = sum(dl["length"] for dl in self.drip_lines)
        cost += total_drip_line_length * PIPE_SPECS["drip_line"]["prices"][0]

        return cost

    def _get_total_head_loss(self):
        """计算总水头损失"""
        return sum(segment["head_loss"] for segment in self.main_pipe
                   if segment["flow_rate"] > 0)

    def _get_pressure_variance(self):
        """计算富裕水头方差"""
        active_nodes = [i + 1 for i, submain in enumerate(self.submains) if submain["flow_rate"] > 0]
        return self.calculate_pressure_variance(active_nodes)

    def _check_pressure_requirements(self):
        """检查压力要求满足情况"""
        pressures = [submain["outlet_pressure"] for submain in self.submains
                     if submain["flow_rate"] > 0]
        return all(p >= DEFAULT_DRIP_LINE_INLET_PRESSURE for p in pressures)

    def _create_main_pipe(self):
        """创建干管"""
        segments = []
        for i in range(self.node_count + 1):
            segments.append({
                "index": i,
                "length": self.first_segment_length if i == 0 else self.node_spacing,
                "diameter": 500,
                "flow_rate": 0.0,
                "velocity": 0.0,
                "head_loss": 0.0,
                "pressure": 0.0
            })
        return segments

    def _create_submains(self):
        """创建斗管"""
        return [{"index": i,
                 "length": self.submain_length,
                 "diameter_first_half": 160,
                 "diameter_second_half": 140,
                 "flow_rate": 0.0,
                 "head_loss": 0.0,
                 "inlet_pressure": 0.0,
                 "outlet_pressure": 0.0
                 } for i in range(self.node_count)]

    def _create_laterals(self):
        """创建农管"""
        laterals = []
        for submain in self.submains:
            lateral_count = math.ceil(submain["length"] / (50 * 2)) * 2

            for i in range(lateral_count):
                laterals.append({
                    "submain_index": submain["index"],
                    "index": i,
                    "length": DEFAULT_LATERAL_LENGTH,
                    "diameter": 90,
                    "flow_rate": 0.0,
                    "head_loss": 0.0,
                    "inlet_pressure": 0.0,
                    "outlet_pressure": 0.0
                })
        return laterals

    def _create_drip_lines(self):
        """创建滴灌带"""
        drip_lines = []
        for lateral in self.laterals:
            auxiliary_count = math.floor(lateral["length"] / DEFAULT_AUXILIARY_LENGTH)
            for i in range(auxiliary_count):
                drip_line_count = math.floor(DEFAULT_AUXILIARY_LENGTH / DEFAULT_DRIP_LINE_SPACING) * 2
                for j in range(drip_line_count):
                    drip_lines.append({
                        "lateral_index": lateral["index"],
                        "index": j,
                        "length": DEFAULT_DRIP_LINE_LENGTH,
                        "diameter": 16,
                        "dripper_count": math.ceil(DEFAULT_DRIP_LINE_LENGTH / DRIPPER_SPACING),
                        "flow_rate": 0.0
                    })
        return drip_lines

    def calculate_pressure_variance(self, active_nodes):
        """统一计算水头均方差的标准方法"""
        pressure_margins = [self.submains[node - 1]["inlet_pressure"] - PRESSURE_BASELINE
                            for node in active_nodes
                            if node <= len(self.submains) and self.submains[node - 1]["flow_rate"] > 0]

        if not pressure_margins:
            return 0

        avg_margin = sum(pressure_margins) / len(pressure_margins)
        variance = sum((p - avg_margin) ** 2 for p in pressure_margins) / len(pressure_margins)
        return variance ** 0.5


class PSOOptimizationTracker:
    def __init__(self, show_dynamic_plots=False, auto_save=True, enable_smoothing=True):
        self.iterations = []
        self.best_costs = []
        self.best_variances = []
        self.all_costs = []
        self.all_variances = []
        self.all_iterations = []

        # 字体配置
        self.font_config = configure_fonts()

        # 动态图表显示设置
        self.show_dynamic_plots = show_dynamic_plots
        self.auto_save = auto_save
        self.enable_smoothing = enable_smoothing  # 新增控制平滑的选项
        self.fig_2d = None
        self.ax1 = None
        self.ax2 = None
        self.fig_3d = None
        self.ax_3d = None
        self.scatter = None
        self.line1 = None
        self.line2 = None

        # 如果启用动态图表，初始化图表
        if self.show_dynamic_plots:
            self._init_plots()

    def _init_plots(self):
        """初始化动态图表"""
        # 确保交互模式开启
        plt.ion()

        # 获取字体配置
        chinese_font = self.font_config['chinese_font']
        english_font = self.font_config['english_font']
        chinese_size = self.font_config['chinese_size']
        english_size = self.font_config['english_size']

        # 初始化2D图表
        self.fig_2d, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        self.ax1.set_ylabel('系统成本 (元)', fontproperties=chinese_font, fontsize=chinese_size)
        self.ax1.set_title('梳齿状PSO算法优化迭代曲线', fontproperties=chinese_font, fontsize=chinese_size + 2)
        self.ax1.grid(True, linestyle='--', alpha=0.7)

        self.ax2.set_xlabel('迭代次数', fontproperties=chinese_font, fontsize=chinese_size)
        self.ax2.set_ylabel('水头均方差', fontproperties=chinese_font, fontsize=chinese_size)
        self.ax2.grid(True, linestyle='--', alpha=0.7)

        # 设置tick标签字体
        for label in self.ax1.get_xticklabels() + self.ax1.get_yticklabels():
            label.set_fontname(english_font)
            label.set_fontsize(english_size)

        for label in self.ax2.get_xticklabels() + self.ax2.get_yticklabels():
            label.set_fontname(english_font)
            label.set_fontsize(english_size)

        # 创建空的线条对象 - 移除标记点，只使用线条
        self.line1, = self.ax1.plot([], [], 'b-', linewidth=2)
        self.line2, = self.ax2.plot([], [], 'r-', linewidth=2)

        # 初始化3D图表
        self.fig_3d = plt.figure(figsize=(12, 10))
        self.ax_3d = self.fig_3d.add_subplot(111, projection='3d')
        self.ax_3d.set_xlabel('系统成本 (元)', fontproperties=chinese_font, fontsize=chinese_size)
        self.ax_3d.set_ylabel('水头均方差', fontproperties=chinese_font, fontsize=chinese_size)
        self.ax_3d.set_zlabel('迭代次数', fontproperties=chinese_font, fontsize=chinese_size)
        self.ax_3d.set_title('梳齿状PSO算法优化3D进度图', fontproperties=chinese_font, fontsize=chinese_size + 2)
        self.ax_3d.view_init(elev=30, azim=-35)  # 默认30，45

        # 设置3D图的tick标签字体
        for label in self.ax_3d.get_xticklabels() + self.ax_3d.get_yticklabels() + self.ax_3d.get_zticklabels():
            label.set_fontname(english_font)
            label.set_fontsize(english_size)

        # 显示图表
        self.fig_2d.canvas.draw()
        self.fig_2d.canvas.flush_events()
        self.fig_3d.canvas.draw()
        self.fig_3d.canvas.flush_events()

    # 新增平滑曲线相关方法
    def _smooth_curve(self, data, window_size=21, poly_order=3):
        """使用Savitzky-Golay滤波器平滑数据"""
        import numpy as np

        # 如果数据点不足，返回原始数据
        if len(data) < window_size:
            return data

        # 确保窗口大小是奇数
        if window_size % 2 == 0:
            window_size += 1

        # 确保窗口大小小于数据长度
        if window_size >= len(data):
            window_size = min(len(data) - 2, 15)
            if window_size % 2 == 0:
                window_size -= 1

        try:
            # 尝试导入scipy并使用Savitzky-Golay滤波器
            from scipy.signal import savgol_filter
            return savgol_filter(data, window_size, poly_order)
        except (ImportError, ValueError):
            # 如果scipy不可用或出错，使用简单的移动平均
            return self._moving_average(data, window_size=5)

    def _moving_average(self, data, window_size=5):
        """计算移动平均"""
        import numpy as np

        if len(data) < window_size:
            return data

        # 创建均匀权重的窗口
        weights = np.ones(window_size) / window_size
        # 使用卷积计算移动平均
        smoothed = np.convolve(data, weights, mode='same')

        # 处理边缘效应
        # 前半部分
        for i in range(window_size // 2):
            if i < len(smoothed):
                window = data[:i + window_size // 2 + 1]
                smoothed[i] = sum(window) / len(window)

        # 后半部分
        for i in range(len(data) - window_size // 2, len(data)):
            if i < len(smoothed):
                window = data[i - window_size // 2:]
                smoothed[i] = sum(window) / len(window)

        return smoothed

    def _exponential_moving_average(self, data, alpha=0.15):
        """计算指数移动平均"""
        import numpy as np

        if len(data) < 2:
            return data

        ema = np.zeros_like(data)
        ema[0] = data[0]

        for i in range(1, len(data)):
            ema[i] = alpha * data[i] + (1 - alpha) * ema[i - 1]

        return ema

    # 新方法 - 直接接收帕累托前沿
    def update_with_front(self, iteration, swarm, pareto_front):
        """接收直接传入的帕累托前沿，与最终结果使用相同的选择方法，增加平滑处理"""
        self.iterations.append(iteration)

        # 获取有效解用于3D图
        valid_solutions = [particle for particle in pareto_front if np.all(np.isfinite(particle.best_fitness))]

        if valid_solutions:
            # 直接使用与主函数相同的选择策略
            # 使用相同的选择函数和水头均方差阈值
            # 从外部模块导入select_best_solution_by_marginal_improvement函数
            import __main__
            if hasattr(__main__, 'select_best_solution_by_marginal_improvement'):
                # 使用与主程序相同的函数和参数
                best_solution = __main__.select_best_solution_by_marginal_improvement(valid_solutions)
            else:
                # 如果无法导入，使用内部实现
                best_solution = self._select_best_solution(valid_solutions)

            # 平滑处理：如果之前已有数据，检测变化是否过大
            if self.best_costs and self.best_variances:
                prev_cost = self.best_costs[-1]
                prev_variance = self.best_variances[-1]

                curr_cost = best_solution.best_fitness[0]
                curr_variance = best_solution.best_fitness[1]

                # 计算相对变化
                if prev_cost != 0:
                    cost_change_ratio = abs((curr_cost - prev_cost) / prev_cost)
                else:
                    cost_change_ratio = 0

                if prev_variance != 0:
                    var_change_ratio = abs((curr_variance - prev_variance) / prev_variance)
                else:
                    var_change_ratio = 0

                # 如果变化过大(超过15%)，进行平滑处理
                if iteration > 20 and (cost_change_ratio > 0.15 or var_change_ratio > 0.15):
                    # 指数平滑公式: new_value = α * current + (1-α) * previous
                    # α = 0.3 意味着更重视历史值，减轻当前值的影响
                    smoothed_cost = 0.3 * curr_cost + 0.7 * prev_cost
                    smoothed_variance = 0.3 * curr_variance + 0.7 * prev_variance

                    # 记录平滑后的值
                    self.best_costs.append(smoothed_cost)
                    self.best_variances.append(smoothed_variance)
                else:
                    # 变化不大，记录实际值
                    self.best_costs.append(curr_cost)
                    self.best_variances.append(curr_variance)
            else:
                # 第一个数据点，直接记录
                self.best_costs.append(best_solution.best_fitness[0])
                self.best_variances.append(best_solution.best_fitness[1])
        else:
            # 处理没有有效解的情况
            if self.best_costs:
                self.best_costs.append(self.best_costs[-1])
                self.best_variances.append(self.best_variances[-1])
            else:
                self.best_costs.append(float('inf'))
                self.best_variances.append(float('inf'))

        # 收集所有解的数据用于3D可视化
        for solution in valid_solutions:
            self.all_costs.append(solution.best_fitness[0])
            self.all_variances.append(solution.best_fitness[1])
            self.all_iterations.append(iteration)

        # 如果启用了动态图表，更新图表
        if self.show_dynamic_plots and iteration % 20 == 0:
            try:
                self._update_plots()
            except Exception as e:
                print(f"图表更新时出错: {e}")

    # 保留原来的update方法，但使用新的前沿逻辑
    def update(self, iteration, swarm, pareto_front=None):
        """更新跟踪器的数据（为向下兼容保留，但内部使用新逻辑）"""
        if pareto_front is None:
            # 获取有效解
            valid_solutions = [particle for particle in swarm if np.all(np.isfinite(particle.best_fitness))]
            # 从有效粒子中提取帕累托前沿
            pareto_front = self._extract_pareto_front(valid_solutions)

        # 使用新方法
        self.update_with_front(iteration, swarm, pareto_front)

    # 内部辅助方法 - 从粒子群中提取帕累托前沿
    def _extract_pareto_front(self, particles):
        """从粒子列表中提取帕累托前沿"""
        pareto_front = []
        for particle in particles:
            is_dominated = False
            for other in particles:
                if self._dominates(other.best_fitness, particle.best_fitness):
                    is_dominated = True
                    break
            if not is_dominated:
                pareto_front.append(particle)
        return pareto_front

    # 内部辅助方法 - 判断一个适应度是否支配另一个
    def _dominates(self, fitness_a, fitness_b):
        """判断fitness_a是否支配fitness_b"""
        not_worse = all(a <= b for a, b in zip(fitness_a, fitness_b))
        better = any(a < b for a, b in zip(fitness_a, fitness_b))
        return not_worse and better

    # 内部辅助方法 - 确保与全局函数完全一致
    def _select_best_solution(self, solutions, max_variance_threshold=5.0):
        """
        内部L方法选择函数 - 保持与全局函数一致，确保在跟踪器无法导入全局函数时可以使用
        """
        # 确保有解可选
        if not solutions:
            return None

        # 首先筛选出水头均方差低于阈值的解
        valid_solutions = [sol for sol in solutions if sol.best_fitness[1] <= max_variance_threshold]

        # 如果没有满足条件的解，从原始解集中选择方差最小的
        if not valid_solutions:
            return min(solutions, key=lambda x: x.best_fitness[1])

        # 按成本升序排序
        sorted_solutions = sorted(valid_solutions, key=lambda particle: particle.best_fitness[0])

        # 如果只有几个解，使用简单策略
        if len(sorted_solutions) <= 3:
            # 如果只有1-3个解，返回成本最低的
            return sorted_solutions[0]

        # 提取成本和方差值
        costs = [sol.best_fitness[0] for sol in sorted_solutions]
        variances = [sol.best_fitness[1] for sol in sorted_solutions]

        # 正规化数据到[0,1]范围
        min_cost = min(costs)
        max_cost = max(costs)
        min_var = min(variances)
        max_var = max(variances)

        # 避免除以零
        cost_range = max_cost - min_cost if max_cost > min_cost else 1
        var_range = max_var - min_var if max_var > min_var else 1

        normalized_costs = [(c - min_cost) / cost_range for c in costs]
        normalized_vars = [(v - min_var) / var_range for v in variances]

        # 对所有可能的分割点计算L方法的误差
        best_error = float('inf')
        best_idx = 0

        for i in range(1, len(sorted_solutions) - 1):
            # 前半部分线性拟合误差
            c1 = np.array(normalized_costs[:i + 1])
            v1 = np.array(normalized_vars[:i + 1])

            # 避免只有一个点的情况
            if len(c1) > 1:
                slope1, intercept1 = np.polyfit(c1, v1, 1)
                line1 = slope1 * c1 + intercept1
                error1 = np.sum((v1 - line1) ** 2)
            else:
                error1 = 0

            # 后半部分线性拟合误差
            c2 = np.array(normalized_costs[i:])
            v2 = np.array(normalized_vars[i:])

            # 避免只有一个点的情况
            if len(c2) > 1:
                slope2, intercept2 = np.polyfit(c2, v2, 1)
                line2 = slope2 * c2 + intercept2
                error2 = np.sum((v2 - line2) ** 2)
            else:
                error2 = 0

            # 计算总误差
            total_error = error1 + error2

            # 更新最佳分割点
            if total_error < best_error:
                best_error = total_error
                best_idx = i

        # 返回L方法确定的最佳解
        return sorted_solutions[best_idx]

    def _update_plots(self):
        """更新动态图表"""
        if not self.iterations:
            return

        # 获取字体配置
        chinese_font = self.font_config['chinese_font']
        english_font = self.font_config['english_font']
        chinese_size = self.font_config['chinese_size']
        english_size = self.font_config['english_size']

        try:
            # 平滑处理数据
            cost_data = self.best_costs
            variance_data = self.best_variances

            if self.enable_smoothing and len(self.iterations) > 5:
                # 应用平滑算法
                smoothed_cost = self._smooth_curve(cost_data)
                smoothed_variance = self._smooth_curve(variance_data)

                # 应用二次平滑 - 指数移动平均
                smoothed_cost = self._exponential_moving_average(smoothed_cost)
                smoothed_variance = self._exponential_moving_average(smoothed_variance)
            else:
                smoothed_cost = cost_data
                smoothed_variance = variance_data

            # 更新2D图表中的数据
            self.line1.set_data(self.iterations, smoothed_cost)
            self.line2.set_data(self.iterations, smoothed_variance)

            # 调整坐标轴范围
            self.ax1.relim()
            self.ax1.autoscale_view()
            self.ax2.relim()
            self.ax2.autoscale_view()
            # 更新标题，反映这是最优个体的迭代曲线
            self.ax1.set_title('梳齿状PSO算法优化最优个体迭代曲线', fontproperties=chinese_font,
                               fontsize=chinese_size + 2)

            # 更新3D图表
            self.ax_3d.clear()
            self.scatter = self.ax_3d.scatter(
                self.all_costs,
                self.all_variances,
                self.all_iterations,
                c=self.all_iterations,
                cmap='viridis',
                s=50,
                alpha=0.6
            )

            self.ax_3d.set_xlabel('系统成本 (元)', fontproperties=chinese_font, fontsize=chinese_size)
            self.ax_3d.set_ylabel('水头均方差', fontproperties=chinese_font, fontsize=chinese_size)
            self.ax_3d.set_zlabel('迭代次数', fontproperties=chinese_font, fontsize=chinese_size)
            self.ax_3d.set_title('梳齿状PSO算法优化3D进度图', fontproperties=chinese_font, fontsize=chinese_size + 2)
            self.ax_3d.view_init(elev=30, azim=-35)  # 默认30，45

            # 设置3D图的tick标签字体
            for label in self.ax_3d.get_xticklabels() + self.ax_3d.get_yticklabels() + self.ax_3d.get_zticklabels():
                label.set_fontname(english_font)
                label.set_fontsize(english_size)

            # 刷新图表 - 使用更可靠的方法
            self.fig_2d.canvas.draw_idle()
            self.fig_3d.canvas.draw_idle()

            # 处理事件队列
            self.fig_2d.canvas.flush_events()
            self.fig_3d.canvas.flush_events()

            # 短暂暂停以确保图形更新，但减少暂停时间
            plt.pause(0.0001)  # 减少暂停时间

        except Exception as e:
            print(f"图表更新过程中出错: {str(e)}")

    def finalize_plots(self):
        """优化结束后最终更新图表"""
        if not self.show_dynamic_plots:
            # 如果没有启用动态图表，创建新图表
            self.plot_2d_curves()
            self.plot_3d_progress()
            return

        # 获取字体配置
        chinese_font = self.font_config['chinese_font']
        english_font = self.font_config['english_font']
        chinese_size = self.font_config['chinese_size']
        english_size = self.font_config['english_size']

        # 更新已有的动态图表
        try:
            # 平滑处理数据
            cost_data = self.best_costs
            variance_data = self.best_variances

            if self.enable_smoothing and len(self.iterations) > 5:
                # 应用平滑算法 - 使用更大的窗口进行最终展示
                smoothed_cost = self._smooth_curve(cost_data, window_size=25)
                smoothed_variance = self._smooth_curve(variance_data, window_size=25)

                # 应用二次平滑 - 指数移动平均
                smoothed_cost = self._exponential_moving_average(smoothed_cost, alpha=0.12)
                smoothed_variance = self._exponential_moving_average(smoothed_variance, alpha=0.12)
            else:
                smoothed_cost = cost_data
                smoothed_variance = variance_data

            # 更新2D图表
            self.line1.set_data(self.iterations, smoothed_cost)
            self.line2.set_data(self.iterations, smoothed_variance)

            # 调整坐标轴范围
            self.ax1.relim()
            self.ax1.autoscale_view()
            self.ax2.relim()
            self.ax2.autoscale_view()
            self.fig_2d.tight_layout()

            # 更新3D图表
            self.ax_3d.clear()
            self.scatter = self.ax_3d.scatter(
                self.all_costs,
                self.all_variances,
                self.all_iterations,
                c=self.all_iterations,
                cmap='viridis',
                s=50,
                alpha=0.6
            )

            # 添加颜色条
            cbar = self.fig_3d.colorbar(self.scatter, ax=self.ax_3d, pad=0.1)
            cbar.set_label('迭代次数', fontproperties=chinese_font, fontsize=chinese_size)

            # 设置颜色条刻度标签字体
            for label in cbar.ax.get_yticklabels():
                label.set_fontname(english_font)
                label.set_fontsize(english_size)

            self.ax_3d.set_xlabel('系统成本 (元)', fontproperties=chinese_font, fontsize=chinese_size)
            self.ax_3d.set_ylabel('水头均方差', fontproperties=chinese_font, fontsize=chinese_size)
            self.ax_3d.set_zlabel('迭代次数', fontproperties=chinese_font, fontsize=chinese_size)
            self.ax_3d.set_title('梳齿状PSO算法优化3D进度图', fontproperties=chinese_font, fontsize=chinese_size + 2)

            # 设置3D图的tick标签字体
            for label in self.ax_3d.get_xticklabels() + self.ax_3d.get_yticklabels() + self.ax_3d.get_zticklabels():
                label.set_fontname(english_font)
                label.set_fontsize(english_size)

            # 如果需要，保存图表
            if self.auto_save:
                self.fig_2d.savefig('PSO_DAN_2d_curves.png', dpi=300, bbox_inches='tight')
                self.fig_3d.savefig('PSO_DAN_3d_progress.png', dpi=300, bbox_inches='tight')

            # 刷新图表
            self.fig_2d.canvas.draw()
            self.fig_3d.canvas.draw()

        except Exception as e:
            print(f"最终图表更新时出错: {e}")

    def plot_2d_curves(self):
        """绘制最终的2D迭代曲线"""
        if not self.iterations:
            print("没有数据可供绘图")
            return

        # 获取字体配置
        chinese_font = self.font_config['chinese_font']
        english_font = self.font_config['english_font']
        chinese_size = self.font_config['chinese_size']
        english_size = self.font_config['english_size']

        # 确保在非交互模式下创建新图表
        plt.ioff()

        # 创建新图表
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

        # 平滑处理数据
        cost_data = self.best_costs
        variance_data = self.best_variances

        if self.enable_smoothing and len(self.iterations) > 5:
            # 应用平滑算法 - 最终图表使用更大的窗口
            smoothed_cost = self._smooth_curve(cost_data, window_size=25)
            smoothed_variance = self._smooth_curve(variance_data, window_size=25)

            # 应用二次平滑 - 指数移动平均
            smoothed_cost = self._exponential_moving_average(smoothed_cost, alpha=0.12)
            smoothed_variance = self._exponential_moving_average(smoothed_variance, alpha=0.12)
        else:
            smoothed_cost = cost_data
            smoothed_variance = variance_data

        # 成本曲线 - 移除标记点，只使用线条
        ax1.plot(self.iterations, smoothed_cost, 'b-', linewidth=2)
        ax1.set_ylabel('系统成本 (元)', fontproperties=chinese_font, fontsize=chinese_size)
        ax1.set_title('梳齿状PSO算法优化迭代曲线', fontproperties=chinese_font, fontsize=chinese_size + 2)
        ax1.grid(True, linestyle='--', alpha=0.7)

        # 方差曲线 - 移除标记点，只使用线条
        ax2.plot(self.iterations, smoothed_variance, 'r-', linewidth=2)
        ax2.set_xlabel('迭代次数', fontproperties=chinese_font, fontsize=chinese_size)
        ax2.set_ylabel('水头均方差', fontproperties=chinese_font, fontsize=chinese_size)
        ax2.grid(True, linestyle='--', alpha=0.7)

        # 设置tick标签字体
        for label in ax1.get_xticklabels() + ax1.get_yticklabels():
            label.set_fontname(english_font)
            label.set_fontsize(english_size)

        for label in ax2.get_xticklabels() + ax2.get_yticklabels():
            label.set_fontname(english_font)
            label.set_fontsize(english_size)

        plt.tight_layout()

        # 如果需要，保存图表
        if self.auto_save:
            plt.savefig('PSO_DAN_2d_curves.png', dpi=300, bbox_inches='tight')

        # 显示图表
        plt.ion()  # 重新开启交互模式以便能打开多个图表
        plt.show(block=False)

    def plot_3d_progress(self):
        """绘制最终的3D进度图"""
        if not self.all_iterations:
            print("没有数据可供绘图")
            return

        # 获取字体配置
        chinese_font = self.font_config['chinese_font']
        english_font = self.font_config['english_font']
        chinese_size = self.font_config['chinese_size']
        english_size = self.font_config['english_size']

        # 确保在非交互模式下创建新图表
        plt.ioff()

        # 创建新图表
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        # 绘制3D散点图
        scatter = ax.scatter(
            self.all_costs,
            self.all_variances,
            self.all_iterations,
            c=self.all_iterations,  # 按迭代次数上色
            cmap='viridis',
            s=50,
            alpha=0.6
        )

        # 添加颜色条
        cbar = fig.colorbar(scatter, ax=ax, pad=0.1)
        cbar.set_label('迭代次数', fontproperties=chinese_font, fontsize=chinese_size)

        # 设置颜色条刻度标签字体
        for label in cbar.ax.get_yticklabels():
            label.set_fontname(english_font)
            label.set_fontsize(english_size)

        # 设置轴标签
        ax.set_xlabel('系统成本 (元)', fontproperties=chinese_font, fontsize=chinese_size)
        ax.set_ylabel('水头均方差', fontproperties=chinese_font, fontsize=chinese_size)
        ax.set_zlabel('迭代次数', fontproperties=chinese_font, fontsize=chinese_size)

        # 设置图表标题
        ax.set_title('梳齿状PSO算法优化3D进度图', fontproperties=chinese_font, fontsize=chinese_size + 2)

        # 设置3D图的tick标签字体
        for label in ax.get_xticklabels() + ax.get_yticklabels() + ax.get_zticklabels():
            label.set_fontname(english_font)
            label.set_fontsize(english_size)

        # 调整视角
        ax.view_init(elev=30, azim=-35)

        # 如果需要，保存图表
        if self.auto_save:
            plt.savefig('PSO_DAN_3d_progress.png', dpi=300, bbox_inches='tight')

        # 显示图表
        plt.ion()  # 重新开启交互模式以便能打开多个图表
        plt.show(block=False)

# 定义PSO算法所需的粒子类
class Particle:
    def __init__(self, dimensions, value_ranges):
        """初始化粒子
        dimensions: 维度数（决策变量数量）
        value_ranges: 每个维度的取值范围列表
        """
        self.dimensions = dimensions
        self.value_ranges = value_ranges

        # 初始化位置和速度
        self.position = self.initialize_position()
        self.velocity = np.zeros(dimensions)

        # 初始化个体最优位置和适应度
        self.best_position = self.position.copy()
        self.fitness = None
        self.best_fitness = None

    def initialize_position(self):
        """初始化位置"""
        position = np.zeros(self.dimensions, dtype=int)
        for i in range(self.dimensions):
            min_val, max_val = self.value_ranges[i]
            position[i] = random.randint(min_val, max_val)
        return position

    def update_velocity(self, global_best_position, w=0.7, c1=1.5, c2=1.5):
        """更新速度
        w: 惯性权重
        c1: 个体认知系数
        c2: 社会认知系数
        """
        r1 = np.random.random(self.dimensions)
        r2 = np.random.random(self.dimensions)

        cognitive_component = c1 * r1 * (self.best_position - self.position)
        social_component = c2 * r2 * (global_best_position - self.position)

        self.velocity = w * self.velocity + cognitive_component + social_component

    def update_position(self):
        """更新位置"""
        # 更新位置
        new_position = self.position + np.round(self.velocity).astype(int)

        # 确保位置在合法范围内
        for i in range(self.dimensions):
            min_val, max_val = self.value_ranges[i]
            new_position[i] = max(min_val, min(max_val, new_position[i]))

        self.position = new_position


def multi_objective_pso(irrigation_system, lgz1, lgz2, swarm_size=100, max_iterations=180, show_plots=True,
                        auto_save=True):
    """多目标PSO优化函数"""
    # 创建跟踪器
    tracker = PSOOptimizationTracker(show_dynamic_plots=show_plots, auto_save=auto_save)

    # 初始化轮灌组配置
    group_count = irrigation_system.initialize_irrigation_groups(lgz1, lgz2)

    # 定义维度和取值范围
    main_pipe_dims = len(irrigation_system.main_pipe) - 1  # 干管段数（不含第0段）
    submain_dims = len(irrigation_system.submains) * 2  # 斗管段数*2（每条斗管分两段）

    dimensions = main_pipe_dims + submain_dims

    # 每个维度的取值范围
    value_ranges = []
    for i in range(main_pipe_dims):
        value_ranges.append((0, len(PIPE_SPECS["main"]["diameters"]) - 1))

    for i in range(submain_dims):
        value_ranges.append((0, len(PIPE_SPECS["submain"]["diameters"]) - 1))

    # 辅助函数 - 判断解A是否支配解B
    def dominates(fitness_a, fitness_b):
        """判断解A是否支配解B（对所有目标都不差，至少有一个更好）"""
        # 对于最小化问题
        not_worse = all(a <= b for a, b in zip(fitness_a, fitness_b))
        better = any(a < b for a, b in zip(fitness_a, fitness_b))
        return not_worse and better

    # 辅助函数 - 选择全局引导者
    def select_leader(pareto_front, particle):
        """从Pareto前沿中选择一个引导者"""
        if not pareto_front:
            return particle

        # 计算拥挤度
        crowding_distances = calculate_crowding_distance(pareto_front)

        # 基于拥挤度进行锦标赛选择
        tournament_size = min(3, len(pareto_front))
        candidates = random.sample(list(zip(pareto_front, crowding_distances)), tournament_size)

        # 选择拥挤度最大的
        return max(candidates, key=lambda x: x[1])[0]

    # 辅助函数 - 计算拥挤度
    def calculate_crowding_distance(pareto_front):
        """计算Pareto前沿中各解的拥挤度"""
        n = len(pareto_front)
        if n <= 2:
            return [float('inf')] * n

        # 初始化拥挤度
        distances = [0.0] * n

        # 对每个目标计算拥挤度
        for obj_idx in range(2):  # 两个目标：成本和压力方差
            # 按目标值排序
            sorted_indices = sorted(range(n), key=lambda i: pareto_front[i].best_fitness[obj_idx])

            # 边界点拥挤度无穷大
            distances[sorted_indices[0]] = float('inf')
            distances[sorted_indices[-1]] = float('inf')

            # 计算中间点的拥挤度
            if n > 2:
                # 目标的范围
                obj_range = (pareto_front[sorted_indices[-1]].best_fitness[obj_idx] -
                             pareto_front[sorted_indices[0]].best_fitness[obj_idx])

                # 如果范围为0，拥挤度不变
                if obj_range > 0:
                    for i in range(1, n - 1):
                        distances[sorted_indices[i]] += (
                                (pareto_front[sorted_indices[i + 1]].best_fitness[obj_idx] -
                                 pareto_front[sorted_indices[i - 1]].best_fitness[obj_idx]) / obj_range
                        )

        return distances

    def adjust_pipe_diameters(position, active_nodes):
        """根据水力要求调整管径"""
        # 将位置数组转换为individual格式进行处理
        individual = position.tolist()

        max_attempts = 10
        attempt = 0

        while attempt < max_attempts:
            # 更新管网配置
            _update_pipe_diameters(irrigation_system, individual)

            # 计算水力特性
            irrigation_system._update_flow_rates(active_nodes)
            irrigation_system._calculate_hydraulics()
            irrigation_system._calculate_pressures()

            pressure_satisfied = True
            diameter_increased = False

            # 检查每个活跃节点的压力
            for node in active_nodes:
                if node <= len(irrigation_system.submains):
                    submain = irrigation_system.submains[node - 1]
                    if submain["inlet_pressure"] < PRESSURE_BASELINE:
                        pressure_satisfied = False

                        # 找到从起点到该节点的路径上最小管径的管段
                        path_segments = list(range(1, node + 1))
                        min_diameter_segment = min(
                            path_segments,
                            key=lambda x: irrigation_system.main_pipe[x]["diameter"]
                        )

                        # 增加管径
                        current_diameter = irrigation_system.main_pipe[min_diameter_segment]["diameter"]
                        larger_diameters = [d for d in PIPE_SPECS["main"]["diameters"] if d > current_diameter]

                        if larger_diameters:
                            new_diameter = min(larger_diameters)
                            irrigation_system.main_pipe[min_diameter_segment]["diameter"] = new_diameter
                            # 更新individual中对应的基因
                            segment_index = min_diameter_segment - 1  # 考虑管段0
                            individual[segment_index] = PIPE_SPECS["main"]["diameters"].index(new_diameter)
                            diameter_increased = True

            if pressure_satisfied:
                return True, np.array(individual)

            if not diameter_increased:
                return False, np.array(individual)

            attempt += 1

        return False, np.array(individual)

    def evaluate(position):
        """评估函数"""
        try:
            position_copy = position.copy()
            total_cost = 0
            pressure_variances = []

            for group_idx in range(group_count):
                active_nodes = irrigation_system.irrigation_groups[group_idx]

                # 调整管径直到满足要求或无法继续调整
                success, adjusted_position = adjust_pipe_diameters(position_copy, active_nodes)
                if success:
                    position_copy = adjusted_position
                else:
                    return float('inf'), float('inf')

                # 计算该组的评估指标
                metrics = irrigation_system.evaluate_group(group_idx)
                total_cost += metrics['cost']
                pressure_variances.append(metrics['pressure_variance'])

            avg_cost = total_cost / group_count
            avg_pressure_variance = np.mean(pressure_variances)

            return avg_cost, avg_pressure_variance

        except Exception as e:
            logging.error(f"评估错误: {str(e)}")
            return float('inf'), float('inf')

    def _update_pipe_diameters(irrigation_system, individual):
        """更新管网直径配置"""
        # 解码个体
        main_indices = individual[:len(irrigation_system.main_pipe) - 1]
        submain_first_indices = individual[len(irrigation_system.main_pipe) - 1:
                                           len(irrigation_system.main_pipe) + len(irrigation_system.submains) - 1]
        submain_second_indices = individual[len(irrigation_system.main_pipe) +
                                            len(irrigation_system.submains) - 1:]

        # 更新干管管径（确保管径递减）
        prev_diameter = None
        for i, index in enumerate(main_indices, start=1):
            available_diameters = [d for d in PIPE_SPECS["main"]["diameters"]
                                   if prev_diameter is None or d <= prev_diameter]
            if not available_diameters:
                return

            normalized_index = min(index, len(available_diameters) - 1)
            diameter = available_diameters[normalized_index]
            irrigation_system.main_pipe[i]["diameter"] = diameter
            prev_diameter = diameter

        # 更新斗管管径
        for i, (first_index, second_index) in enumerate(zip(submain_first_indices,
                                                            submain_second_indices)):
            # 获取连接处干管直径
            main_connection_diameter = irrigation_system.main_pipe[i + 1]["diameter"]

            # 确保斗管第一段管径不大于干管
            available_first_diameters = [d for d in PIPE_SPECS["submain"]["diameters"]
                                         if d <= main_connection_diameter]
            if not available_first_diameters:
                return

            normalized_first_index = min(first_index, len(available_first_diameters) - 1)
            first_diameter = available_first_diameters[normalized_first_index]

            # 确保斗管第二段管径不大于第一段
            available_second_diameters = [d for d in PIPE_SPECS["submain"]["diameters"]
                                          if d <= first_diameter]
            if not available_second_diameters:
                return

            normalized_second_index = min(second_index, len(available_second_diameters) - 1)
            second_diameter = available_second_diameters[normalized_second_index]

            irrigation_system.submains[i]["diameter_first_half"] = first_diameter
            irrigation_system.submains[i]["diameter_second_half"] = second_diameter

    # 初始化粒子群
    swarm = []
    for _ in range(swarm_size):
        swarm.append(Particle(dimensions, value_ranges))

    # 初始化全局最优解
    # 对于多目标PSO，我们维护一个非支配解集作为全局最优解
    pareto_front = []

    # 初始化记录簿
    logbook = tools.Logbook()
    stats = tools.Statistics()
    stats.register("min", np.min, axis=0)
    stats.register("avg", np.mean, axis=0)
    stats.register("std", np.std, axis=0)

    # 初始评估
    for particle in swarm:
        particle.fitness = evaluate(particle.position)
        particle.best_fitness = particle.fitness

    # 初始化Pareto前沿
    for particle in swarm:
        # 只添加有限适应度值的粒子
        if np.all(np.isfinite(particle.fitness)):
            is_dominated = False
            pareto_front_copy = pareto_front.copy()
            for idx, solution in enumerate(pareto_front_copy):
                # 如果现有解支配当前粒子
                if dominates(solution.best_fitness, particle.fitness):
                    is_dominated = True
                    break
                # 如果当前粒子支配现有解
                elif dominates(particle.fitness, solution.best_fitness):
                    pareto_front.remove(solution)

            if not is_dominated:
                # 创建粒子的深拷贝以添加到Pareto前沿
                pareto_particle = Particle(dimensions, value_ranges)
                pareto_particle.position = particle.position.copy()
                pareto_particle.best_position = particle.position.copy()
                pareto_particle.fitness = particle.fitness
                pareto_particle.best_fitness = particle.fitness
                pareto_front.append(pareto_particle)

    # 记录初始统计信息
    fitness_values = [p.fitness for p in swarm if np.all(np.isfinite(p.fitness))]
    if fitness_values:
        record = stats.compile(fitness_values)
        logbook.record(gen=0, **record)

        # 输出初始状态
        logging.info(f"初始化完成，群体大小: {len(swarm)}，Pareto前沿大小: {len(pareto_front)}")
        logging.info(f"Iteration 0: {record}")

        # 更新跟踪器 - 使用新的update_with_front方法
        tracker.update_with_front(0, swarm, pareto_front)

    # 迭代优化
    for iteration in range(1, max_iterations + 1):
        # 对每个粒子
        for particle in swarm:
            # 选择一个全局引导者（这里使用基于拥挤度的选择策略）
            if pareto_front:
                leader = select_leader(pareto_front, particle)
                # 更新速度和位置
                particle.update_velocity(leader.best_position)
                particle.update_position()

                # 评估新位置
                particle.fitness = evaluate(particle.position)

                # 更新个体最优（基于Pareto支配关系）
                if (np.all(np.isfinite(particle.fitness)) and
                        (not np.all(np.isfinite(particle.best_fitness)) or
                         dominates(particle.fitness, particle.best_fitness) or
                         np.array_equal(particle.fitness, particle.best_fitness))):
                    particle.best_position = particle.position.copy()
                    particle.best_fitness = particle.fitness

                # 更新全局Pareto前沿
                if np.all(np.isfinite(particle.fitness)):
                    # 检查当前粒子是否可以加入Pareto前沿
                    is_dominated = False
                    pareto_front_copy = pareto_front.copy()
                    for solution in pareto_front_copy:
                        if dominates(solution.best_fitness, particle.fitness):
                            is_dominated = True
                            break
                        elif dominates(particle.fitness, solution.best_fitness):
                            pareto_front.remove(solution)

                    if not is_dominated:
                        # 创建粒子的深拷贝以添加到Pareto前沿
                        pareto_particle = Particle(dimensions, value_ranges)
                        pareto_particle.position = particle.position.copy()
                        pareto_particle.best_position = particle.position.copy()
                        pareto_particle.fitness = particle.fitness
                        pareto_particle.best_fitness = particle.fitness

                        # 检查是否已存在相同位置的粒子
                        already_exists = False
                        for p in pareto_front:
                            if np.array_equal(p.position, pareto_particle.position):
                                already_exists = True
                                break

                        if not already_exists:
                            pareto_front.append(pareto_particle)

        # 记录本次迭代的统计信息
        fitness_values = [p.fitness for p in swarm if np.all(np.isfinite(p.fitness))]
        if fitness_values:
            record = stats.compile(fitness_values)
            logbook.record(gen=iteration, **record)

            # 更新跟踪器 - 使用新的update_with_front方法
            tracker.update_with_front(iteration, swarm, pareto_front)

            # 每隔一定代数输出进度
            if iteration % 10 == 0:
                logging.info(f"Iteration {iteration}: {record}")
                logging.info(f"当前Pareto前沿大小: {len(pareto_front)}")

    # 创建一个最终的、无重复的帕累托前沿
    final_pareto_front = []
    for particle in pareto_front:
        is_duplicate = False
        for existing in final_pareto_front:
            if np.array_equal(particle.position, existing.position):
                is_duplicate = True
                break
        if not is_duplicate:
            final_pareto_front.append(particle)

    # 再次评估所有解，确保使用完全相同的评估标准
    for particle in final_pareto_front:
        particle.fitness = evaluate(particle.position)
        particle.best_fitness = particle.fitness

    # 确保在返回前进行一次最后的帕累托优化
    non_dominated_front = []
    for particle in final_pareto_front:
        if np.all(np.isfinite(particle.best_fitness)):
            is_dominated = False
            for other in final_pareto_front:
                if dominates(other.best_fitness, particle.best_fitness) and not np.array_equal(other.position,
                                                                                               particle.position):
                    is_dominated = True
                    break
            if not is_dominated:
                non_dominated_front.append(particle)

    # 绘制最终图表
    tracker.finalize_plots()

    # 返回最终优化过的帕累托前沿
    return non_dominated_front, logbook


def multi_objective_optimization(irrigation_system, lgz1, lgz2):
    """多目标优化函数的接口，改为使用PSO算法"""
    return multi_objective_pso(irrigation_system, lgz1, lgz2)


def print_detailed_results(irrigation_system, best_particle, lgz1, lgz2,
                           output_file="optimization_results_PSO_DAN.txt"):
    """优化后的结果输出函数，包含压力分析"""

    with open(output_file, 'w', encoding='utf-8') as f:
        def write_line(text):
            print(text)
            f.write(text + '\n')

        # 输出斗管管径配置
        write_line("\n=== 斗管分段管径配置 ===")
        write_line("节点编号    第一段管径    第二段管径")
        write_line("           (mm)         (mm)")
        write_line("-" * 45)
        for i in range(irrigation_system.node_count):
            submain = irrigation_system.submains[i]
            write_line(f"{i + 1:4d}         {submain['diameter_first_half']:4d}         "
                       f"{submain['diameter_second_half']:4d}")
        write_line("-" * 45)

        # 存储所有压力差值用于组内统计
        all_pressure_margins = []

        # 存储每个组的压力方差，用于计算全局平均方差
        all_group_variances = []

        # 对每个轮灌组输出信息
        for group_idx, nodes in enumerate(irrigation_system.irrigation_groups):
            write_line(f"\n=== 轮灌组 {group_idx + 1} ===")
            write_line(f"启用节点: {nodes}")

            # 更新水力计算
            irrigation_system._update_flow_rates(nodes)
            irrigation_system._calculate_hydraulics()
            irrigation_system._calculate_pressures()

            # 收集该组的压力方差
            group_variance = irrigation_system.calculate_pressure_variance(nodes)
            all_group_variances.append(group_variance)

            # 压力分析数据收集
            group_pressures = []
            write_line("\n管段水力参数表:")
            write_line(
                "编号    后端距起点    管径    段前启用状态    流量       流速     水头损失    段前水头压力    压力富裕")
            write_line("         (m)       (mm)                (m³/s)     (m/s)     (m)          (m)         (m)")
            write_line("-" * 100)

            distance = 0
            for i in range(irrigation_system.node_count + 1):
                segment = irrigation_system.main_pipe[i]
                distance += segment["length"]
                status = "*" if i in nodes else " "

                # 计算压力富裕度
                pressure_margin = segment["pressure"] - PRESSURE_BASELINE if i in nodes else 0
                if i in nodes:
                    group_pressures.append(pressure_margin)
                    all_pressure_margins.append(pressure_margin)

                write_line(f"{i:2d}    {distance:6.1f}    {segment['diameter']:4d}      {status}      "
                           f"{segment['flow_rate']:8.6f}  {segment['velocity']:6.2f}    "
                           f"{segment['head_loss']:6.2f}     {segment['pressure']:6.2f}     "
                           f"{pressure_margin:6.2f}" if i in nodes else
                           f"{i:2d}    {distance:6.1f}    {segment['diameter']:4d}      {status}      "
                           f"{segment['flow_rate']:8.6f}  {segment['velocity']:6.2f}    "
                           f"{segment['head_loss']:6.2f}     {segment['pressure']:6.2f}     -")

            # 计算并输出该组的统计指标
            if group_pressures:
                avg_margin = sum(group_pressures) / len(group_pressures)
                variance = sum((p - avg_margin) ** 2 for p in group_pressures) / len(group_pressures)
                std_dev = variance ** 0.5

                write_line("\n该轮灌组压力统计:")
                write_line(f"平均压力富裕程度: {avg_margin:.2f} m")
                write_line(f"压力均方差: {std_dev:.2f}")

            write_line("\n注: * 表示该管段对应节点在当前轮灌组中启用")
            write_line("-" * 100)

        # 计算全局统计指标
        # 使用与优化算法相同的方法计算全局压力均方差：所有轮灌组方差的平均值
        global_std_dev = sum(all_group_variances) / len(all_group_variances) if all_group_variances else 0

        # 全局平均压力富裕程度仍然可以用所有收集的压力值计算
        if all_pressure_margins:
            global_avg_margin = sum(all_pressure_margins) / len(all_pressure_margins)
        else:
            global_avg_margin = 0

        write_line("\n=== 全局压力统计 ===")
        write_line(f"系统整体平均压力富裕程度: {global_avg_margin:.2f} m")
        write_line(f"系统整体压力均方差: {global_std_dev:.2f}")
        write_line("-" * 45)

        # 输出系统经济指标
        total_cost = irrigation_system.get_system_cost()
        total_long = sum(seg['length'] for seg in irrigation_system.main_pipe)
        irrigation_area = (irrigation_system.node_count + 1) * irrigation_system.node_spacing * DEFAULT_SUBMAIN_LENGTH
        change_area = irrigation_area / (2000 / 3)
        cost_per_area = total_cost / change_area

        write_line("\n=== 系统总体信息 ===")
        write_line(f"系统总成本: {total_cost:.2f} 元")
        write_line(f"灌溉面积: {change_area:.1f} 亩")
        write_line(f"单位面积成本: {cost_per_area:.2f} 元/亩")
        write_line(f"总轮灌组数: {len(irrigation_system.irrigation_groups)}")
        write_line(f"管网总长度: {total_long:.1f} m")


def visualize_pareto_front(pareto_front_particles):
    """可视化Pareto前沿"""
    try:
        if not pareto_front_particles:
            print("没有可视化的解集")
            return

        # 获取字体配置
        font_config = configure_fonts()
        chinese_font = font_config['chinese_font']
        english_font = font_config['english_font']
        chinese_size = font_config['chinese_size']
        english_size = font_config['english_size']

        costs = []
        variances = []
        for particle in pareto_front_particles:
            if np.all(np.isfinite(particle.best_fitness)):
                costs.append(particle.best_fitness[0])
                variances.append(particle.best_fitness[1])

        if not costs or not variances:
            print("没有有效的解集数据可供可视化")
            return

        plt.figure(figsize=(10, 6), dpi=100)
        plt.scatter(costs, variances, c='blue', marker='o', s=50, alpha=0.6, label='Pareto解')

        plt.title('多目标梳齿PSO管网优化Pareto前沿', fontproperties=chinese_font, fontsize=chinese_size + 2, pad=15)
        plt.xlabel('系统成本', fontproperties=chinese_font, fontsize=chinese_size)
        plt.ylabel('压力方差', fontproperties=chinese_font, fontsize=chinese_size)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.ticklabel_format(style='sci', scilimits=(-2, 3), axis='both')

        # 设置图例字体
        legend = plt.legend(loc='upper right')
        for text in legend.get_texts():
            text.set_fontproperties(chinese_font)
            text.set_fontsize(chinese_size)

        # 设置tick标签字体
        for label in plt.gca().get_xticklabels() + plt.gca().get_yticklabels():
            label.set_fontname(english_font)
            label.set_fontsize(english_size)

        plt.tight_layout()
        plt.savefig('PSO_DAN_pareto_front.png', dpi=300, bbox_inches='tight')
    except Exception as e:
        logging.error(f"可视化过程中发生错误: {str(e)}")
        print(f"可视化失败: {str(e)}")


def select_best_solution_by_marginal_improvement(solutions, max_variance_threshold=5.0):
    """
    使用科学的多目标选择方法，引入更合理的平衡机制和平滑处理

    特点:
    1. 保留水头均方差阈值筛选
    2. 使用科学的效用函数进行多目标平衡
    3. 增加数值计算的稳定性
    4. 为大规模数据集使用L方法
    """
    # 确保有解可选
    if not solutions:
        return None

    # 首先筛选出水头均方差低于阈值的解
    valid_solutions = [sol for sol in solutions if sol.best_fitness[1] <= max_variance_threshold]

    # 如果没有满足条件的解，从原始解集中选择方差最小的
    if not valid_solutions:
        return min(solutions, key=lambda x: x.best_fitness[1])

    # 按成本升序排序
    sorted_solutions = sorted(valid_solutions, key=lambda particle: particle.best_fitness[0])

    # 如果解的数量较少，使用简单的效用函数
    if len(sorted_solutions) < 5:
        return sorted_solutions[0]

    # 对于较大的前沿，使用科学的效用函数方法
    try:
        # 提取成本和方差值
        costs = np.array([sol.best_fitness[0] for sol in sorted_solutions])
        variances = np.array([sol.best_fitness[1] for sol in sorted_solutions])

        # 正规化数据到[0,1]范围
        min_cost = np.min(costs)
        max_cost = np.max(costs)
        min_var = np.min(variances)
        max_var = np.max(variances)

        # 确保数值稳定性
        cost_range = max(max_cost - min_cost, 1e-6)
        var_range = max(max_var - min_var, 1e-6)

        norm_costs = (costs - min_cost) / cost_range
        norm_vars = (variances - min_var) / var_range

        # 计算每个解的综合效用值(需要权衡成本和方差)
        # 成本权重高于方差，比例约为4:1
        utility_values = 0.8 * (1 - norm_costs) + 0.2 * (1 - norm_vars)

        # 寻找效用最大的解
        best_idx = np.argmax(utility_values)

        # 对于足够大的数据集，还可以尝试L方法，但仅在方差较大时
        if len(sorted_solutions) >= 10 and var_range > 0.5:
            try:
                # 计算"拐点"，即帕累托前沿的转折处
                # L方法实现
                best_l_error = float('inf')
                best_l_idx = 0

                for i in range(2, len(sorted_solutions) - 2):
                    # 只计算数值稳定的部分
                    if np.std(norm_costs[:i]) < 1e-4 or np.std(norm_costs[i:]) < 1e-4:
                        continue

                    try:
                        # 前半部分线性拟合
                        slope1, intercept1 = np.polyfit(norm_costs[:i], norm_vars[:i], 1)
                        pred1 = slope1 * norm_costs[:i] + intercept1
                        error1 = np.sum((norm_vars[:i] - pred1) ** 2)

                        # 后半部分线性拟合
                        slope2, intercept2 = np.polyfit(norm_costs[i:], norm_vars[i:], 1)
                        pred2 = slope2 * norm_costs[i:] + intercept2
                        error2 = np.sum((norm_vars[i:] - pred2) ** 2)

                        # 总误差
                        total_error = error1 + error2

                        if total_error < best_l_error:
                            best_l_error = total_error
                            best_l_idx = i
                    except:
                        continue

                # 如果L方法成功找到拐点，使用它作为一个参考点
                if best_l_error < float('inf'):
                    # 计算拐点与效用最大点的距离
                    distance = abs(best_l_idx - best_idx)

                    # 如果两点很接近，优先选择效用最大点
                    # 如果距离较远，综合考虑两点
                    if distance <= 3:
                        # 两点很接近，使用效用最大点
                        return sorted_solutions[best_idx]
                    else:
                        # 在拐点和效用最大点之间选择一个平衡点
                        # 更偏向于效用最大点(70% 效用, 30% 拐点)
                        balanced_idx = int(0.7 * best_idx + 0.3 * best_l_idx)
                        return sorted_solutions[balanced_idx]
            except:
                # L方法失败，使用效用最大点
                return sorted_solutions[best_idx]

        # 返回效用最大的解
        return sorted_solutions[best_idx]

    except Exception as e:
        # 如果计算过程出错，回退到最简单的策略
        print(f"效用计算出错，使用成本最低解: {e}")
        return sorted_solutions[0]


def main():
    """主函数"""
    try:
        # 初始化字体配置
        configure_fonts()
        # 创建灌溉系统
        irrigation_system = IrrigationSystem(
            node_count=23,
            rukoushuitou=49.62
        )

        # 设置轮灌参数
        best_lgz1, best_lgz2 = 8, 4
        logging.info("开始进行多目标PSO优化...")

        # 执行优化
        start_time = time.time()
        pareto_front, logbook = multi_objective_pso(irrigation_system, best_lgz1, best_lgz2)
        end_time = time.time()

        logging.info(f"优化完成，耗时: {end_time - start_time:.2f}秒")

        # 输出结果
        if pareto_front:
            valid_solutions = [particle for particle in pareto_front if np.all(np.isfinite(particle.best_fitness))]
            if valid_solutions:
                # 选择最优解
                best_solution = select_best_solution_by_marginal_improvement(valid_solutions)
                # 使用相同的最优解进行详细结果输出
                print_detailed_results(irrigation_system, best_solution, best_lgz1, best_lgz2)
                # 可视化Pareto前沿，并标记出相同的最优解
                visualize_pareto_front(pareto_front)

                logging.info("结果已保存并可视化完成")
            else:
                logging.error("未找到有效的解决方案")
        else:
            logging.error("多目标优化未能产生有效的Pareto前沿")
        auto_save = 0
        # 保存所有打开的图表,0关，1开
        if auto_save == 1:
            figures = [plt.figure(i) for i in plt.get_fignums()]
            for i, fig in enumerate(figures):
                try:
                    fig.savefig(f'PSO_DAN_result_fig{i}.png', dpi=300, bbox_inches='tight')
                    logging.info(f"图表{i}已保存为 PSO_DAN_result_fig{i}.png")
                except Exception as e:
                    logging.warning(f"保存图表{i}时出错: {e}")

        # 显示程序已完成的消息
        print("=========================================================")
        print("程序计算已完成，所有图表窗口将保持打开状态")
        print("关闭图表窗口不会影响结果，可随时查看保存的PNG图像文件")
        print("=========================================================")

        # 关键修改：使用源程序中的方式保持窗口打开
        # 关闭交互模式，然后显示窗口
        plt.ioff()  # 关闭交互模式
        plt.show()  # 阻塞直到所有窗口关闭

    except Exception as e:
        logging.error(f"程序执行出错: {str(e)}")
        raise


if __name__ == "__main__":
    # 设置随机种子以确保结果可重复
    random.seed(42)
    np.random.seed(42)
    # 执行主程序
    main()
