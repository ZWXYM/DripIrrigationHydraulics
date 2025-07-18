import json
import math
import os
import random
import logging
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from deap import base, creator, tools, algorithms
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
PRESSURE_BASELINE = 27.51  # 基准压力值
MAX_VARIANCE = 7.0  # 水头均方差上限

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
    def __init__(self, node_count, frist_pressure, frist_diameter, node_spacing=DEFAULT_NODE_SPACING,
                 first_segment_length=DEFAULT_FIRST_SEGMENT_LENGTH,
                 submain_length=DEFAULT_SUBMAIN_LENGTH,
                 lateral_layout="single"):
        """初始化灌溉系统"""
        self.node_count = node_count
        self.node_spacing = node_spacing
        self.first_segment_length = first_segment_length
        self.submain_length = submain_length
        self.lateral_layout = lateral_layout
        self.frist_pressure = frist_pressure
        self.frist_diameter = frist_diameter

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
        """
        修改说明：
        此版本为【方案一：普通设计】的【修正版】，采用您提出的【阶梯式流量递减模型】。
        1.  入口总流量由 `submain_flow * lgz1` 决定。
        2.  流量不再是逐点递减，而是每隔 `step_size` 个管段才减少一个 `submain_flow`。
            `step_size` 由 `floor(node_count / lgz1)` 计算得出。
        3.  参数 `active_nodes` 在此方案中仍然被忽略。
        4.  为简化模型，此函数仅计算干管流量。斗管和农管流量在此模型下可视为0，因为其水头损失已包含在“基准水头”的计算中，作为压力约束的目标值。
        """
        # 清空所有流量
        for pipe in self.main_pipe:
            pipe["flow_rate"] = 0
        for submain in self.submains:
            submain["flow_rate"] = 0
        for lateral in self.laterals:
            lateral["flow_rate"] = 0

        # 1. 计算基础的单位斗管流量 (submain_flow)
        drippers_per_line = math.ceil(DEFAULT_DRIP_LINE_LENGTH / DRIPPER_SPACING)
        single_dripper_flow = DEFAULT_DRIPPER_FLOW_RATE / 3600000
        lateral_flow = drippers_per_line * single_dripper_flow * 100
        lgz2_val = self.lgz2 if self.lgz2 is not None else 4
        submain_flow = lateral_flow * lgz2_val

        lgz1_val = self.lgz1 if self.lgz1 is not None else self.node_count
        if lgz1_val <= 1:  # 如果总流量单元小于等于1，则不递减
            total_flow = submain_flow * lgz1_val
            for i in range(len(self.main_pipe)):
                self.main_pipe[i]["flow_rate"] = total_flow
            return

        # 2. 计算入口总流量
        total_flow = submain_flow * lgz1_val

        # 3. 【修正】基于 lgz1 - 1 个“出口”来计算流量递减的步长
        # 我们有 lgz1-1 个流量递减点，均匀分布在 node_count 个节点上
        num_drops = lgz1_val - 1
        # 修正步长计算，确保更均匀的分布
        step_size = self.node_count / num_drops if num_drops > 0 else float('inf')

        # 4. 实施阶梯式流量分配
        current_flow = total_flow

        # 使用一个变量来跟踪理论上的下一个递减点的位置
        next_drop_point = step_size

        for i in range(len(self.main_pipe)):
            self.main_pipe[i]["flow_rate"] = current_flow

            # 管段0是入口，从管段1（节点1之后）开始判断是否递减
            if i > 0:
                # 当管段索引 i 越过或等于下一个递减点时，减少流量
                # 使用一个小的容差(1e-9)来处理浮点数比较
                if i + 1e-9 >= next_drop_point:
                    current_flow -= submain_flow
                    current_flow = max(0, current_flow)
                    # 更新下一个递减点的位置
                    next_drop_point += step_size

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
                segment["pressure"] = self.frist_pressure
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
                cost += ((submain["length"] / 2) - DEFAULT_DRIP_LINE_LENGTH) * price_lookup["submain"][submain["diameter_second_half"]]

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
        """检查压力要求满足情况（使用节点基准水头）"""
        # 获取各节点的基准水头
        baseline_heads_data = self.calculate_node_baseline_heads()
        baseline_heads = {data['node']: data['baseline_head'] for data in baseline_heads_data}

        # 检查每个活跃节点的压力是否满足其基准水头需求
        for i, submain in enumerate(self.submains):
            if submain["flow_rate"] > 0:  # 只检查活跃的斗管
                node_number = i + 1
                required_pressure = baseline_heads.get(node_number, DEFAULT_DRIP_LINE_INLET_PRESSURE)
                actual_pressure = submain["inlet_pressure"]

                if actual_pressure < required_pressure:
                    return False

        return True

    def _create_main_pipe(self):
        """创建干管"""
        segments = []
        for i in range(self.node_count + 1):
            segments.append({
                "index": i,
                "length": self.first_segment_length if i == 0 else self.node_spacing,
                "diameter": self.frist_diameter,
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
            lateral_count = math.ceil(submain["length"] / (DEFAULT_DRIP_LINE_LENGTH * 2)) * 2

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
        """
        统一计算水头均方差的标准方法 - 【方案一修正版】
        修改说明：
        - 移除了 `and self.submains[node - 1]["flow_rate"] > 0` 的判断。
        - 在方案一中，所有节点都被视为“激活”来参与压力评估，无论其斗管流量属性是否为0。
          这样可以正确计算出整个管网在设计工况下的压力均匀性。
        """
        baseline_heads_data = self.calculate_node_baseline_heads()
        baseline_heads = {data['node']: data['baseline_head'] for data in baseline_heads_data}

        pressure_margins = []
        for node in active_nodes:
            if node <= len(self.submains):
                # 【修正】不再检查 submain 的 flow_rate
                actual_pressure = self.submains[node - 1]["inlet_pressure"]
                required_pressure = baseline_heads.get(node, DEFAULT_DRIP_LINE_INLET_PRESSURE)
                pressure_margin = actual_pressure - required_pressure
                pressure_margins.append(pressure_margin)

        if not pressure_margins:
            return 0

        avg_margin = sum(pressure_margins) / len(pressure_margins)
        variance = sum((p - avg_margin) ** 2 for p in pressure_margins) / len(pressure_margins)
        return variance ** 0.5

    def calculate_node_baseline_heads(self):
        """计算每个节点真正需要的基准水头（从滴灌带入口倒推）"""
        # 计算基础流量
        drippers_per_line = math.ceil(DEFAULT_DRIP_LINE_LENGTH / DRIPPER_SPACING)
        single_dripper_flow = DEFAULT_DRIPPER_FLOW_RATE / 3600000  # 转换为m³/s
        lateral_flow = drippers_per_line * single_dripper_flow * 100  # 一条农管的流量

        # 如果lgz2没有设置，使用默认值4
        lgz2 = self.lgz2 if self.lgz2 is not None else 4
        submain_flow = lateral_flow * lgz2  # 一条斗管的流量

        node_baseline_heads = []

        for i in range(self.node_count):
            # 滴灌带入口水头
            drip_inlet_head = DEFAULT_DRIP_LINE_INLET_PRESSURE  # 10米

            # 计算农管水头损失
            # 从laterals中获取农管直径，如果没有则使用默认值90mm
            if i < len(self.laterals) and len(self.laterals) > 0:
                lateral_diameter = self.laterals[i * lgz2]["diameter"]  # 取该斗管对应的第一条农管直径
            else:
                lateral_diameter = 90  # 默认直径

            lateral_length = DEFAULT_LATERAL_LENGTH
            lateral_head_loss = pressure_loss(lateral_diameter, lateral_length, lateral_flow)

            # 计算斗管水头损失（分两段计算）
            # 第一段：使用160mm直径
            submain_first_diameter = 160
            submain_first_length = self.submain_length / 2
            submain_first_head_loss = pressure_loss(submain_first_diameter, submain_first_length, submain_flow)

            # 第二段：使用160mm直径（根据代码中的默认配置）
            submain_second_diameter = 160
            submain_second_length = self.submain_length / 2
            submain_second_head_loss = pressure_loss(submain_second_diameter, submain_second_length, submain_flow)

            # 斗管总水头损失
            submain_total_head_loss = submain_first_head_loss + submain_second_head_loss

            # 计算该节点需要的基准水头
            baseline_head = drip_inlet_head + lateral_head_loss + submain_total_head_loss
            node_baseline_heads.append({
                'node': i + 1,
                'drip_inlet_head': drip_inlet_head,
                'lateral_head_loss': lateral_head_loss,
                'submain_head_loss': submain_total_head_loss,
                'baseline_head': baseline_head,
                'lateral_flow': lateral_flow,
                'submain_flow': submain_flow
            })

        return node_baseline_heads

    def print_node_baseline_heads(self):
        """输出每个节点的基准水头计算结果"""
        baseline_heads = self.calculate_node_baseline_heads()

        print("\n" + "=" * 80)
        print("各节点基准水头计算结果（从滴灌带入口倒推）")
        print("=" * 80)
        print(
            f"{'节点':<4} {'滴灌带入口':<8} {'农管水损':<8} {'斗管水损':<8} {'基准水头':<8} {'农管流量':<10} {'斗管流量':<10}")
        print(f"{'编号':<4} {'水头(m)':<8} {'(m)':<8} {'(m)':<8} {'需求(m)':<8} {'(m³/s)':<10} {'(m³/s)':<10}")
        print("-" * 80)

        for data in baseline_heads:
            print(f"{data['node']:<4} {data['drip_inlet_head']:<8.2f} {data['lateral_head_loss']:<8.3f} "
                  f"{data['submain_head_loss']:<8.3f} {data['baseline_head']:<8.2f} "
                  f"{data['lateral_flow']:<10.6f} {data['submain_flow']:<10.6f}")

        print("-" * 80)
        print(f"说明：")
        print(f"- 滴灌带入口水头：{DEFAULT_DRIP_LINE_INLET_PRESSURE}m（固定值）")
        print(f"- 农管长度：{DEFAULT_LATERAL_LENGTH}m")
        print(f"- 斗管长度：{DEFAULT_SUBMAIN_LENGTH}m（分两段，每段{DEFAULT_SUBMAIN_LENGTH / 2}m）")
        print(f"- 斗管管径：160mm/160mm（两段均为160mm）")
        print(f"- 基准水头 = 滴灌带入口水头 + 农管水头损失 + 斗管水头损失")
        print("=" * 80)


class NSGAOptimizationTracker:
    def __init__(self, show_dynamic_plots=False, auto_save=True, enable_smoothing=False):
        self.generations = []
        self.best_costs = []
        self.best_variances = []
        self.all_costs = []
        self.all_variances = []
        self.all_generations = []

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
        self.ax1.set_title('梳齿状NSGA-II算法优化迭代曲线', fontproperties=chinese_font, fontsize=chinese_size + 2)
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
        self.ax_3d.set_title('梳齿状NSGA-II算法优化3D进度图', fontproperties=chinese_font, fontsize=chinese_size + 2)
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
    def update_with_front(self, generation, population, pareto_front):
        """接收直接传入的帕累托前沿，与最终结果使用相同的选择方法，增加平滑处理"""
        self.generations.append(generation)

        # 获取有效解用于3D图
        valid_solutions = [ind for ind in population if np.all(np.isfinite(ind.fitness.values))]

        if pareto_front and len(pareto_front) > 0:
            # 直接使用与主函数相同的选择策略
            valid_front = [ind for ind in pareto_front if np.all(np.isfinite(ind.fitness.values))]

            if valid_front:
                # 使用相同的选择函数和水头均方差阈值
                # 从外部模块导入select_best_solution_by_marginal_improvement函数
                import __main__
                if hasattr(__main__, 'select_best_solution_by_marginal_improvement'):
                    # 使用与主程序相同的函数和参数
                    best_solution = __main__.select_best_solution_by_marginal_improvement(valid_front,
                                                                                          max_variance_threshold=MAX_VARIANCE)
                else:
                    # 如果无法导入，使用内部实现
                    best_solution = self._select_best_solution(valid_front)

                # 平滑处理：如果之前已有数据，检测变化是否过大
                if self.best_costs and self.best_variances:
                    prev_cost = self.best_costs[-1]
                    prev_variance = self.best_variances[-1]

                    curr_cost = best_solution.fitness.values[0]
                    curr_variance = best_solution.fitness.values[1]

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
                    if generation > 20 and (cost_change_ratio > 0.15 or var_change_ratio > 0.15):
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
                    self.best_costs.append(best_solution.fitness.values[0])
                    self.best_variances.append(best_solution.fitness.values[1])
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
            self.all_costs.append(solution.fitness.values[0])
            self.all_variances.append(solution.fitness.values[1])
            self.all_generations.append(generation)

        # 如果启用了动态图表，更新图表
        if self.show_dynamic_plots and generation % 20 == 0:
            try:
                self._update_plots()
            except Exception as e:
                print(f"图表更新时出错: {e}")

    # 保留原来的update方法，但使用新的帕累托逻辑
    def update(self, generation, population):
        """更新跟踪器的数据（为向下兼容保留，但内部使用新逻辑）"""
        # 获取有效解
        valid_solutions = [ind for ind in population if np.all(np.isfinite(ind.fitness.values))]

        if valid_solutions:
            # 从DEAP工具库导入sortNondominated
            from deap import tools
            # 提取帕累托前沿
            pareto_front = tools.sortNondominated(valid_solutions, len(valid_solutions), first_front_only=True)[0]
            # 使用新方法
            self.update_with_front(generation, population, pareto_front)

    # 内部辅助方法 - 确保与全局函数完全一致
    def _select_best_solution(self, solutions, max_variance_threshold=MAX_VARIANCE):
        """
        内部L方法选择函数 - 保持与全局函数一致，确保在跟踪器无法导入全局函数时可以使用
        """
        # 确保有解可选
        if not solutions:
            return None

        # 首先筛选出水头均方差低于阈值的解
        valid_solutions = [sol for sol in solutions if sol.fitness.values[1] <= max_variance_threshold]

        # 如果没有满足条件的解，从原始解集中选择方差最小的
        if not valid_solutions:
            return min(solutions, key=lambda x: x.fitness.values[1])

        # 按成本升序排序
        sorted_solutions = sorted(valid_solutions, key=lambda ind: ind.fitness.values[0])

        # 如果只有几个解，使用简单策略
        if len(sorted_solutions) <= 3:
            # 如果只有1-3个解，返回成本最低的
            return sorted_solutions[0]

        # 提取成本和方差值
        costs = [sol.fitness.values[0] for sol in sorted_solutions]
        variances = [sol.fitness.values[1] for sol in sorted_solutions]

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
        if not self.generations:
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

            if self.enable_smoothing and len(self.generations) > 5:
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
            self.line1.set_data(self.generations, smoothed_cost)
            self.line2.set_data(self.generations, smoothed_variance)

            # 调整坐标轴范围
            self.ax1.relim()
            self.ax1.autoscale_view()
            self.ax2.relim()
            self.ax2.autoscale_view()
            # 更新标题，反映这是最优个体的迭代曲线
            self.ax1.set_title('梳齿状NSGA-II算法优化最优个体迭代曲线', fontproperties=chinese_font,
                               fontsize=chinese_size + 2)

            # 更新3D图表
            self.ax_3d.clear()
            self.scatter = self.ax_3d.scatter(
                self.all_costs,
                self.all_variances,
                self.all_generations,
                c=self.all_generations,
                cmap='viridis',
                s=50,
                alpha=0.6
            )

            self.ax_3d.view_init(elev=30, azim=-35)  # 默认30，45
            self.ax_3d.set_xlabel('系统成本 (元)', fontproperties=chinese_font, fontsize=chinese_size)
            self.ax_3d.set_ylabel('水头均方差', fontproperties=chinese_font, fontsize=chinese_size)
            self.ax_3d.set_zlabel('迭代次数', fontproperties=chinese_font, fontsize=chinese_size)
            self.ax_3d.set_title('梳齿状NSGA-II算法优化3D进度图', fontproperties=chinese_font, fontsize=chinese_size + 2)

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

            if self.enable_smoothing and len(self.generations) > 5:
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
            self.line1.set_data(self.generations, smoothed_cost)
            self.line2.set_data(self.generations, smoothed_variance)

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
                self.all_generations,
                c=self.all_generations,
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

            self.ax_3d.view_init(elev=30, azim=-35)
            self.ax_3d.set_xlabel('系统成本 (元)', fontproperties=chinese_font, fontsize=chinese_size)
            self.ax_3d.set_ylabel('水头均方差', fontproperties=chinese_font, fontsize=chinese_size)
            self.ax_3d.set_zlabel('迭代次数', fontproperties=chinese_font, fontsize=chinese_size)
            self.ax_3d.set_title('梳齿状NSGA-II算法优化3D进度图', fontproperties=chinese_font, fontsize=chinese_size + 2)

            # 设置3D图的tick标签字体
            for label in self.ax_3d.get_xticklabels() + self.ax_3d.get_yticklabels() + self.ax_3d.get_zticklabels():
                label.set_fontname(english_font)
                label.set_fontsize(english_size)

            # 如果需要，保存图表
            if self.auto_save:
                self.fig_2d.savefig('NSGA_DAN_CHUSHI_2d_curves.png', dpi=300, bbox_inches='tight')
                self.fig_3d.savefig('NSGA_DAN_CHUSHI_3d_progress.png', dpi=300, bbox_inches='tight')

            # 刷新图表
            self.fig_2d.canvas.draw()
            self.fig_3d.canvas.draw()

        except Exception as e:
            print(f"最终图表更新时出错: {e}")

    def plot_2d_curves(self):
        """绘制最终的2D迭代曲线"""
        if not self.generations:
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

        if self.enable_smoothing and len(self.generations) > 5:
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
        ax1.plot(self.generations, smoothed_cost, 'b-', linewidth=2)
        ax1.set_ylabel('系统成本 (元)', fontproperties=chinese_font, fontsize=chinese_size)
        ax1.set_title('梳齿状NSGA-II算法优化迭代曲线', fontproperties=chinese_font, fontsize=chinese_size + 2)
        ax1.grid(True, linestyle='--', alpha=0.7)

        # 方差曲线 - 移除标记点，只使用线条
        ax2.plot(self.generations, smoothed_variance, 'r-', linewidth=2)
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
            plt.savefig('NSGA_DAN_CHUSHI_2d_curves.png', dpi=300, bbox_inches='tight')

        # 显示图表
        plt.ion()  # 重新开启交互模式以便能打开多个图表
        plt.show(block=False)

    def plot_3d_progress(self):
        """绘制最终的3D进度图"""
        if not self.all_generations:
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
            self.all_generations,
            c=self.all_generations,  # 按迭代次数上色
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
        ax.set_title('梳齿状NSGA-II算法优化3D进度图', fontproperties=chinese_font, fontsize=chinese_size + 2)

        # 设置3D图的tick标签字体
        for label in ax.get_xticklabels() + ax.get_yticklabels() + ax.get_zticklabels():
            label.set_fontname(english_font)
            label.set_fontsize(english_size)

        # 调整视角
        ax.view_init(elev=30, azim=-35)

        # 如果需要，保存图表
        if self.auto_save:
            plt.savefig('NSGA_DAN_CHUSHI_3d_progress.png', dpi=300, bbox_inches='tight')

        # 显示图表
        plt.ion()  # 重新开启交互模式以便能打开多个图表
        plt.show(block=False)


def multi_objective_optimization(irrigation_system, lgz1, lgz2, swarm_size, max_iterations, show_plots, auto_save):
    """多目标优化函数"""

    if hasattr(creator, "FitnessMulti"):
        del creator.FitnessMulti
    if hasattr(creator, "Individual"):
        del creator.Individual

    # 创建跟踪器
    tracker = NSGAOptimizationTracker(show_dynamic_plots=show_plots, auto_save=auto_save, enable_smoothing=False)

    # 初始化轮灌组配置
    group_count = irrigation_system.initialize_irrigation_groups(lgz1, lgz2)

    def adjust_pipe_diameters(individual, active_nodes):
        """根据水力要求调整管径"""
        # 获取各节点的基准水头
        baseline_heads_data = irrigation_system.calculate_node_baseline_heads()
        baseline_heads = {data['node']: data['baseline_head'] for data in baseline_heads_data}

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
                    required_pressure = baseline_heads.get(node, DEFAULT_DRIP_LINE_INLET_PRESSURE)

                    # 修改这里：使用节点基准水头而不是PRESSURE_BASELINE
                    if submain["inlet_pressure"] < required_pressure:
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
                # 关键修复：从当前irrigation_system状态重新编码individual，确保完全一致
                corrected_individual = encode_pipe_diameters_from_system(irrigation_system)
                # 更新individual的所有基因
                individual[:] = corrected_individual
                return True, individual

            if not diameter_increased:
                return False, individual

            attempt += 1

        return False, individual

    def evaluate(individual):
        """
        评估函数 - 适用于【方案一：普通设计】
        修改说明：
        此版本不再遍历轮灌组。它将所有节点视为一个整体进行评估。
        1. 调用 adjust_pipe_diameters 调整管径，确保所有节点（尤其是最后一个）满足压力。
        2. 计算一次总成本和一次全系统的压力方差。
        """
        try:
            # 在此方案中，active_nodes 包含所有节点
            all_nodes = list(range(1, irrigation_system.node_count + 1))

            # 创建 individual 的副本用于调整
            working_individual = list(individual)

            # 调整管径直到满足要求
            # 注意：adjust_pipe_diameters 函数也需要感知到这是“普通设计”模式
            success, adjusted_individual = adjust_pipe_diameters(working_individual, all_nodes)

            if not success:
                return float('inf'), float('inf')

            # 关键：将最终的正确基因编码更新到原 individual 中
            individual[:] = adjusted_individual

            # 使用调整后的系统状态计算最终指标
            irrigation_system._update_flow_rates(all_nodes)
            irrigation_system._calculate_hydraulics()
            irrigation_system._calculate_pressures()

            cost = irrigation_system.get_system_cost()
            # 压力方差计算现在基于所有节点
            pressure_variance = irrigation_system.calculate_pressure_variance(all_nodes)

            return cost, pressure_variance

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
        prev_diameter = irrigation_system.main_pipe[0]["diameter"]
        for i, index in enumerate(main_indices, start=1):
            available_diameters = [d for d in PIPE_SPECS["main"]["diameters"]
                                   if d <= prev_diameter]
            if not available_diameters:
                available_diameters = [min(PIPE_SPECS["main"]["diameters"])]

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
                available_first_diameters = [min(PIPE_SPECS["submain"]["diameters"])]

            normalized_first_index = min(first_index, len(available_first_diameters) - 1)
            first_diameter = available_first_diameters[normalized_first_index]

            # 确保斗管第二段管径不大于第一段
            available_second_diameters = [d for d in PIPE_SPECS["submain"]["diameters"]
                                          if d <= first_diameter]
            if not available_second_diameters:
                available_second_diameters = [min(PIPE_SPECS["submain"]["diameters"])]

            normalized_second_index = min(second_index, len(available_second_diameters) - 1)
            second_diameter = available_second_diameters[normalized_second_index]

            irrigation_system.submains[i]["diameter_first_half"] = first_diameter
            irrigation_system.submains[i]["diameter_second_half"] = second_diameter

    # 配置遗传算法
    creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0))
    creator.create("Individual", list, fitness=creator.FitnessMulti)

    toolbox = base.Toolbox()

    # 创建个体生成器
    def create_individual():
        main_pipe_options = len(PIPE_SPECS["main"]["diameters"]) - 1
        submain_pipe_options = len(PIPE_SPECS["submain"]["diameters"]) - 1
        main_genes = [random.randint(0, main_pipe_options)
                      for _ in range(len(irrigation_system.main_pipe) - 1)]
        submain_genes = [random.randint(0, submain_pipe_options)
                         for _ in range(len(irrigation_system.submains) * 2)]
        return creator.Individual(main_genes + submain_genes)

    toolbox.register("individual", create_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutUniformInt,
                     low=0,
                     up=max(len(PIPE_SPECS['main']['diameters']) - 1,
                            len(PIPE_SPECS['submain']['diameters']) - 1),
                     indpb=0.1)
    toolbox.register("select", tools.selNSGA2)

    # 执行优化
    population = toolbox.population(n=swarm_size)
    stats = tools.Statistics(key=lambda ind: ind.fitness.values)
    stats.register("min", np.min, axis=0)
    stats.register("avg", np.mean, axis=0)
    stats.register("std", np.std, axis=0)

    logbook = tools.Logbook()

    # 评估初始种群
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    # 记录初始统计信息
    record = stats.compile(population)
    logbook.record(gen=0, **record)

    # 提取帕累托前沿并更新跟踪器
    current_front = tools.sortNondominated(population, len(population), first_front_only=True)[0]
    tracker.update_with_front(0, population, current_front)

    # 演化过程
    for gen in range(1, max_iterations):
        offspring = algorithms.varOr(population, toolbox, 100, 0.7, 0.2)
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        population = toolbox.select(population + offspring, 100)
        record = stats.compile(population)
        logbook.record(gen=gen, **record)

        # 提取帕累托前沿并更新跟踪器
        current_front = tools.sortNondominated(population, len(population), first_front_only=True)[0]
        tracker.update_with_front(gen, population, current_front)

        # 输出进度
        if gen % 10 == 0:
            logging.info(f"Generation {gen}: {record}")

    # 绘制图表
    tracker.finalize_plots()

    # 在返回前计算并保存帕累托前沿
    try:
        # 首先获取最终的帕累托前沿
        final_pareto_front = tools.sortNondominated(population, len(population), first_front_only=True)[0]

        # 验证帕累托前沿的正确性，只保留通过验证的个体
        print("正在验证最终帕累托前沿的正确性...")
        verified_individuals = []
        failed_individuals = []

        for i, ind in enumerate(final_pareto_front):
            if verify_individual_correctness(ind, irrigation_system, group_count):
                verified_individuals.append(ind)
                print(f"✓ 个体{i}验证通过: 成本={ind.fitness.values[0]:.2f}, 方差={ind.fitness.values[1]:.4f}")
            else:
                failed_individuals.append(ind)
                print(f"✗ 个体{i}验证失败: 成本={ind.fitness.values[0]:.2f}, 方差={ind.fitness.values[1]:.4f}")

        print(f"帕累托前沿验证完成: {len(verified_individuals)}/{len(final_pareto_front)} 个体通过验证")

        if len(verified_individuals) == 0:
            print("警告：没有个体通过验证！将返回原始帕累托前沿")
            verified_pareto_front = final_pareto_front
        else:
            print(f"最终返回{len(verified_individuals)}个通过验证的个体")
            verified_pareto_front = verified_individuals

        # 提取验证后帕累托前沿的适应度值
        pareto_front_values = np.array([ind.fitness.values for ind in verified_pareto_front])

        # 保存到CSV和JSON（只保存通过验证的个体）
        save_pareto_front(pareto_front_values, "NSGA_DAN_CHUSHI_Verified")
        save_pareto_solutions(verified_pareto_front, "NSGA_DAN_CHUSHI_Verified")
        print("验证通过的帕累托解集已成功保存")

        # 如果有失败的个体，也保存一份用于分析
        if failed_individuals:
            failed_values = np.array([ind.fitness.values for ind in failed_individuals])
            save_pareto_front(failed_values, "NSGA_DAN_CHUSHI_Failed")
            save_pareto_solutions(failed_individuals, "NSGA_DAN_CHUSHI_Failed")
            print(f"验证失败的{len(failed_individuals)}个个体已保存用于分析")

    except Exception as e:
        print(f"保存帕累托解集时出错: {str(e)}")
        # 如果验证过程出错，返回原始帕累托前沿
        verified_pareto_front = tools.sortNondominated(population, len(population), first_front_only=True)[0]

        # 确保优化结果返回，不阻塞在图表上
    return verified_pareto_front, logbook


def save_pareto_front(pareto_front, algorithm_name, save_dir='NSGA_DAN_CHUSHI_result'):
    """
    保存帕累托前沿到CSV文件（按成本从低到高排序）
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

    # 按成本从低到高排序
    df = df.sort_values(by='system_cost', ascending=True).reset_index(drop=True)

    # 保存为CSV文件
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = f"{algorithm_name}_pareto_front_{timestamp}.csv"
    filepath = os.path.join(save_dir, filename)
    df.to_csv(filepath, index=False)
    print(f"帕累托前沿已保存到：{filepath}")
    print(f"共{len(df)}个解，按成本从{df['system_cost'].min():.2f}元到{df['system_cost'].max():.2f}元排序")
    return filepath


def save_pareto_solutions(solutions, algorithm_name, save_dir='NSGA_DAN_CHUSHI_result'):
    """
    保存帕累托解集（包括决策变量和目标值）到JSON文件（按成本从低到高排序）
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
    for ind in solutions:
        sol_dict = {
            'position': list(ind),  # Individual对象本身就是决策变量的列表
            'objectives': list(ind.fitness.values)  # 从fitness属性获取适应度值
        }
        solution_list.append(sol_dict)

    # 按成本（objectives的第一个元素）从低到高排序
    solution_list.sort(key=lambda x: x['objectives'][0])

    # 保存为JSON文件
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = f"{algorithm_name}_solutions_{timestamp}.json"
    filepath = os.path.join(save_dir, filename)
    with open(filepath, 'w') as f:
        json.dump(solution_list, f, indent=2)

    print(f"帕累托解集已保存到：{filepath}")
    if solution_list:
        min_cost = solution_list[0]['objectives'][0]
        max_cost = solution_list[-1]['objectives'][0]
        print(f"共{len(solution_list)}个解，按成本从{min_cost:.2f}元到{max_cost:.2f}元排序")

    return filepath


def decode_solution_to_pipe_diameters(irrigation_system, individual):
    """
    纯解码函数：直接将个体基因转换为管径配置，不进行任何约束检查
    """
    # 解码个体
    main_indices = individual[:len(irrigation_system.main_pipe) - 1]
    submain_first_indices = individual[len(irrigation_system.main_pipe) - 1:
                                       len(irrigation_system.main_pipe) + len(irrigation_system.submains) - 1]
    submain_second_indices = individual[len(irrigation_system.main_pipe) +
                                        len(irrigation_system.submains) - 1:]

    # 直接根据基因索引设置干管管径
    for i, index in enumerate(main_indices, start=1):
        # 直接使用基因索引从管径列表中选择，不做约束检查
        normalized_index = min(index, len(PIPE_SPECS["main"]["diameters"]) - 1)
        diameter = PIPE_SPECS["main"]["diameters"][normalized_index]
        irrigation_system.main_pipe[i]["diameter"] = diameter

    # 直接根据基因索引设置斗管管径
    for i, (first_index, second_index) in enumerate(zip(submain_first_indices, submain_second_indices)):
        # 直接使用基因索引从管径列表中选择，不做约束检查
        normalized_first_index = min(first_index, len(PIPE_SPECS["submain"]["diameters"]) - 1)
        first_diameter = PIPE_SPECS["submain"]["diameters"][normalized_first_index]

        normalized_second_index = min(second_index, len(PIPE_SPECS["submain"]["diameters"]) - 1)
        second_diameter = PIPE_SPECS["submain"]["diameters"][normalized_second_index]

        irrigation_system.submains[i]["diameter_first_half"] = first_diameter
        irrigation_system.submains[i]["diameter_second_half"] = second_diameter


def encode_pipe_diameters_from_system(irrigation_system):
    """从irrigation_system的当前状态编码为individual基因"""
    individual = []

    # 编码干管管径（从管段1开始，管段0不编码）
    for i in range(1, len(irrigation_system.main_pipe)):
        diameter = irrigation_system.main_pipe[i]["diameter"]
        try:
            diameter_index = PIPE_SPECS["main"]["diameters"].index(diameter)
        except ValueError:
            # 如果直径不在标准列表中，找最接近的
            diameter_index = 0
            min_diff = float('inf')
            for idx, std_diameter in enumerate(PIPE_SPECS["main"]["diameters"]):
                diff = abs(std_diameter - diameter)
                if diff < min_diff:
                    min_diff = diff
                    diameter_index = idx
        individual.append(diameter_index)

    # 编码斗管管径
    for submain in irrigation_system.submains:
        # 第一段管径
        first_diameter = submain["diameter_first_half"]
        try:
            first_index = PIPE_SPECS["submain"]["diameters"].index(first_diameter)
        except ValueError:
            first_index = 0
            min_diff = float('inf')
            for idx, std_diameter in enumerate(PIPE_SPECS["submain"]["diameters"]):
                diff = abs(std_diameter - first_diameter)
                if diff < min_diff:
                    min_diff = diff
                    first_index = idx
        individual.append(first_index)

        # 第二段管径
        second_diameter = submain["diameter_second_half"]
        try:
            second_index = PIPE_SPECS["submain"]["diameters"].index(second_diameter)
        except ValueError:
            second_index = 0
            min_diff = float('inf')
            for idx, std_diameter in enumerate(PIPE_SPECS["submain"]["diameters"]):
                diff = abs(std_diameter - second_diameter)
                if diff < min_diff:
                    min_diff = diff
                    second_index = idx
        individual.append(second_index)

    return individual


def verify_individual_correctness(individual, irrigation_system, group_count):
    """
    验证individual的正确性 - 【方案一专用版】
    修改说明：
    - 返回值变更为元组 (is_valid, reason)，以便于输出更详细的验证信息。
    """
    try:
        import copy
        temp_system = copy.deepcopy(irrigation_system)
        decode_solution_to_pipe_diameters(temp_system, individual)

        # 检查项一：静态管径约束
        if not check_diameter_constraints_only(temp_system):
            return (False, "不满足静态管径递减约束。")

        # 检查项二：动态压力约束
        all_nodes = list(range(1, temp_system.node_count + 1))
        temp_system._update_flow_rates(all_nodes)
        temp_system._calculate_hydraulics()
        temp_system._calculate_pressures()
        baseline_heads_data = temp_system.calculate_node_baseline_heads()
        baseline_heads = {data['node']: data['baseline_head'] for data in baseline_heads_data}

        for node in all_nodes:
            required_pressure = baseline_heads.get(node, DEFAULT_DRIP_LINE_INLET_PRESSURE)
            actual_pressure = temp_system.main_pipe[node]["pressure"]
            if actual_pressure < required_pressure:
                reason = f"节点 {node} 压力不足(需求:{required_pressure:.2f}m, 实际:{actual_pressure:.2f}m)。"
                return (False, reason)

        # 检查项三：适应度值一致性
        recalculated_cost = temp_system.get_system_cost()
        recalculated_variance = temp_system.calculate_pressure_variance(all_nodes)
        stored_cost = individual.fitness.values[0]
        stored_variance = individual.fitness.values[1]

        cost_diff = abs(recalculated_cost - stored_cost)
        var_diff = abs(recalculated_variance - stored_variance)

        if cost_diff > 1e-4 or var_diff > 1e-4:
            reason = f"适应度值不一致。成本(存:{stored_cost:.2f},算:{recalculated_cost:.2f}), 均方差(存:{stored_variance:.4f},算:{recalculated_variance:.4f})。"
            return (False, reason)

        return (True, "验证通过")

    except Exception as e:
        return (False, f"发生意外错误: {e}")


def check_diameter_constraints_only(irrigation_system):
    """仅检查管径约束（不修改）"""
    # 检查干管直径递减约束
    for i in range(1, len(irrigation_system.main_pipe)):
        current_diameter = irrigation_system.main_pipe[i]["diameter"]
        prev_diameter = irrigation_system.main_pipe[i - 1]["diameter"]
        if current_diameter > prev_diameter:
            return False

    # 检查斗管约束
    for i, submain in enumerate(irrigation_system.submains):
        main_diameter = irrigation_system.main_pipe[i + 1]["diameter"]
        first_diameter = submain["diameter_first_half"]
        second_diameter = submain["diameter_second_half"]

        if first_diameter > main_diameter or second_diameter > first_diameter:
            return False

    return True


# 将_update_pipe_diameters从局部函数提升为全局函数
def print_detailed_results(irrigation_system, best_individual, output_file="optimization_results_NSGA_DAN_CHUSHI.txt"):
    """
    优化后的结果输出函数 - 适用于【方案一：普通设计】
    修改说明：
    - 输出文件名已更改，以反映此方案。
    - 移除了关于轮灌组的循环和说明。
    - 只进行一次水力计算和结果展示。
    - 全局压力统计现在就是系统唯一的压力统计。
    """
    print("正在解码最优解的管径配置...")
    decode_solution_to_pipe_diameters(irrigation_system, best_individual)
    print("管径配置解码完成，开始输出详细结果...")

    # 获取各节点的基准水头
    baseline_heads_data = irrigation_system.calculate_node_baseline_heads()
    baseline_heads = {data['node']: data['baseline_head'] for data in baseline_heads_data}

    with open(output_file, 'w', encoding='utf-8') as f:
        def write_line(text):
            print(text)
            f.write(text + '\n')

        # 直接调用print_node_baseline_heads方法输出详细的基准水头表
        write_line("正在输出各节点基准水头详细信息...")

        # 临时重定向输出到文件
        import sys
        from io import StringIO

        # 保存原始stdout
        original_stdout = sys.stdout

        # 创建字符串缓冲区来捕获print_node_baseline_heads的输出
        captured_output = StringIO()
        sys.stdout = captured_output

        # 调用print_node_baseline_heads方法
        irrigation_system.print_node_baseline_heads()

        # 恢复原始stdout
        sys.stdout = original_stdout

        # 获取捕获的输出内容
        baseline_output = captured_output.getvalue()

        # 输出到控制台和文件
        print(baseline_output, end='')
        f.write(baseline_output)

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

        # 在此方案中，所有节点都启用
        active_nodes = list(range(1, irrigation_system.node_count + 1))

        write_line(f"\n=== 全系统水力计算结果 ===")

        # 更新水力计算
        irrigation_system._update_flow_rates(active_nodes)
        irrigation_system._calculate_hydraulics()
        irrigation_system._calculate_pressures()

        # --- 修正并确认统计逻辑 ---
        pressure_margins = []
        write_line("\n干管水力参数表:")
        write_line(
            "编号    后端距起点    管径    流量       流速     水头损失    段前水头压力    基准水头    压力富裕")
        write_line(
            "         (m)       (mm)   (m³/s)     (m/s)     (m)          (m)         (m)        (m)")
        write_line("-" * 100)

        distance = 0
        # 获取基准水头
        baseline_heads_data = irrigation_system.calculate_node_baseline_heads()
        baseline_heads = {data['node']: data['baseline_head'] for data in baseline_heads_data}

        for i in range(irrigation_system.node_count + 1):
            segment = irrigation_system.main_pipe[i]
            distance += segment["length"]

            # 从节点1开始才有对应的基准水头和压力富裕度
            node_id = i
            if node_id > 0:
                required_pressure = baseline_heads.get(node_id, DEFAULT_DRIP_LINE_INLET_PRESSURE)
                # 对应的斗管入口压力，就是其上游干管段的段前压力
                actual_pressure = irrigation_system.main_pipe[node_id]["pressure"]
                pressure_margin = actual_pressure - required_pressure
                pressure_margins.append(pressure_margin)

                write_line(f"{i:2d}    {distance:6.1f}    {segment['diameter']:4d}    "
                           f"{segment['flow_rate']:8.6f}  {segment['velocity']:6.2f}    "
                           f"{segment['head_loss']:6.2f}     {segment['pressure']:6.2f}     "
                           f"{required_pressure:6.2f}     {pressure_margin:6.2f}")
            else:  # 管段0没有对应的节点
                write_line(f"{i:2d}    {distance:6.1f}    {segment['diameter']:4d}    "
                           f"{segment['flow_rate']:8.6f}  {segment['velocity']:6.2f}    "
                           f"{segment['head_loss']:6.2f}     {segment['pressure']:6.2f}     "
                           f"{'—':>9}     {'—':>9}")

        # 计算并输出统计指标
        if pressure_margins:
            avg_margin = sum(pressure_margins) / len(pressure_margins)
            variance = sum((p - avg_margin) ** 2 for p in pressure_margins) / len(pressure_margins)
            std_dev = variance ** 0.5

            write_line("\n系统压力统计:")
            write_line(f"平均压力富裕程度: {avg_margin:.2f} m")
            write_line(f"压力均方差: {std_dev:.2f}")
        else:
            write_line("\n系统压力统计: 未能计算压力指标。")

        write_line("\n注: 压力富裕 = 对应节点上游干管段的段前水头压力 - 该节点基准水头")
        write_line("-" * 100)

        # 计算全局统计指标

        write_line("\n=== 全局压力统计 ===")
        write_line(f"系统整体平均压力富裕程度: {avg_margin:.2f} m")
        write_line(f"系统整体压力均方差: {std_dev:.2f}")
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


def visualize_pareto_front(pareto_front):
    """可视化Pareto前沿"""
    try:
        if not pareto_front:
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
        for ind in pareto_front:
            if np.all(np.isfinite(ind.fitness.values)):
                costs.append(ind.fitness.values[0])
                variances.append(ind.fitness.values[1])

        if not costs or not variances:
            print("没有有效的解集数据可供可视化")
            return

        plt.figure(figsize=(10, 6), dpi=100)
        plt.scatter(costs, variances, c='blue', marker='o', s=50, alpha=0.6, label='Pareto解')

        plt.title('多目标梳齿状NSGAⅡ管网优化Pareto前沿', fontproperties=chinese_font, fontsize=chinese_size + 2, pad=15)
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
        plt.savefig('NSGA_DAN_CHUSHI_pareto_front.png', dpi=300, bbox_inches='tight')
    except Exception as e:
        logging.error(f"可视化过程中发生错误: {str(e)}")
        print(f"可视化失败: {str(e)}")


def select_best_solution_by_marginal_improvement(solutions, max_variance_threshold=MAX_VARIANCE):
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
    valid_solutions = [sol for sol in solutions if sol.fitness.values[1] <= max_variance_threshold]

    # 如果没有满足条件的解，从原始解集中选择方差最小的
    if not valid_solutions:
        return min(solutions, key=lambda x: x.fitness.values[1])

    # 按成本升序排序
    sorted_solutions = sorted(valid_solutions, key=lambda ind: ind.fitness.values[0])

    # 如果解的数量较少，使用简单的效用函数
    if len(sorted_solutions) < 5:
        return sorted_solutions[0]

    # 对于较大的前沿，使用科学的效用函数方法
    try:
        # 提取成本和方差值
        costs = np.array([sol.fitness.values[0] for sol in sorted_solutions])
        variances = np.array([sol.fitness.values[1] for sol in sorted_solutions])

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


def main(node_count, frist_pressure, frist_diameter, LGZ1, LGZ2, Size, Max_iterations, SHOW, SAVE):
    """主函数"""
    try:
        # 初始化字体配置
        configure_fonts()
        # 创建灌溉系统
        irrigation_system = IrrigationSystem(
            node_count=node_count,
            frist_pressure=frist_pressure,
            frist_diameter=frist_diameter
        )

        # 设置轮灌参数
        best_lgz1, best_lgz2 = LGZ1, LGZ2
        logging.info("开始进行多目标优化...")

        # 执行优化
        start_time = time.time()
        pareto_front, logbook = multi_objective_optimization(irrigation_system, best_lgz1, best_lgz2, Size,
                                                             Max_iterations, SHOW, SAVE)

        end_time = time.time()

        logging.info(f"优化完成，耗时: {end_time - start_time:.2f}秒")

        # 输出结果
        if pareto_front:
            valid_solutions = [ind for ind in pareto_front if np.all(np.isfinite(ind.fitness.values))]
            if valid_solutions:
                # 选择最优解
                best_solution = select_best_solution_by_marginal_improvement(valid_solutions)
                # 使用相同的最优解进行详细结果输出
                print_detailed_results(irrigation_system, best_solution)
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
                    fig.savefig(f'NSGA_DAN_CHUSHI_result_fig{i}.png', dpi=300, bbox_inches='tight')
                    logging.info(f"图表{i}已保存为 NSGA_DAN_CHUSHI_result_fig{i}.png")
                except Exception as e:
                    logging.warning(f"保存图表{i}时出错: {e}")

        # 显示程序已完成的消息
        print("=========================================================")
        print("程序计算已完成，所有图表窗口将保持打开状态")
        print("关闭图表窗口不会影响结果，可随时查看保存的PNG图像文件")
        print("=========================================================")

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
    main(23, 49.62, 500, 8, 4, 50, 50, True, False)
    # main_batch(23, 49.62, 500, 8, 4, 200, 50, False, True)
