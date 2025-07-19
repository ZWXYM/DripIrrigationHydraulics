import json
import math
import os
import random
import logging
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
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
        # 【修改】对于传统设计，虽然也生成轮灌组，但在评估时通常不使用
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
            'pressure_variance': self.calculate_pressure_variance(active_nodes),
            'cost': self.get_system_cost(),
            'pressure_satisfaction': self._check_pressure_requirements()
        }

    def _update_flow_rates(self, active_nodes):
        """
        【修改】此版本为【传统设计模型】，采用【阶梯式流量递减模型】。
        1.  入口总流量由 `submain_flow * lgz1` 决定。
        2.  流量每隔 `step_size` 个管段才减少一个 `submain_flow`。
        3.  参数 `active_nodes` 在此方案中被忽略。
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
        if lgz1_val <= 1:
            total_flow = submain_flow * lgz1_val
            for i in range(len(self.main_pipe)):
                self.main_pipe[i]["flow_rate"] = total_flow
            return

        # 2. 计算入口总流量
        total_flow = submain_flow * lgz1_val

        # 3. 计算流量递减的步长
        num_drops = lgz1_val - 1
        step_size = self.node_count / num_drops if num_drops > 0 else float('inf')

        # 4. 实施阶梯式流量分配
        current_flow = total_flow
        next_drop_point = step_size

        for i in range(len(self.main_pipe)):
            self.main_pipe[i]["flow_rate"] = current_flow

            if i > 0:
                if i + 1e-9 >= next_drop_point:
                    current_flow -= submain_flow
                    current_flow = max(0, current_flow)
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
        for i, segment in enumerate(self.main_pipe):
            if i == 0:
                segment["pressure"] = self.frist_pressure
            else:
                previous_pressure = self.main_pipe[i - 1]["pressure"]
                current_loss = self.main_pipe[i - 1]["head_loss"]
                segment["pressure"] = previous_pressure - current_loss

            # 计算对应斗管压力 (即使斗管流量为0，其入口压力也等于干管节点压力)
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
                cost += ((submain["length"] / 2) - DEFAULT_DRIP_LINE_LENGTH) * price_lookup["submain"][
                    submain["diameter_second_half"]]

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

    def _check_pressure_requirements(self):
        """检查压力要求满足情况（使用节点基准水头）"""
        baseline_heads_data = self.calculate_node_baseline_heads()
        baseline_heads = {data['node']: data['baseline_head'] for data in baseline_heads_data}

        # 在传统设计模式下，检查所有节点
        for i, submain in enumerate(self.submains):
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
        【修改】统一计算水头均方差的标准方法 - 【传统设计模型】
        - 在此模型中，所有节点都被视为“激活”来参与压力评估。
        - 移除了 `and self.submains[node - 1]["flow_rate"] > 0` 的判断。
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
            if i < len(self.laterals) and len(self.laterals) > 0:
                lateral_diameter = self.laterals[i * lgz2]["diameter"]
            else:
                lateral_diameter = 90  # 默认直径

            lateral_length = DEFAULT_LATERAL_LENGTH
            lateral_head_loss = pressure_loss(lateral_diameter, lateral_length, lateral_flow)

            # 计算斗管水头损失（分两段计算）
            submain_first_diameter = 160
            submain_first_length = self.submain_length / 2
            submain_first_head_loss = pressure_loss(submain_first_diameter, submain_first_length, submain_flow)

            submain_second_diameter = 160
            submain_second_length = self.submain_length / 2
            submain_second_head_loss = pressure_loss(submain_second_diameter, submain_second_length, submain_flow)

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


class PSOOptimizationTracker:
    def __init__(self, show_dynamic_plots=False, auto_save=True, enable_smoothing=False):
        self.iterations = []
        self.best_costs = []
        self.best_variances = []
        self.all_costs = []
        self.all_variances = []
        self.all_iterations = []
        self.font_config = configure_fonts()
        self.show_dynamic_plots = show_dynamic_plots
        self.auto_save = auto_save
        self.enable_smoothing = enable_smoothing
        self.fig_2d = None
        self.ax1 = None
        self.ax2 = None
        self.fig_3d = None
        self.ax_3d = None
        self.scatter = None
        self.line1 = None
        self.line2 = None
        if self.show_dynamic_plots:
            self._init_plots()

    def _init_plots(self):
        plt.ion()
        chinese_font = self.font_config['chinese_font']
        english_font = self.font_config['english_font']
        chinese_size = self.font_config['chinese_size']
        english_size = self.font_config['english_size']
        self.fig_2d, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        self.ax1.set_ylabel('系统成本 (元)', fontproperties=chinese_font, fontsize=chinese_size)
        self.ax1.set_title('梳齿状PSO算法优化迭代曲线', fontproperties=chinese_font, fontsize=chinese_size + 2)
        self.ax1.grid(True, linestyle='--', alpha=0.7)
        self.ax2.set_xlabel('迭代次数', fontproperties=chinese_font, fontsize=chinese_size)
        self.ax2.set_ylabel('水头均方差', fontproperties=chinese_font, fontsize=chinese_size)
        self.ax2.grid(True, linestyle='--', alpha=0.7)
        for label in self.ax1.get_xticklabels() + self.ax1.get_yticklabels():
            label.set_fontname(english_font)
            label.set_fontsize(english_size)
        for label in self.ax2.get_xticklabels() + self.ax2.get_yticklabels():
            label.set_fontname(english_font)
            label.set_fontsize(english_size)
        self.line1, = self.ax1.plot([], [], 'b-', linewidth=2)
        self.line2, = self.ax2.plot([], [], 'r-', linewidth=2)
        self.fig_3d = plt.figure(figsize=(12, 10))
        self.ax_3d = self.fig_3d.add_subplot(111, projection='3d')
        self.ax_3d.set_xlabel('系统成本 (元)', fontproperties=chinese_font, fontsize=chinese_size)
        self.ax_3d.set_ylabel('水头均方差', fontproperties=chinese_font, fontsize=chinese_size)
        self.ax_3d.set_zlabel('迭代次数', fontproperties=chinese_font, fontsize=chinese_size)
        self.ax_3d.set_title('梳齿状PSO算法优化3D进度图', fontproperties=chinese_font, fontsize=chinese_size + 2)
        self.ax_3d.view_init(elev=30, azim=-35)
        for label in self.ax_3d.get_xticklabels() + self.ax_3d.get_yticklabels() + self.ax_3d.get_zticklabels():
            label.set_fontname(english_font)
            label.set_fontsize(english_size)
        self.fig_2d.canvas.draw()
        self.fig_2d.canvas.flush_events()
        self.fig_3d.canvas.draw()
        self.fig_3d.canvas.flush_events()

    def _smooth_curve(self, data, window_size=21, poly_order=3):
        if len(data) < window_size:
            return data
        if window_size % 2 == 0:
            window_size += 1
        if window_size >= len(data):
            window_size = min(len(data) - 2, 15)
            if window_size % 2 == 0:
                window_size -= 1
        try:
            from scipy.signal import savgol_filter
            return savgol_filter(data, window_size, poly_order)
        except (ImportError, ValueError):
            return self._moving_average(data, window_size=5)

    def _moving_average(self, data, window_size=5):
        import numpy as np
        if len(data) < window_size:
            return data
        weights = np.ones(window_size) / window_size
        smoothed = np.convolve(data, weights, mode='same')
        for i in range(window_size // 2):
            if i < len(smoothed):
                window = data[:i + window_size // 2 + 1]
                smoothed[i] = sum(window) / len(window)
        for i in range(len(data) - window_size // 2, len(data)):
            if i < len(smoothed):
                window = data[i - window_size // 2:]
                smoothed[i] = sum(window) / len(window)
        return smoothed

    def _exponential_moving_average(self, data, alpha=0.15):
        import numpy as np
        if len(data) < 2:
            return data
        ema = np.zeros_like(data)
        ema[0] = data[0]
        for i in range(1, len(data)):
            ema[i] = alpha * data[i] + (1 - alpha) * ema[i - 1]
        return ema

    def update_with_front(self, iteration, swarm, pareto_front):
        self.iterations.append(iteration)
        valid_solutions = [particle for particle in pareto_front if np.all(np.isfinite(particle.best_fitness))]
        if valid_solutions:
            import __main__
            if hasattr(__main__, 'select_best_solution_by_marginal_improvement'):
                best_solution = __main__.select_best_solution_by_marginal_improvement(valid_solutions)
            else:
                best_solution = self._select_best_solution(valid_solutions)

            if self.best_costs and self.best_variances:
                prev_cost = self.best_costs[-1]
                prev_variance = self.best_variances[-1]
                curr_cost = best_solution.best_fitness[0]
                curr_variance = best_solution.best_fitness[1]
                cost_change_ratio = abs((curr_cost - prev_cost) / prev_cost) if prev_cost != 0 else 0
                var_change_ratio = abs((curr_variance - prev_variance) / prev_variance) if prev_variance != 0 else 0
                if iteration > 20 and (cost_change_ratio > 0.15 or var_change_ratio > 0.15):
                    smoothed_cost = 0.3 * curr_cost + 0.7 * prev_cost
                    smoothed_variance = 0.3 * curr_variance + 0.7 * prev_variance
                    self.best_costs.append(smoothed_cost)
                    self.best_variances.append(smoothed_variance)
                else:
                    self.best_costs.append(curr_cost)
                    self.best_variances.append(curr_variance)
            else:
                self.best_costs.append(best_solution.best_fitness[0])
                self.best_variances.append(best_solution.best_fitness[1])
        else:
            if self.best_costs:
                self.best_costs.append(self.best_costs[-1])
                self.best_variances.append(self.best_variances[-1])
            else:
                self.best_costs.append(float('inf'))
                self.best_variances.append(float('inf'))

        for solution in valid_solutions:
            self.all_costs.append(solution.best_fitness[0])
            self.all_variances.append(solution.best_fitness[1])
            self.all_iterations.append(iteration)

        if self.show_dynamic_plots and iteration % 20 == 0:
            try:
                self._update_plots()
            except Exception as e:
                print(f"图表更新时出错: {e}")

    def update(self, iteration, swarm, pareto_front=None):
        if pareto_front is None:
            valid_solutions = [particle for particle in swarm if np.all(np.isfinite(particle.best_fitness))]
            pareto_front = self._extract_pareto_front(valid_solutions)
        self.update_with_front(iteration, swarm, pareto_front)

    def _extract_pareto_front(self, particles):
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

    def _dominates(self, fitness_a, fitness_b):
        not_worse = all(a <= b for a, b in zip(fitness_a, fitness_b))
        better = any(a < b for a, b in zip(fitness_a, fitness_b))
        return not_worse and better

    def _select_best_solution(self, solutions, max_variance_threshold=MAX_VARIANCE):
        if not solutions:
            return None
        valid_solutions = [sol for sol in solutions if sol.best_fitness[1] <= max_variance_threshold]
        if not valid_solutions:
            return min(solutions, key=lambda x: x.best_fitness[1])
        sorted_solutions = sorted(valid_solutions, key=lambda particle: particle.best_fitness[0])
        if len(sorted_solutions) <= 3:
            return sorted_solutions[0]
        costs = [sol.best_fitness[0] for sol in sorted_solutions]
        variances = [sol.best_fitness[1] for sol in sorted_solutions]
        min_cost, max_cost = min(costs), max(costs)
        min_var, max_var = min(variances), max(variances)
        cost_range = max_cost - min_cost if max_cost > min_cost else 1
        var_range = max_var - min_var if max_var > min_var else 1
        normalized_costs = [(c - min_cost) / cost_range for c in costs]
        normalized_vars = [(v - min_var) / var_range for v in variances]
        best_error, best_idx = float('inf'), 0
        for i in range(1, len(sorted_solutions) - 1):
            c1, v1 = np.array(normalized_costs[:i + 1]), np.array(normalized_vars[:i + 1])
            error1 = np.sum((v1 - (np.poly1d(np.polyfit(c1, v1, 1))(c1))) ** 2) if len(c1) > 1 else 0
            c2, v2 = np.array(normalized_costs[i:]), np.array(normalized_vars[i:])
            error2 = np.sum((v2 - (np.poly1d(np.polyfit(c2, v2, 1))(c2))) ** 2) if len(c2) > 1 else 0
            total_error = error1 + error2
            if total_error < best_error:
                best_error, best_idx = total_error, i
        return sorted_solutions[best_idx]

    def _update_plots(self):
        if not self.iterations:
            return
        chinese_font = self.font_config['chinese_font']
        english_font = self.font_config['english_font']
        chinese_size = self.font_config['chinese_size']
        english_size = self.font_config['english_size']
        try:
            cost_data, variance_data = self.best_costs, self.best_variances
            if self.enable_smoothing and len(self.iterations) > 5:
                smoothed_cost = self._exponential_moving_average(self._smooth_curve(cost_data))
                smoothed_variance = self._exponential_moving_average(self._smooth_curve(variance_data))
            else:
                smoothed_cost, smoothed_variance = cost_data, variance_data
            self.line1.set_data(self.iterations, smoothed_cost)
            self.line2.set_data(self.iterations, smoothed_variance)
            self.ax1.relim()
            self.ax1.autoscale_view()
            self.ax2.relim()
            self.ax2.autoscale_view()
            self.ax1.set_title('梳齿状PSO算法优化最优个体迭代曲线', fontproperties=chinese_font,
                               fontsize=chinese_size + 2)
            self.ax_3d.clear()
            self.scatter = self.ax_3d.scatter(self.all_costs, self.all_variances, self.all_iterations,
                                              c=self.all_iterations, cmap='viridis', s=50, alpha=0.6)
            self.ax_3d.view_init(elev=30, azim=-35)
            self.ax_3d.set_xlabel('系统成本 (元)', fontproperties=chinese_font, fontsize=chinese_size)
            self.ax_3d.set_ylabel('水头均方差', fontproperties=chinese_font, fontsize=chinese_size)
            self.ax_3d.set_zlabel('迭代次数', fontproperties=chinese_font, fontsize=chinese_size)
            self.ax_3d.set_title('梳齿状PSO算法优化3D进度图', fontproperties=chinese_font, fontsize=chinese_size + 2)
            for label in self.ax_3d.get_xticklabels() + self.ax_3d.get_yticklabels() + self.ax_3d.get_zticklabels():
                label.set_fontname(english_font)
                label.set_fontsize(english_size)
            self.fig_2d.canvas.draw_idle()
            self.fig_3d.canvas.draw_idle()
            self.fig_2d.canvas.flush_events()
            self.fig_3d.canvas.flush_events()
            plt.pause(0.0001)
        except Exception as e:
            print(f"图表更新过程中出错: {str(e)}")

    def finalize_plots(self):
        if not self.show_dynamic_plots:
            self.plot_2d_curves()
            self.plot_3d_progress()
            return
        chinese_font = self.font_config['chinese_font']
        english_font = self.font_config['english_font']
        chinese_size = self.font_config['chinese_size']
        english_size = self.font_config['english_size']
        try:
            cost_data, variance_data = self.best_costs, self.best_variances
            if self.enable_smoothing and len(self.iterations) > 5:
                smoothed_cost = self._exponential_moving_average(self._smooth_curve(cost_data, window_size=25),
                                                                 alpha=0.12)
                smoothed_variance = self._exponential_moving_average(self._smooth_curve(variance_data, window_size=25),
                                                                     alpha=0.12)
            else:
                smoothed_cost, smoothed_variance = cost_data, variance_data
            self.line1.set_data(self.iterations, smoothed_cost)
            self.line2.set_data(self.iterations, smoothed_variance)
            self.ax1.relim()
            self.ax1.autoscale_view()
            self.ax2.relim()
            self.ax2.autoscale_view()
            self.fig_2d.tight_layout()
            self.ax_3d.clear()
            self.scatter = self.ax_3d.scatter(self.all_costs, self.all_variances, self.all_iterations,
                                              c=self.all_iterations, cmap='viridis', s=50, alpha=0.6)
            cbar = self.fig_3d.colorbar(self.scatter, ax=self.ax_3d, pad=0.1)
            cbar.set_label('迭代次数', fontproperties=chinese_font, fontsize=chinese_size)
            for label in cbar.ax.get_yticklabels():
                label.set_fontname(english_font)
                label.set_fontsize(english_size)
            self.ax_3d.view_init(elev=30, azim=-35)
            self.ax_3d.set_xlabel('系统成本 (元)', fontproperties=chinese_font, fontsize=chinese_size)
            self.ax_3d.set_ylabel('水头均方差', fontproperties=chinese_font, fontsize=chinese_size)
            self.ax_3d.set_zlabel('迭代次数', fontproperties=chinese_font, fontsize=chinese_size)
            self.ax_3d.set_title('梳齿状PSO算法优化3D进度图', fontproperties=chinese_font, fontsize=chinese_size + 2)
            for label in self.ax_3d.get_xticklabels() + self.ax_3d.get_yticklabels() + self.ax_3d.get_zticklabels():
                label.set_fontname(english_font)
                label.set_fontsize(english_size)
            if self.auto_save:
                self.fig_2d.savefig('PSO_DAN_CHUSHI_2d_curves.png', dpi=300, bbox_inches='tight')
                self.fig_3d.savefig('PSO_DAN_CHUSHI_3d_progress.png', dpi=300, bbox_inches='tight')
            self.fig_2d.canvas.draw()
            self.fig_3d.canvas.draw()
        except Exception as e:
            print(f"最终图表更新时出错: {e}")

    def plot_2d_curves(self):
        if not self.iterations:
            print("没有数据可供绘图")
            return
        chinese_font, english_font = self.font_config['chinese_font'], self.font_config['english_font']
        chinese_size, english_size = self.font_config['chinese_size'], self.font_config['english_size']
        plt.ioff()
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        cost_data, variance_data = self.best_costs, self.best_variances
        if self.enable_smoothing and len(self.iterations) > 5:
            smoothed_cost = self._exponential_moving_average(self._smooth_curve(cost_data, window_size=25), alpha=0.12)
            smoothed_variance = self._exponential_moving_average(self._smooth_curve(variance_data, window_size=25),
                                                                 alpha=0.12)
        else:
            smoothed_cost, smoothed_variance = cost_data, variance_data
        ax1.plot(self.iterations, smoothed_cost, 'b-', linewidth=2)
        ax1.set_ylabel('系统成本 (元)', fontproperties=chinese_font, fontsize=chinese_size)
        ax1.set_title('梳齿状PSO算法优化迭代曲线', fontproperties=chinese_font, fontsize=chinese_size + 2)
        ax1.grid(True, linestyle='--', alpha=0.7)
        ax2.plot(self.iterations, smoothed_variance, 'r-', linewidth=2)
        ax2.set_xlabel('迭代次数', fontproperties=chinese_font, fontsize=chinese_size)
        ax2.set_ylabel('水头均方差', fontproperties=chinese_font, fontsize=chinese_size)
        ax2.grid(True, linestyle='--', alpha=0.7)
        for label in ax1.get_xticklabels() + ax1.get_yticklabels():
            label.set_fontname(english_font)
            label.set_fontsize(english_size)
        for label in ax2.get_xticklabels() + ax2.get_yticklabels():
            label.set_fontname(english_font)
            label.set_fontsize(english_size)
        plt.tight_layout()
        if self.auto_save:
            plt.savefig('PSO_DAN_CHUSHI_2d_curves.png', dpi=300, bbox_inches='tight')
        plt.ion()
        plt.show(block=False)

    def plot_3d_progress(self):
        if not self.all_iterations:
            print("没有数据可供绘图")
            return
        chinese_font, english_font = self.font_config['chinese_font'], self.font_config['english_font']
        chinese_size, english_size = self.font_config['chinese_size'], self.font_config['english_size']
        plt.ioff()
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(self.all_costs, self.all_variances, self.all_iterations, c=self.all_iterations,
                             cmap='viridis', s=50, alpha=0.6)
        cbar = fig.colorbar(scatter, ax=ax, pad=0.1)
        cbar.set_label('迭代次数', fontproperties=chinese_font, fontsize=chinese_size)
        for label in cbar.ax.get_yticklabels():
            label.set_fontname(english_font)
            label.set_fontsize(english_size)
        ax.set_xlabel('系统成本 (元)', fontproperties=chinese_font, fontsize=chinese_size)
        ax.set_ylabel('水头均方差', fontproperties=chinese_font, fontsize=chinese_size)
        ax.set_zlabel('迭代次数', fontproperties=chinese_font, fontsize=chinese_size)
        ax.set_title('梳齿状PSO算法优化3D进度图', fontproperties=chinese_font, fontsize=chinese_size + 2)
        for label in ax.get_xticklabels() + ax.get_yticklabels() + ax.get_zticklabels():
            label.set_fontname(english_font)
            label.set_fontsize(english_size)
        ax.view_init(elev=30, azim=-35)
        if self.auto_save:
            plt.savefig('PSO_DAN_CHUSHI_3d_progress.png', dpi=300, bbox_inches='tight')
        plt.ion()
        plt.show(block=False)


class Particle:
    def __init__(self, dimensions, value_ranges):
        self.dimensions = dimensions
        self.value_ranges = value_ranges
        self.position = self.initialize_position()
        self.velocity = np.zeros(dimensions)
        self.best_position = self.position.copy()
        self.fitness = None
        self.best_fitness = None

    def initialize_position(self):
        position = np.zeros(self.dimensions, dtype=int)
        for i in range(self.dimensions):
            min_val, max_val = self.value_ranges[i]
            position[i] = random.randint(min_val, max_val)
        return position

    def update_velocity(self, global_best_position, w=0.7, c1=1.5, c2=1.5):
        r1 = np.random.random(self.dimensions)
        r2 = np.random.random(self.dimensions)
        cognitive_component = c1 * r1 * (self.best_position - self.position)
        social_component = c2 * r2 * (global_best_position - self.position)
        self.velocity = w * self.velocity + cognitive_component + social_component

    def update_position(self):
        new_position = self.position + np.round(self.velocity).astype(int)
        for i in range(self.dimensions):
            min_val, max_val = self.value_ranges[i]
            new_position[i] = max(min_val, min(max_val, new_position[i]))
        self.position = new_position


def multi_objective_pso(irrigation_system, lgz1, lgz2, swarm_size, max_iterations, show_plots,
                        auto_save):
    """多目标PSO优化函数"""
    tracker = PSOOptimizationTracker(show_dynamic_plots=show_plots, auto_save=auto_save)
    group_count = irrigation_system.initialize_irrigation_groups(lgz1, lgz2)
    main_pipe_dims = len(irrigation_system.main_pipe) - 1
    submain_dims = len(irrigation_system.submains) * 2
    dimensions = main_pipe_dims + submain_dims
    value_ranges = []
    for i in range(main_pipe_dims):
        value_ranges.append((0, len(PIPE_SPECS["main"]["diameters"]) - 1))
    for i in range(submain_dims):
        value_ranges.append((0, len(PIPE_SPECS["submain"]["diameters"]) - 1))

    def dominates(fitness_a, fitness_b):
        not_worse = all(a <= b for a, b in zip(fitness_a, fitness_b))
        better = any(a < b for a, b in zip(fitness_a, fitness_b))
        return not_worse and better

    def select_leader(pareto_front, particle):
        if not pareto_front:
            return particle
        crowding_distances = calculate_crowding_distance(pareto_front)
        tournament_size = min(3, len(pareto_front))
        candidates = random.sample(list(zip(pareto_front, crowding_distances)), tournament_size)
        return max(candidates, key=lambda x: x[1])[0]

    def calculate_crowding_distance(pareto_front):
        n = len(pareto_front)
        if n <= 2:
            return [float('inf')] * n
        distances = [0.0] * n
        for obj_idx in range(2):
            sorted_indices = sorted(range(n), key=lambda i: pareto_front[i].best_fitness[obj_idx])
            distances[sorted_indices[0]] = float('inf')
            distances[sorted_indices[-1]] = float('inf')
            if n > 2:
                obj_range = (pareto_front[sorted_indices[-1]].best_fitness[obj_idx] -
                             pareto_front[sorted_indices[0]].best_fitness[obj_idx])
                if obj_range > 0:
                    for i in range(1, n - 1):
                        distances[sorted_indices[i]] += ((pareto_front[sorted_indices[i + 1]].best_fitness[obj_idx] -
                                                          pareto_front[sorted_indices[i - 1]].best_fitness[
                                                              obj_idx]) / obj_range)
        return distances

    def adjust_pipe_diameters(position, active_nodes):
        individual = position.tolist()
        max_attempts = 10
        attempt = 0
        baseline_heads_data = irrigation_system.calculate_node_baseline_heads()
        baseline_heads = {data['node']: data['baseline_head'] for data in baseline_heads_data}

        while attempt < max_attempts:
            _update_pipe_diameters(irrigation_system, individual)
            irrigation_system._update_flow_rates(active_nodes)
            irrigation_system._calculate_hydraulics()
            irrigation_system._calculate_pressures()

            pressure_satisfied = True
            diameter_increased = False
            for node in active_nodes:
                if node <= len(irrigation_system.submains):
                    submain = irrigation_system.submains[node - 1]
                    required_pressure = baseline_heads.get(node, DEFAULT_DRIP_LINE_INLET_PRESSURE)
                    if submain["inlet_pressure"] < required_pressure:
                        pressure_satisfied = False
                        path_segments = list(range(1, node + 1))
                        min_diameter_segment = min(path_segments,
                                                   key=lambda x: irrigation_system.main_pipe[x]["diameter"])
                        current_diameter = irrigation_system.main_pipe[min_diameter_segment]["diameter"]
                        larger_diameters = [d for d in PIPE_SPECS["main"]["diameters"] if d > current_diameter]
                        if larger_diameters:
                            new_diameter = min(larger_diameters)
                            irrigation_system.main_pipe[min_diameter_segment]["diameter"] = new_diameter
                            segment_index = min_diameter_segment - 1
                            individual[segment_index] = PIPE_SPECS["main"]["diameters"].index(new_diameter)
                            diameter_increased = True
            if pressure_satisfied:
                corrected_position = encode_pipe_diameters_from_system_pso(irrigation_system)
                return True, corrected_position
            if not diameter_increased:
                return False, np.array(individual)
            attempt += 1
        return False, np.array(individual)

    def evaluate(position):
        """
        【修改】评估函数 - 适用于【传统设计模型】
        1. 不再遍历轮灌组，将所有节点视为一个整体进行评估。
        2. 调用 adjust_pipe_diameters 一次，确保所有节点满足压力。
        3. 计算一次总成本和一次全系统的压力方差。
        """
        try:
            # 在此模型中，active_nodes 包含所有节点
            all_nodes = list(range(1, irrigation_system.node_count + 1))

            # 创建 position 的副本用于调整
            position_copy = position.copy()

            # 调整管径直到满足要求
            success, adjusted_position = adjust_pipe_diameters(position_copy, all_nodes)

            if not success:
                return float('inf'), float('inf')

            # 关键：将最终的正确位置编码更新到原 position 中
            position[:] = adjusted_position

            # 使用调整后的系统状态计算最终指标
            irrigation_system._update_flow_rates(all_nodes)
            irrigation_system._calculate_hydraulics()
            irrigation_system._calculate_pressures()

            cost = irrigation_system.get_system_cost()
            pressure_variance = irrigation_system.calculate_pressure_variance(all_nodes)

            return cost, pressure_variance

        except Exception as e:
            logging.error(f"评估错误: {str(e)}")
            return float('inf'), float('inf')

    def _update_pipe_diameters(irrigation_system, individual):
        main_indices = individual[:len(irrigation_system.main_pipe) - 1]
        submain_first_indices = individual[len(irrigation_system.main_pipe) - 1:
                                           len(irrigation_system.main_pipe) + len(irrigation_system.submains) - 1]
        submain_second_indices = individual[len(irrigation_system.main_pipe) +
                                            len(irrigation_system.submains) - 1:]
        prev_diameter = irrigation_system.main_pipe[0]["diameter"]
        for i, index in enumerate(main_indices, start=1):
            available_diameters = [d for d in PIPE_SPECS["main"]["diameters"] if d <= prev_diameter]
            if not available_diameters:
                available_diameters = [min(PIPE_SPECS["main"]["diameters"])]
            normalized_index = min(index, len(available_diameters) - 1)
            diameter = available_diameters[normalized_index]
            irrigation_system.main_pipe[i]["diameter"] = diameter
            prev_diameter = diameter
        for i, (first_index, second_index) in enumerate(zip(submain_first_indices, submain_second_indices)):
            main_connection_diameter = irrigation_system.main_pipe[i + 1]["diameter"]
            available_first_diameters = [d for d in PIPE_SPECS["submain"]["diameters"] if d <= main_connection_diameter]
            if not available_first_diameters:
                available_first_diameters = [min(PIPE_SPECS["submain"]["diameters"])]
            normalized_first_index = min(first_index, len(available_first_diameters) - 1)
            first_diameter = available_first_diameters[normalized_first_index]
            available_second_diameters = [d for d in PIPE_SPECS["submain"]["diameters"] if d <= first_diameter]
            if not available_second_diameters:
                available_second_diameters = [min(PIPE_SPECS["submain"]["diameters"])]
            normalized_second_index = min(second_index, len(available_second_diameters) - 1)
            second_diameter = available_second_diameters[normalized_second_index]
            irrigation_system.submains[i]["diameter_first_half"] = first_diameter
            irrigation_system.submains[i]["diameter_second_half"] = second_diameter

    swarm = [Particle(dimensions, value_ranges) for _ in range(swarm_size)]
    pareto_front = []
    logbook = tools.Logbook()
    stats = tools.Statistics()
    stats.register("min", np.min, axis=0)
    stats.register("avg", np.mean, axis=0)
    stats.register("std", np.std, axis=0)

    for particle in swarm:
        particle.fitness = evaluate(particle.position)
        particle.best_fitness = particle.fitness

    for particle in swarm:
        if np.all(np.isfinite(particle.fitness)):
            is_dominated = False
            pareto_front_copy = pareto_front.copy()
            for idx, solution in enumerate(pareto_front_copy):
                if dominates(solution.best_fitness, particle.fitness):
                    is_dominated = True
                    break
                elif dominates(particle.fitness, solution.best_fitness):
                    pareto_front.remove(solution)
            if not is_dominated:
                pareto_particle = Particle(dimensions, value_ranges)
                pareto_particle.position = particle.position.copy()
                pareto_particle.best_position = particle.position.copy()
                pareto_particle.fitness = particle.fitness
                pareto_particle.best_fitness = particle.fitness
                pareto_front.append(pareto_particle)

    fitness_values = [p.fitness for p in swarm if np.all(np.isfinite(p.fitness))]
    if fitness_values:
        record = stats.compile(fitness_values)
        logbook.record(gen=0, **record)
        logging.info(f"初始化完成，群体大小: {len(swarm)}，Pareto前沿大小: {len(pareto_front)}")
        logging.info(f"Iteration 0: {record}")
        tracker.update_with_front(0, swarm, pareto_front)

    for iteration in range(1, max_iterations + 1):
        for particle in swarm:
            if pareto_front:
                leader = select_leader(pareto_front, particle)
                particle.update_velocity(leader.best_position)
                particle.update_position()
                particle.fitness = evaluate(particle.position)
                if (np.all(np.isfinite(particle.fitness)) and
                        (not np.all(np.isfinite(particle.best_fitness)) or
                         dominates(particle.fitness, particle.best_fitness) or
                         np.array_equal(particle.fitness, particle.best_fitness))):
                    particle.best_position = particle.position.copy()
                    particle.best_fitness = particle.fitness
                if np.all(np.isfinite(particle.fitness)):
                    is_dominated = False
                    pareto_front_copy = pareto_front.copy()
                    for solution in pareto_front_copy:
                        if dominates(solution.best_fitness, particle.fitness):
                            is_dominated = True
                            break
                        elif dominates(particle.fitness, solution.best_fitness):
                            pareto_front.remove(solution)
                    if not is_dominated:
                        pareto_particle = Particle(dimensions, value_ranges)
                        pareto_particle.position = particle.position.copy()
                        pareto_particle.best_position = particle.position.copy()
                        pareto_particle.fitness = particle.fitness
                        pareto_particle.best_fitness = particle.fitness
                        if not any(np.array_equal(p.position, pareto_particle.position) for p in pareto_front):
                            pareto_front.append(pareto_particle)

        fitness_values = [p.fitness for p in swarm if np.all(np.isfinite(p.fitness))]
        if fitness_values:
            record = stats.compile(fitness_values)
            logbook.record(gen=iteration, **record)
            tracker.update_with_front(iteration, swarm, pareto_front)
            if iteration % 10 == 0:
                logging.info(f"Iteration {iteration}: {record}")
                logging.info(f"当前Pareto前沿大小: {len(pareto_front)}")

    final_pareto_front = [p for i, p in enumerate(pareto_front) if
                          not any(np.array_equal(p.position, q.position) for q in pareto_front[:i])]
    for particle in final_pareto_front:
        particle.fitness = evaluate(particle.position)
        particle.best_fitness = particle.fitness
    non_dominated_front = [p for p in final_pareto_front if np.all(np.isfinite(p.best_fitness)) and not any(
        dominates(o.best_fitness, p.best_fitness) for o in final_pareto_front if
        not np.array_equal(o.position, p.position))]
    tracker.finalize_plots()

    try:
        print("正在验证最终帕累托前沿的正确性...")
        verified_particles = []
        failed_particles = []

        for i, particle in enumerate(non_dominated_front):
            is_valid, reason = verify_particle_correctness(particle, irrigation_system)
            if is_valid:
                verified_particles.append(particle)
                print(f"✓ 粒子{i}验证通过: 成本={particle.best_fitness[0]:.2f}, 方差={particle.best_fitness[1]:.4f}")
            else:
                failed_particles.append(particle)
                print(
                    f"✗ 粒子{i}验证失败: 成本={particle.best_fitness[0]:.2f}, 方差={particle.best_fitness[1]:.4f}. 原因: {reason}")

        print(f"帕累托前沿验证完成: {len(verified_particles)}/{len(non_dominated_front)} 粒子通过验证")

        verified_pareto_front = verified_particles if verified_particles else non_dominated_front
        if not verified_particles:
            print("警告：没有粒子通过验证！将返回原始帕累托前沿")

        pareto_front_values = np.array([p.best_fitness for p in verified_pareto_front])
        save_pareto_front(pareto_front_values, "PSO_DAN_CHUSHI_Verified")
        save_pareto_solutions(verified_pareto_front, "PSO_DAN_CHUSHI_Verified")
        print("验证通过的帕累托解集已成功保存")

        if failed_particles:
            failed_values = np.array([p.best_fitness for p in failed_particles])
            save_pareto_front(failed_values, "PSO_DAN_CHUSHI_Failed")
            save_pareto_solutions(failed_particles, "PSO_DAN_CHUSHI_Failed")
            print(f"验证失败的{len(failed_particles)}个粒子已保存用于分析")

    except Exception as e:
        print(f"保存帕累托解集时出错: {str(e)}")
        verified_pareto_front = non_dominated_front

    return verified_pareto_front, logbook


def save_pareto_front(pareto_front, algorithm_name, save_dir='PSO_DAN_CHUSHI_result'):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    df = pd.DataFrame(pareto_front, columns=['system_cost', 'pressure_variance'])
    df = df.sort_values(by='system_cost', ascending=True).reset_index(drop=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = f"{algorithm_name}_pareto_front_{timestamp}.csv"
    filepath = os.path.join(save_dir, filename)
    df.to_csv(filepath, index=False)
    print(f"帕累托前沿已保存到：{filepath}")
    if not df.empty:
        print(f"共{len(df)}个解，按成本从{df['system_cost'].min():.2f}元到{df['system_cost'].max():.2f}元排序")
    return filepath


def save_pareto_solutions(solutions, algorithm_name, save_dir='PSO_DAN_CHUSHI_result'):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    solution_list = []
    for solution in solutions:
        position = solution.best_position.tolist() if hasattr(solution.best_position,
                                                              'tolist') else solution.best_position
        fitness = solution.best_fitness.tolist() if hasattr(solution.best_fitness, 'tolist') else solution.best_fitness
        sol_dict = {'position': position, 'objectives': fitness}
        solution_list.append(sol_dict)
    solution_list.sort(key=lambda x: x['objectives'][0])
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


def decode_solution_to_pipe_diameters_pso(irrigation_system, position):
    main_indices = position[:len(irrigation_system.main_pipe) - 1]
    submain_first_indices = position[len(irrigation_system.main_pipe) - 1:
                                     len(irrigation_system.main_pipe) + len(irrigation_system.submains) - 1]
    submain_second_indices = position[len(irrigation_system.main_pipe) +
                                      len(irrigation_system.submains) - 1:]
    for i, index in enumerate(main_indices, start=1):
        normalized_index = min(int(index), len(PIPE_SPECS["main"]["diameters"]) - 1)
        irrigation_system.main_pipe[i]["diameter"] = PIPE_SPECS["main"]["diameters"][normalized_index]
    for i, (first_index, second_index) in enumerate(zip(submain_first_indices, submain_second_indices)):
        normalized_first_index = min(int(first_index), len(PIPE_SPECS["submain"]["diameters"]) - 1)
        first_diameter = PIPE_SPECS["submain"]["diameters"][normalized_first_index]
        normalized_second_index = min(int(second_index), len(PIPE_SPECS["submain"]["diameters"]) - 1)
        second_diameter = PIPE_SPECS["submain"]["diameters"][normalized_second_index]
        irrigation_system.submains[i]["diameter_first_half"] = first_diameter
        irrigation_system.submains[i]["diameter_second_half"] = second_diameter


def encode_pipe_diameters_from_system_pso(irrigation_system):
    position = []
    for i in range(1, len(irrigation_system.main_pipe)):
        diameter = irrigation_system.main_pipe[i]["diameter"]
        try:
            diameter_index = PIPE_SPECS["main"]["diameters"].index(diameter)
        except ValueError:
            diameter_index = min(range(len(PIPE_SPECS["main"]["diameters"])),
                                 key=lambda i: abs(PIPE_SPECS["main"]["diameters"][i] - diameter))
        position.append(diameter_index)
    for submain in irrigation_system.submains:
        first_diameter = submain["diameter_first_half"]
        try:
            first_index = PIPE_SPECS["submain"]["diameters"].index(first_diameter)
        except ValueError:
            first_index = min(range(len(PIPE_SPECS["submain"]["diameters"])),
                              key=lambda i: abs(PIPE_SPECS["submain"]["diameters"][i] - first_diameter))
        position.append(first_index)
        second_diameter = submain["diameter_second_half"]
        try:
            second_index = PIPE_SPECS["submain"]["diameters"].index(second_diameter)
        except ValueError:
            second_index = min(range(len(PIPE_SPECS["submain"]["diameters"])),
                               key=lambda i: abs(PIPE_SPECS["submain"]["diameters"][i] - second_diameter))
        position.append(second_index)
    return np.array(position)


def verify_particle_correctness(particle, irrigation_system):
    """
    【修改】验证粒子的正确性 - 【传统设计模型专用版】
    - 不再需要 group_count 参数。
    - 验证逻辑基于对整个系统的单次评估。
    - 返回值变更为元组 (is_valid, reason)，以便于输出更详细的验证信息。
    """
    try:
        import copy
        temp_system = copy.deepcopy(irrigation_system)
        decode_solution_to_pipe_diameters_pso(temp_system, particle.best_position)

        if not check_diameter_constraints_only_pso(temp_system):
            return (False, "不满足静态管径递减约束。")

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

        recalculated_cost = temp_system.get_system_cost()
        recalculated_variance = temp_system.calculate_pressure_variance(all_nodes)
        stored_cost, stored_variance = particle.best_fitness

        cost_diff = abs(recalculated_cost - stored_cost)
        var_diff = abs(recalculated_variance - stored_variance)

        if cost_diff > 1e-4 or var_diff > 1e-4:
            reason = f"适应度值不一致。成本(存:{stored_cost:.2f},算:{recalculated_cost:.2f}), 均方差(存:{stored_variance:.4f},算:{recalculated_variance:.4f})。"
            return (False, reason)

        return (True, "验证通过")

    except Exception as e:
        return (False, f"发生意外错误: {e}")


def check_diameter_constraints_only_pso(irrigation_system):
    """PSO版本：仅检查管径约束（不修改）"""
    for i in range(1, len(irrigation_system.main_pipe)):
        if irrigation_system.main_pipe[i]["diameter"] > irrigation_system.main_pipe[i - 1]["diameter"]:
            return False
    for i, submain in enumerate(irrigation_system.submains):
        if (submain["diameter_first_half"] > irrigation_system.main_pipe[i + 1]["diameter"] or
                submain["diameter_second_half"] > submain["diameter_first_half"]):
            return False
    return True


def print_detailed_results(irrigation_system, best_solution, output_file="optimization_results_PSO_DAN_CHUSHI.txt"):
    """
    【修改】优化后的结果输出函数 - 适用于【传统设计模型】
    - 移除了关于轮灌组的循环和说明。
    - 只进行一次水力计算和结果展示。
    """
    print("正在解码最优解的管径配置...")
    decode_solution_to_pipe_diameters_pso(irrigation_system, best_solution.best_position)
    print("管径配置解码完成，开始输出详细结果...")

    with open(output_file, 'w', encoding='utf-8') as f:
        def write_line(text):
            print(text)
            f.write(text + '\n')

        import sys
        from io import StringIO
        original_stdout = sys.stdout
        captured_output = StringIO()
        sys.stdout = captured_output
        irrigation_system.print_node_baseline_heads()
        sys.stdout = original_stdout
        baseline_output = captured_output.getvalue()
        print(baseline_output, end='')
        f.write(baseline_output)

        write_line("\n=== 斗管分段管径配置 ===")
        write_line("节点编号    第一段管径    第二段管径\n           (mm)         (mm)")
        write_line("-" * 45)
        for i, submain in enumerate(irrigation_system.submains):
            write_line(
                f"{i + 1:4d}         {submain['diameter_first_half']:4d}         {submain['diameter_second_half']:4d}")
        write_line("-" * 45)

        active_nodes = list(range(1, irrigation_system.node_count + 1))
        write_line(f"\n=== 全系统水力计算结果 ===")
        irrigation_system._update_flow_rates(active_nodes)
        irrigation_system._calculate_hydraulics()
        irrigation_system._calculate_pressures()

        pressure_margins = []
        write_line("\n干管水力参数表:")
        write_line("编号    后端距起点    管径    流量       流速     水头损失    段前水头压力    基准水头    压力富裕")
        write_line("         (m)       (mm)   (m³/s)     (m/s)     (m)          (m)         (m)        (m)")
        write_line("-" * 100)

        distance = 0
        baseline_heads_data = irrigation_system.calculate_node_baseline_heads()
        baseline_heads = {data['node']: data['baseline_head'] for data in baseline_heads_data}

        for i in range(irrigation_system.node_count + 1):
            segment = irrigation_system.main_pipe[i]
            distance += segment["length"]
            node_id = i
            if node_id > 0:
                required_pressure = baseline_heads.get(node_id, DEFAULT_DRIP_LINE_INLET_PRESSURE)
                actual_pressure = irrigation_system.main_pipe[node_id]["pressure"]
                pressure_margin = actual_pressure - required_pressure
                pressure_margins.append(pressure_margin)
                write_line(
                    f"{i:2d}    {distance:6.1f}    {segment['diameter']:4d}    {segment['flow_rate']:8.6f}  {segment['velocity']:6.2f}    {segment['head_loss']:6.2f}     {segment['pressure']:6.2f}     {required_pressure:6.2f}     {pressure_margin:6.2f}")
            else:
                write_line(
                    f"{i:2d}    {distance:6.1f}    {segment['diameter']:4d}    {segment['flow_rate']:8.6f}  {segment['velocity']:6.2f}    {segment['head_loss']:6.2f}     {segment['pressure']:6.2f}     {'—':>9}     {'—':>9}")

        if pressure_margins:
            avg_margin = np.mean(pressure_margins)
            std_dev = np.std(pressure_margins)
            write_line("\n系统压力统计:")
            write_line(f"平均压力富裕程度: {avg_margin:.2f} m")
            write_line(f"压力均方差: {std_dev:.2f}")
        else:
            write_line("\n系统压力统计: 未能计算压力指标。")

        write_line("-" * 100)

        total_cost = irrigation_system.get_system_cost()
        total_long = sum(seg['length'] for seg in irrigation_system.main_pipe)
        irrigation_area = (irrigation_system.node_count + 1) * irrigation_system.node_spacing * DEFAULT_SUBMAIN_LENGTH
        change_area = irrigation_area / (2000 / 3)
        cost_per_area = total_cost / change_area

        write_line("\n=== 系统总体信息 ===")
        write_line(f"系统总成本: {total_cost:.2f} 元")
        write_line(f"灌溉面积: {change_area:.1f} 亩")
        write_line(f"单位面积成本: {cost_per_area:.2f} 元/亩")
        write_line(f"管网总长度: {total_long:.1f} m")


def visualize_pareto_front(pareto_front_particles):
    """可视化Pareto前沿"""
    try:
        if not pareto_front_particles:
            print("没有可视化的解集")
            return
        font_config = configure_fonts()
        chinese_font, english_font = font_config['chinese_font'], font_config['english_font']
        chinese_size, english_size = font_config['chinese_size'], font_config['english_size']
        costs, variances = [], []
        for particle in pareto_front_particles:
            if np.all(np.isfinite(particle.best_fitness)):
                costs.append(particle.best_fitness[0])
                variances.append(particle.best_fitness[1])
        if not costs or not variances:
            print("没有有效的解集数据可供可视化")
            return
        plt.figure(figsize=(10, 6), dpi=100)
        plt.scatter(costs, variances, c='blue', marker='o', s=50, alpha=0.6, label='Pareto解')
        plt.title('多目标梳齿状PSO管网优化Pareto前沿 (传统设计)', fontproperties=chinese_font,
                  fontsize=chinese_size + 2, pad=15)
        plt.xlabel('系统成本', fontproperties=chinese_font, fontsize=chinese_size)
        plt.ylabel('压力方差', fontproperties=chinese_font, fontsize=chinese_size)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.ticklabel_format(style='sci', scilimits=(-2, 3), axis='both')
        legend = plt.legend(loc='upper right')
        for text in legend.get_texts():
            text.set_fontproperties(chinese_font)
            text.set_fontsize(chinese_size)
        for label in plt.gca().get_xticklabels() + plt.gca().get_yticklabels():
            label.set_fontname(english_font)
            label.set_fontsize(english_size)
        plt.tight_layout()
        plt.savefig('PSO_DAN_CHUSHI_pareto_front.png', dpi=300, bbox_inches='tight')
    except Exception as e:
        logging.error(f"可视化过程中发生错误: {str(e)}")


def select_best_solution_by_marginal_improvement(solutions, max_variance_threshold=MAX_VARIANCE):
    if not solutions:
        return None
    valid_solutions = [sol for sol in solutions if sol.best_fitness[1] <= max_variance_threshold]
    if not valid_solutions:
        return min(solutions, key=lambda x: x.best_fitness[1])
    sorted_solutions = sorted(valid_solutions, key=lambda p: p.best_fitness[0])
    if len(sorted_solutions) < 5:
        return sorted_solutions[0]
    try:
        costs = np.array([sol.best_fitness[0] for sol in sorted_solutions])
        variances = np.array([sol.best_fitness[1] for sol in sorted_solutions])
        min_cost, max_cost = np.min(costs), np.max(costs)
        min_var, max_var = np.min(variances), np.max(variances)
        cost_range = max(max_cost - min_cost, 1e-6)
        var_range = max(max_var - min_var, 1e-6)
        norm_costs = (costs - min_cost) / cost_range
        norm_vars = (variances - min_var) / var_range
        utility_values = 0.8 * (1 - norm_costs) + 0.2 * (1 - norm_vars)
        best_idx = np.argmax(utility_values)
        if len(sorted_solutions) >= 10 and var_range > 0.5:
            try:
                best_l_error, best_l_idx = float('inf'), 0
                for i in range(2, len(sorted_solutions) - 2):
                    if np.std(norm_costs[:i]) < 1e-4 or np.std(norm_costs[i:]) < 1e-4: continue
                    try:
                        slope1, intercept1 = np.polyfit(norm_costs[:i], norm_vars[:i], 1)
                        error1 = np.sum(((slope1 * norm_costs[:i] + intercept1) - norm_vars[:i]) ** 2)
                        slope2, intercept2 = np.polyfit(norm_costs[i:], norm_vars[i:], 1)
                        error2 = np.sum(((slope2 * norm_costs[i:] + intercept2) - norm_vars[i:]) ** 2)
                        total_error = error1 + error2
                        if total_error < best_l_error:
                            best_l_error, best_l_idx = total_error, i
                    except:
                        continue
                if best_l_error < float('inf'):
                    distance = abs(best_l_idx - best_idx)
                    if distance <= 3:
                        return sorted_solutions[best_idx]
                    else:
                        balanced_idx = int(0.7 * best_idx + 0.3 * best_l_idx)
                        return sorted_solutions[balanced_idx]
            except:
                return sorted_solutions[best_idx]
        return sorted_solutions[best_idx]
    except Exception as e:
        print(f"效用计算出错，使用成本最低解: {e}")
        return sorted_solutions[0]


def main(node_count, frist_pressure, frist_diameter, LGZ1, LGZ2, Size, Max_iterations, SHOW, SAVE):
    """主函数"""
    try:
        configure_fonts()
        irrigation_system = IrrigationSystem(
            node_count=node_count,
            frist_pressure=frist_pressure,
            frist_diameter=frist_diameter
        )
        logging.info("开始进行多目标PSO优化 (传统设计模型)...")
        start_time = time.time()
        pareto_front, logbook = multi_objective_pso(irrigation_system, LGZ1, LGZ2, Size, Max_iterations, SHOW, SAVE)
        end_time = time.time()
        logging.info(f"优化完成，耗时: {end_time - start_time:.2f}秒")

        if pareto_front:
            valid_solutions = [p for p in pareto_front if np.all(np.isfinite(p.best_fitness))]
            if valid_solutions:
                best_solution = select_best_solution_by_marginal_improvement(valid_solutions)
                print_detailed_results(irrigation_system, best_solution)
                visualize_pareto_front(pareto_front)
                logging.info("结果已保存并可视化完成")
            else:
                logging.error("未找到有效的解决方案")
        else:
            logging.error("多目标优化未能产生有效的Pareto前沿")

        auto_save = 0
        if auto_save == 1:
            figures = [plt.figure(i) for i in plt.get_fignums()]
            for i, fig in enumerate(figures):
                try:
                    fig.savefig(f'PSO_DAN_CHUSHI_result_fig{i}.png', dpi=300, bbox_inches='tight')
                    logging.info(f"图表{i}已保存为 PSO_DAN_CHUSHI_result_fig{i}.png")
                except Exception as e:
                    logging.warning(f"保存图表{i}时出错: {e}")

        print("\n=========================================================")
        print("程序计算已完成，所有图表窗口将保持打开状态")
        print("=========================================================")
        plt.ioff()
        plt.show()

    except Exception as e:
        logging.error(f"程序执行出错: {str(e)}")
        raise


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    # 参数: node_count, frist_pressure, frist_diameter, LGZ1, LGZ2, SwarmSize, MaxIterations, ShowPlots, AutoSave
    main(23, 49.62, 500, 8, 4, 50, 50, True, False)