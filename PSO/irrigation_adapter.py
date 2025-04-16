import numpy as np
import matplotlib.pyplot as plt
import random
import time
from matplotlib import rcParams
import platform
import os
from tqdm import tqdm
from scipy.spatial.distance import cdist, pdist
import logging
import threading
import concurrent.futures
from queue import Queue

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ====================== 灌溉系统问题适配 ======================

class IrrigationProblem:
    """灌溉系统优化问题包装类"""

    def __init__(self, node_count=23, first_pressure=49.62, first_diameter=500,
                 lgz1=8, lgz2=4, baseline_pressure=23.8, max_variance=5.0):
        """
        初始化灌溉系统优化问题

        参数:
        node_count: 节点数量
        first_pressure: 首节点压力
        first_diameter: 首管径
        lgz1: 轮灌组参数1
        lgz2: 轮灌组参数2
        baseline_pressure: 基准压力
        max_variance: 最大水头均方差
        """
        self.name = "灌溉系统优化问题"
        self.node_count = node_count
        self.first_pressure = first_pressure
        self.first_diameter = first_diameter
        self.lgz1 = lgz1
        self.lgz2 = lgz2
        self.baseline_pressure = baseline_pressure
        self.max_variance = max_variance

        # 定义决策变量数量和范围
        self.main_pipes_dims = node_count - 1  # 不含第0段
        self.submain_dims = node_count * 2  # 每条斗管分两段
        self.n_var = self.main_pipes_dims + self.submain_dims

        # 管道规格数据
        self.pipe_specs = {
            "submain": {
                "diameters": [140, 160],
                "prices": [29.99, 36.95]
            },
            "main": {
                "diameters": [300, 350, 400, 450, 500, 600],
                "prices": [145.20, 180.00, 208.80, 234.00, 276.00, 375.60]
            }
        }

        # 设置变量边界
        self.xl = [0] * self.n_var  # 下界
        self.xu = []  # 上界

        # 干管变量上界
        for _ in range(self.main_pipes_dims):
            self.xu.append(len(self.pipe_specs["main"]["diameters"]) - 1)

        # 斗管变量上界
        for _ in range(self.submain_dims):
            self.xu.append(len(self.pipe_specs["submain"]["diameters"]) - 1)

        # 多目标问题
        self.n_obj = 2

        # 初始化灌溉系统模型
        self.irrigation_system = self._create_irrigation_system()

        # 初始化轮灌组
        self.group_count = self.irrigation_system.initialize_irrigation_groups(self.lgz1, self.lgz2)

        # 用于存储Pareto前沿
        self.pareto_front = None

    def _create_irrigation_system(self):
        """创建灌溉系统模型"""
        # 这里应该使用PSO.py中的IrrigationSystem类
        # 但为了简化示例，我们创建一个模拟的系统
        return IrrigationSystemSimulator(
            node_count=self.node_count,
            first_pressure=self.first_pressure,
            first_diameter=self.first_diameter,
            baseline_pressure=self.baseline_pressure
        )

    def evaluate(self, x):
        """
        评估函数 - 计算目标函数值
        返回: [system_cost, pressure_variance]
        """
        # 将决策变量解码为管网配置
        self._update_pipe_diameters(x)

        # 初始化评估指标
        total_cost = 0
        pressure_variances = []

        # 对每个轮灌组进行评估
        for group_idx in range(self.group_count):
            # 获取当前组的活跃节点
            active_nodes = self.irrigation_system.irrigation_groups[group_idx]

            # 调整管径优化过程在实际应用中很复杂，这里简化处理
            # 在真实情况中，我们可能需要迭代调整管径直到满足压力要求

            # 更新流量和计算水力特性
            self.irrigation_system._update_flow_rates(active_nodes)
            self.irrigation_system._calculate_hydraulics()
            self.irrigation_system._calculate_pressures()

            # 评估性能指标
            current_cost = self.irrigation_system.get_system_cost()
            current_variance = self.irrigation_system.calculate_pressure_variance(active_nodes)

            # 累加成本和方差
            total_cost += current_cost
            pressure_variances.append(current_variance)

        # 计算平均值作为目标
        avg_cost = total_cost / self.group_count
        avg_pressure_variance = np.mean(pressure_variances)

        # 检查约束条件
        # 如果压力方差超过最大允许值，可以通过惩罚函数处理
        if avg_pressure_variance > self.max_variance:
            # 惩罚成本
            penalty_factor = 1 + (avg_pressure_variance - self.max_variance) / self.max_variance
            avg_cost = avg_cost * penalty_factor

        return [avg_cost, avg_pressure_variance]

    def _update_pipe_diameters(self, x):
        """根据决策变量更新管网直径配置"""
        # 解码个体
        main_indices = x[:self.main_pipes_dims]
        submain_first_indices = x[self.main_pipes_dims:self.main_pipes_dims + self.node_count]
        submain_second_indices = x[self.main_pipes_dims + self.node_count:]

        # 更新干管管径（确保管径递减）
        prev_diameter = self.irrigation_system.main_pipe[0]["diameter"]
        for i, index in enumerate(main_indices, start=1):
            # 确保管径不大于前一段
            available_diameters = [d for d in self.pipe_specs["main"]["diameters"] if d <= prev_diameter]

            if not available_diameters:
                # 如果没有可用管径，中断更新
                return

            # 规范化索引
            normalized_index = min(int(index), len(available_diameters) - 1)
            diameter = available_diameters[normalized_index]

            # 更新管径
            self.irrigation_system.main_pipe[i]["diameter"] = diameter
            prev_diameter = diameter

        # 更新斗管管径
        for i, (first_index, second_index) in enumerate(zip(submain_first_indices, submain_second_indices)):
            if i >= len(self.irrigation_system.submains):
                break

            # 获取连接处干管直径
            main_connection_diameter = self.irrigation_system.main_pipe[i + 1]["diameter"]

            # 确保斗管第一段管径不大于干管
            available_first_diameters = [d for d in self.pipe_specs["submain"]["diameters"] if d <= main_connection_diameter]

            if not available_first_diameters:
                continue

            # 规范化索引
            normalized_first_index = min(int(first_index), len(available_first_diameters) - 1)
            first_diameter = available_first_diameters[normalized_first_index]

            # 确保斗管第二段管径不大于第一段
            available_second_diameters = [d for d in self.pipe_specs["submain"]["diameters"] if d <= first_diameter]

            if not available_second_diameters:
                continue

            # 规范化索引
            normalized_second_index = min(int(second_index), len(available_second_diameters) - 1)
            second_diameter = available_second_diameters[normalized_second_index]

            # 更新斗管管径
            self.irrigation_system.submains[i]["diameter_first_half"] = first_diameter
            self.irrigation_system.submains[i]["diameter_second_half"] = second_diameter

    def get_pareto_front(self, n_points=100):
        """
        获取理论Pareto前沿（如果有的话）
        对于实际问题，通常没有已知的理论前沿
        """
        return None


# ====================== 灌溉系统模拟器 ======================

class IrrigationSystemSimulator:
    """简化的灌溉系统模拟器，模拟PSO.py中的IrrigationSystem类"""

    def __init__(self, node_count=23, first_pressure=49.62, first_diameter=500, baseline_pressure=23.8):
        """初始化灌溉系统模拟器"""
        self.node_count = node_count
        self.first_pressure = first_pressure
        self.first_diameter = first_diameter
        self.baseline_pressure = baseline_pressure

        # 系统常量定义
        self.DRIPPER_SPACING = 0.3  # 滴灌孔间隔（米）
        self.DEFAULT_NODE_SPACING = 400  # 默认节点间距（米）
        self.DEFAULT_FIRST_SEGMENT_LENGTH = 200  # 第一个管段的默认长度（米）
        self.DEFAULT_SUBMAIN_LENGTH = 800  # 默认斗管长度（米）
        self.DEFAULT_LATERAL_LENGTH = 200  # 默认农管长度（米）
        self.DEFAULT_AUXILIARY_LENGTH = 50  # 默认辅助农管长度（米）
        self.DEFAULT_DRIP_LINE_LENGTH = 50  # 默认滴灌带长度（米）
        self.DEFAULT_DRIP_LINE_SPACING = 1  # 默认滴灌带间隔（米）
        self.DEFAULT_DRIPPER_FLOW_RATE = 2.1  # 默认滴灌孔流量（L/h）
        self.DEFAULT_DRIP_LINE_INLET_PRESSURE = 10  # 默认滴灌带入口水头压力（米）

        # 管道规格和价格数据
        self.PIPE_SPECS = {
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

        # 初始化管网结构
        self.main_pipe = self._create_main_pipe()
        self.submains = self._create_submains()
        self.laterals = self._create_laterals()
        self.drip_lines = self._create_drip_lines()

        # 轮灌组相关属性
        self.lgz1 = None
        self.lgz2 = None
        self.irrigation_groups = []

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

    def _create_main_pipe(self):
        """创建干管"""
        segments = []
        for i in range(self.node_count + 1):
            segments.append({
                "index": i,
                "length": self.DEFAULT_FIRST_SEGMENT_LENGTH if i == 0 else self.DEFAULT_NODE_SPACING,
                "diameter": self.first_diameter,
                "flow_rate": 0.0,
                "velocity": 0.0,
                "head_loss": 0.0,
                "pressure": 0.0
            })
        return segments

    def _create_submains(self):
        """创建斗管"""
        return [{"index": i,
                 "length": self.DEFAULT_SUBMAIN_LENGTH,
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
            lateral_count = int(submain["length"] / (self.DEFAULT_DRIP_LINE_LENGTH * 2)) * 2

            for i in range(lateral_count):
                laterals.append({
                    "submain_index": submain["index"],
                    "index": i,
                    "length": self.DEFAULT_LATERAL_LENGTH,
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
            auxiliary_count = int(lateral["length"] / self.DEFAULT_AUXILIARY_LENGTH)
            for i in range(auxiliary_count):
                drip_line_count = int(self.DEFAULT_AUXILIARY_LENGTH / self.DEFAULT_DRIP_LINE_SPACING) * 2
                for j in range(drip_line_count):
                    drip_lines.append({
                        "lateral_index": lateral["index"],
                        "index": j,
                        "length": self.DEFAULT_DRIP_LINE_LENGTH,
                        "diameter": 16,
                        "dripper_count": int(self.DEFAULT_DRIP_LINE_LENGTH / self.DRIPPER_SPACING),
                        "flow_rate": 0.0
                    })
        return drip_lines

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
        drippers_per_line = int(self.DEFAULT_DRIP_LINE_LENGTH / self.DRIPPER_SPACING)
        single_dripper_flow = self.DEFAULT_DRIPPER_FLOW_RATE / 3600000  # 转换为m³/s
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
                segment["velocity"] = self._water_speed(segment["diameter"], segment["flow_rate"])
                segment["head_loss"] = self._pressure_loss(segment["diameter"],
                                                         segment["length"],
                                                         segment["flow_rate"])
            else:
                segment["velocity"] = 0
                segment["head_loss"] = 0

        # 计算斗管水力特性
        for submain in self.submains:
            if submain["flow_rate"] > 0:
                first_loss = self._pressure_loss(submain["diameter_first_half"],
                                               submain["length"] / 2,
                                               submain["flow_rate"])
                second_loss = self._pressure_loss(submain["diameter_second_half"],
                                                submain["length"] / 2,
                                                submain["flow_rate"])
                submain["head_loss"] = first_loss + second_loss
            else:
                submain["head_loss"] = 0

        # 计算农管水力特性
        for lateral in self.laterals:
            if lateral["flow_rate"] > 0:
                lateral["head_loss"] = self._pressure_loss(lateral["diameter"],
                                                         lateral["length"],
                                                         lateral["flow_rate"])
            else:
                lateral["head_loss"] = 0

    def _calculate_pressures(self):
        """计算所有节点的压力"""
        # 干管压力计算
        for i, segment in enumerate(self.main_pipe):
            if i == 0:
                segment["pressure"] = self.first_pressure
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

    def _water_speed(self, diameter, flow_rate):
        """计算流速"""
        d = diameter / 1000
        speed = flow_rate / ((d / 2) ** 2 * np.pi)
        return speed

    def _friction_factor(self, diameter, flow_rate, pipe_roughness=1.5e-6):
        """计算摩阻系数"""
        d = diameter / 1000
        if d <= 0 or flow_rate <= 0:
            return 0, 0

        v = self._water_speed(diameter, flow_rate)
        Re = 1000 * v * d / 1.004e-3

        if Re == 0:
            return 0, 0

        relative_roughness = pipe_roughness / d

        if Re < 2300:
            return 64 / Re, Re
        elif Re > 4000:
            A = (relative_roughness / 3.7) ** 1.11 + (5.74 / Re) ** 0.9
            f = 0.25 / (np.log10(A) ** 2)
            return f, Re
        else:
            f_2300 = 64 / 2300
            A_4000 = relative_roughness / 3.7 + 5.74 / 4000 ** 0.9
            f_4000 = 0.25 / (np.log10(A_4000) ** 2)
            f = f_2300 + (f_4000 - f_2300) * (Re - 2300) / (4000 - 2300)
            return f, Re

    def _pressure_loss(self, diameter, length, flow_rate):
        """计算压力损失"""
        f, Re = self._friction_factor(diameter, flow_rate)
        d = diameter / 1000
        v = self._water_speed(diameter, flow_rate)
        h_f = f * (length / d) * (v ** 2 / (2 * 9.81))
        return h_f

    def get_system_cost(self):
        """计算系统总成本"""
        cost = 0

        # 创建价格查找表
        price_lookup = {
            "main": {d: p for d, p in zip(self.PIPE_SPECS["main"]["diameters"],
                                        self.PIPE_SPECS["main"]["prices"])},
            "submain": {d: p for d, p in zip(self.PIPE_SPECS["submain"]["diameters"],
                                            self.PIPE_SPECS["submain"]["prices"])},
            "lateral": {d: p for d, p in zip(self.PIPE_SPECS["lateral"]["diameters"],
                                            self.PIPE_SPECS["lateral"]["prices"])}
        }

        # 计算干管成本（从管段1开始）
        for segment in self.main_pipe[1:]:
            if segment["diameter"] > 0:
                cost += segment["length"] * price_lookup["main"][segment["diameter"]]

        # 计算斗管成本
        for submain in self.submains:
            if submain["diameter_first_half"] > 0 and submain["diameter_second_half"] > 0:
                cost += (submain["length"] / 2) * price_lookup["submain"][submain["diameter_first_half"]]
                cost += ((submain["length"] / 2) - self.DEFAULT_DRIP_LINE_LENGTH) * price_lookup["submain"][
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
        cost += total_drip_line_length * self.PIPE_SPECS["drip_line"]["prices"][0]

        return cost

    def calculate_pressure_variance(self, active_nodes):
        """统一计算水头均方差的标准方法"""
        pressure_margins = [self.submains[node - 1]["inlet_pressure"] - self.baseline_pressure
                            for node in active_nodes
                            if node <= len(self.submains) and self.submains[node - 1]["flow_rate"] > 0]

        if not pressure_margins:
            return 0

        avg_margin = sum(pressure_margins) / len(pressure_margins)
        variance = sum((p - avg_margin) ** 2 for p in pressure_margins) / len(pressure_margins)
        return variance ** 0.5


# ====================== 多目标优化算法实现 ======================

class Particle:
    """粒子类，用于PSO算法"""

    def __init__(self, dimensions, bounds):
        """
        初始化粒子
        dimensions: 维度数（决策变量数量）
        bounds: 每个维度的取值范围列表，格式为[(min1,max1), (min2,max2),...]
        """
        self.dimensions = dimensions
        self.bounds = bounds

        # 初始化位置和速度
        self.position = self._initialize_position()
        self.velocity = np.zeros(dimensions)

        # 初始化个体最优位置和适应度
        self.best_position = self.position.copy()
        self.fitness = None
        self.best_fitness = None

    def _initialize_position(self):
        """初始化位置"""
        position = np.zeros(self.dimensions)
        for i in range(self.dimensions):
            min_val, max_val = self.bounds[i]
            position[i] = min_val + np.random.random() * (max_val - min_val)
        return position

    def update_velocity(self, global_best_position, w=0.7, c1=1.5, c2=1.5):
        """
        更新速度
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
        """更新位置并确保在边界内"""
        # 更新位置
        self.position = self.position + self.velocity

        # 确保位置在合法范围内
        for i in range(self.dimensions):
            min_val, max_val = self.bounds[i]
            self.position[i] = max(min_val, min(max_val, self.position[i]))


class MOPSO:
    """多目标粒子群优化算法"""

    def __init__(self, problem, pop_size=100, max_iterations=100,
                 w=0.7, c1=1.5, c2=1.5, use_archive=True):
        """
        初始化MOPSO算法
        problem: 优化问题实例
        pop_size: 种群大小
        max_iterations: 最大迭代次数
        w, c1, c2: PSO参数
        use_archive: 是否使用外部存档
        """
        self.problem = problem
        self.pop_size = pop_size
        self.max_iterations = max_iterations
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.use_archive = use_archive

        # 粒子群和外部存档
        self.particles = []
        self.archive = []
        self.leader_selector = self._crowding_distance_leader
        self.archive_size = 100  # 存档大小限制

        # 性能指标跟踪
        self.tracking = {
            'iterations': [],
            'fronts': [],
            'metrics': {
                'sp': [],
                'igd': [],
                'hv': []
            }
        }

    def optimize(self, tracking=True, verbose=True):
        """执行优化过程"""
        # 初始化粒子群
        bounds = list(zip(self.problem.xl, self.problem.xu))
        self.particles = [Particle(self.problem.n_var, bounds) for _ in range(self.pop_size)]

        # 初始化存档
        self.archive = []

        # 初始评估
        for particle in self.particles:
            particle.fitness = self.problem.evaluate(particle.position)
            particle.best_fitness = particle.fitness

        # 初始化外部存档
        if self.use_archive:
            self._update_archive()

        # 优化迭代
        for iteration in tqdm(range(self.max_iterations), disable=not verbose):
            # 对每个粒子
            for particle in self.particles:
                # 选择领导者
                if self.archive and self.use_archive:
                    leader = self.leader_selector(particle)
                else:
                    leader = self._select_leader_from_swarm(particle)

                # 更新速度和位置
                particle.update_velocity(leader.best_position, self.w, self.c1, self.c2)
                particle.update_position()

                # 评估新位置
                particle.fitness = self.problem.evaluate(particle.position)

                # 更新个体最优
                if self._dominates(particle.fitness, particle.best_fitness) or np.array_equal(particle.fitness, particle.best_fitness):
                    particle.best_position = particle.position.copy()
                    particle.best_fitness = particle.fitness

            # 更新外部存档
            if self.use_archive:
                self._update_archive()

            # 跟踪性能指标
            if tracking and iteration % 10 == 0:
                self._track_performance(iteration)

            # 调整惯性权重（线性递减）
            self.w = max(0.4, self.w - 0.3 / self.max_iterations)

        # 最终评估
        if tracking:
            self._track_performance(self.max_iterations - 1)

        # 返回Pareto前沿
        return self._get_pareto_front()

    def _update_archive(self):
        """更新外部存档"""
        # 将当前粒子的个体最优位置添加到存档中
        for particle in self.particles:
            is_dominated = False
            archive_copy = self.archive.copy()

            # 检查是否被存档中的解支配
            for solution in archive_copy:
                if self._dominates(solution.best_fitness, particle.best_fitness):
                    is_dominated = True
                    break
                # 检查是否支配存档中的解
                elif self._dominates(particle.best_fitness, solution.best_fitness):
                    self.archive.remove(solution)

            # 如果不被支配，添加到存档
            if not is_dominated and not any(np.array_equal(particle.best_position, a.best_position) for a in self.archive):
                # 深拷贝粒子
                archive_particle = Particle(particle.dimensions, particle.bounds)
                archive_particle.position = particle.best_position.copy()
                archive_particle.best_position = particle.best_position.copy()
                archive_particle.fitness = particle.best_fitness
                archive_particle.best_fitness = particle.best_fitness

                self.archive.append(archive_particle)

        # 如果存档超过大小限制，使用拥挤度排序保留多样性
        if len(self.archive) > self.archive_size:
            self._prune_archive()

    def _prune_archive(self):
        """使用拥挤度排序修剪存档"""
        # 使用拥挤度排序保留前N个解
        crowding_distances = self._calculate_crowding_distance([a.best_fitness for a in self.archive])

        # 排序并保留前archive_size个
        sorted_indices = np.argsort(crowding_distances)[::-1]
        self.archive = [self.archive[i] for i in sorted_indices[:self.archive_size]]

    def _crowding_distance_leader(self, particle):
        """基于拥挤度选择领导者"""
        # 随机选择候选
        candidates_idx = np.random.choice(len(self.archive), min(3, len(self.archive)), replace=False)
        candidates = [self.archive[i] for i in candidates_idx]

        # 计算候选的拥挤度
        if len(candidates) > 1:
            fitnesses = [c.best_fitness for c in candidates]
            crowding_distances = self._calculate_crowding_distance(fitnesses)

            # 选择拥挤度最大的
            best_idx = np.argmax(crowding_distances)
            return candidates[best_idx]
        else:
            return candidates[0]

    def _select_leader_from_swarm(self, particle):
        """从粒子群中选择领导者"""
        # 从非支配解中随机选择
        non_dominated = []
        for p in self.particles:
            if not any(self._dominates(other.best_fitness, p.best_fitness) for other in self.particles):
                non_dominated.append(p)

        if not non_dominated:
            return particle

        # 使用轮盘赌选择
        return random.choice(non_dominated)

    def _calculate_crowding_distance(self, fitnesses):
        """计算拥挤度"""
        n = len(fitnesses)
        if n <= 2:
            return [float('inf')] * n

        # 将fitnesses转换为numpy数组
        points = np.array(fitnesses)

        # 初始化距离
        distances = np.zeros(n)

        # 对每个目标
        for i in range(self.problem.n_obj):
            # 按该目标排序
            idx = np.argsort(points[:, i])

            # 边界点设为无穷
            distances[idx[0]] = float('inf')
            distances[idx[-1]] = float('inf')

            # 计算中间点
            if n > 2:
                # 目标范围
                f_range = points[idx[-1], i] - points[idx[0], i]

                # 避免除以零
                if f_range > 0:
                    for j in range(1, n - 1):
                        distances[idx[j]] += (points[idx[j + 1], i] - points[idx[j - 1], i]) / f_range

        return distances

    def _dominates(self, fitness1, fitness2):
        """判断fitness1是否支配fitness2"""
        # 确保转换为numpy数组以便比较
        f1 = np.array(fitness1)
        f2 = np.array(fitness2)

        # 至少一个目标更好
        better = False
        for i in range(len(f1)):
            if f1[i] > f2[i]:  # 假设最小化
                return False
            if f1[i] < f2[i]:
                better = True

        return better

    def _get_pareto_front(self):
        """获取算法生成的Pareto前沿"""
        if self.use_archive and self.archive:
            return np.array([p.best_fitness for p in self.archive])
        else:
            # 从粒子群中提取非支配解
            non_dominated = []
            for p in self.particles:
                if not any(self._dominates(other.best_fitness, p.best_fitness) for other in self.particles):
                    non_dominated.append(p.best_fitness)
            return np.array(non_dominated)

    def _track_performance(self, iteration):
        """跟踪性能指标"""
        # 获取当前Pareto前沿
        front = self._get_pareto_front()

        # 保存迭代次数和前沿
        self.tracking['iterations'].append(iteration)
        self.tracking['fronts'].append(front)

        # 计算性能指标
        true_front = self.problem.get_pareto_front()

        # 均匀性指标SP
        if len(front) > 1:
            sp = self._spacing(front)
            self.tracking['metrics']['sp'].append(sp)
        else:
            self.tracking['metrics']['sp'].append(float('nan'))

        # IGD指标
        if true_front is not None and len(front) > 0:
            igd = self._igd(front, true_front)
            self.tracking['metrics']['igd'].append(igd)
        else:
            self.tracking['metrics']['igd'].append(float('nan'))

        # 超体积指标HV
        if self.problem.n_obj == 2 and len(front) > 0:
            # 设置参考点
            if true_front is not None:
                ref_point = np.max(true_front, axis=0) * 1.1
            else:
                ref_point = np.max(front, axis=0) * 1.1

            hv = self._hypervolume(front, ref_point)
            self.tracking['metrics']['hv'].append(hv)
        else:
            self.tracking['metrics']['hv'].append(float('nan'))

    def _spacing(self, front):
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

    def _igd(self, approximation_front, true_front):
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

    def _hypervolume(self, front, reference_point):
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


class EnhancedMOPSO:
    """增强版多目标粒子群优化算法"""

    def __init__(self, problem, pop_size=100, max_iterations=100,
                 w_init=0.9, w_end=0.4, c1_init=2.5, c1_end=0.5,
                 c2_init=0.5, c2_end=2.5, use_archive=True,
                 archive_size=100, mutation_rate=0.1, adaptive_grid_size=10):
        """
        初始化增强版MOPSO算法
        problem: 优化问题实例
        pop_size: 种群大小
        max_iterations: 最大迭代次数
        w_init, w_end: 惯性权重的初始值和结束值
        c1_init, c1_end: 个体学习因子的初始值和结束值
        c2_init, c2_end: 社会学习因子的初始值和结束值
        use_archive: 是否使用外部存档
        archive_size: 存档大小限制
        mutation_rate: 变异率
        adaptive_grid_size: 自适应网格大小(用于存档管理)
        """
        self.problem = problem
        self.pop_size = pop_size
        self.max_iterations = max_iterations
        self.w_init = w_init
        self.w_end = w_end
        self.c1_init = c1_init
        self.c1_end = c1_end
        self.c2_init = c2_init
        self.c2_end = c2_end
        self.use_archive = use_archive
        self.archive_size = archive_size
        self.mutation_rate = mutation_rate

        # 粒子群和外部存档
        self.particles = []
        self.archive = []

        # 性能指标跟踪
        self.tracking = {
            'iterations': [],
            'fronts': [],
            'metrics': {
                'sp': [],
                'igd': [],
                'hv': []
            }
        }

    def optimize(self, tracking=True, verbose=True):
        """执行优化过程"""
        # 初始化粒子群
        self._initialize_particles()

        # 初始化存档
        self.archive = []

        # 初始评估
        for particle in self.particles:
            particle.fitness = self.problem.evaluate(particle.position)
            particle.best_fitness = particle.fitness.copy() if hasattr(particle.fitness, 'copy') else particle.fitness

        # 初始化外部存档
        if self.use_archive:
            self._update_archive()

        # 优化迭代
        for iteration in range(self.max_iterations):
            if verbose and iteration % 10 == 0:
                print(f"迭代 {iteration}/{self.max_iterations}，当前存档大小: {len(self.archive)}")

            # 更新参数
            progress = iteration / self.max_iterations
            w = self.w_init - (self.w_init - self.w_end) * progress
            c1 = self.c1_init - (self.c1_init - self.c1_end) * progress
            c2 = self.c2_init + (self.c2_end - self.c2_init) * progress

            # 对每个粒子
            for particle in self.particles:
                # 选择领导者
                if self.archive and self.use_archive:
                    leader = self._select_leader(particle)
                else:
                    leader = self._select_leader_from_swarm(particle)

                if leader is None:
                    continue

                # 更新速度和位置
                particle.update_velocity(leader.best_position, w, c1, c2)
                particle.update_position()

                # 应用变异
                self._apply_mutation(particle, progress)

                # 评估新位置
                particle.fitness = self.problem.evaluate(particle.position)

                # 更新个体最优
                if self._dominates(particle.fitness, particle.best_fitness):
                    particle.best_position = particle.position.copy()
                    particle.best_fitness = particle.fitness.copy() if hasattr(particle.fitness, 'copy') else particle.fitness
                elif not self._dominates(particle.best_fitness, particle.fitness):
                    # 非支配情况，随机选择是否更新
                    if np.random.random() < 0.5:
                        particle.best_position = particle.position.copy()
                        particle.best_fitness = particle.fitness.copy() if hasattr(particle.fitness, 'copy') else particle.fitness

            # 更新外部存档
            if self.use_archive:
                self._update_archive()

            # 跟踪性能指标
            if tracking and iteration % 10 == 0:
                self._track_performance(iteration)

        # 最终评估
        if tracking:
            self._track_performance(self.max_iterations - 1)

        if verbose:
            print(f"优化完成，最终存档大小: {len(self.archive)}")

        # 返回Pareto前沿
        return self._get_pareto_front()

    def _initialize_particles(self):
        """初始化粒子群"""
        self.particles = []
        bounds = list(zip(self.problem.xl, self.problem.xu))

        # 创建粒子
        for i in range(self.pop_size):
            particle = Particle(self.problem.n_var, bounds)

            # 特殊初始化第一个变量以获得更好的前沿覆盖
            if self.problem.n_obj == 2 and i < self.pop_size // 3:
                # 为前1/3的粒子均匀分布f1
                particle.position[0] = self.problem.xl[0] + (i / (self.pop_size // 3)) * (
                            self.problem.xu[0] - self.problem.xl[0])

            self.particles.append(particle)

    def _select_leader(self, particle):
        """选择领导者"""
        if not self.archive:
            return None

        # 使用拥挤度选择
        return self._crowding_distance_leader(particle)

    def _crowding_distance_leader(self, particle):
        """基于拥挤度的领导者选择"""
        if len(self.archive) <= 1:
            return self.archive[0] if self.archive else None

        # 选择候选
        tournament_size = min(3, len(self.archive))
        candidates_idx = np.random.choice(len(self.archive), tournament_size, replace=False)
        candidates = [self.archive[i] for i in candidates_idx]

        # 计算拥挤度
        crowding_distances = self._calculate_crowding_distance([c.best_fitness for c in candidates])

        # 选择拥挤度最大的
        max_idx = np.argmax(crowding_distances)
        return candidates[max_idx]

    def _select_leader_from_swarm(self, particle):
        """从粒子群中选择领导者"""
        # 获取非支配解
        non_dominated = []
        for p in self.particles:
            is_dominated = False
            for other in self.particles:
                if self._dominates(other.best_fitness, p.best_fitness):
                    is_dominated = True
                    break
            if not is_dominated:
                non_dominated.append(p)

        if not non_dominated:
            return particle

        # 随机选择一个非支配解
        return random.choice(non_dominated)

    def _apply_mutation(self, particle, progress):
        """应用变异操作"""
        # 根据迭代进度调整变异率
        current_rate = self.mutation_rate * (1 - progress * 0.7)

        # 对每个维度
        for i in range(self.problem.n_var):
            # 第一个变量有更高的变异率，以获取更好的前沿分布
            actual_rate = current_rate * 1.5 if i == 0 else current_rate

            if np.random.random() < actual_rate:
                # 多项式变异
                eta_m = 20  # 分布指数

                delta1 = (particle.position[i] - self.problem.xl[i]) / (self.problem.xu[i] - self.problem.xl[i])
                delta2 = (self.problem.xu[i] - particle.position[i]) / (self.problem.xu[i] - self.problem.xl[i])

                rand = np.random.random()
                mut_pow = 1.0 / (eta_m + 1.0)

                if rand < 0.5:
                    xy = 1.0 - delta1
                    val = 2.0 * rand + (1.0 - 2.0 * rand) * (xy ** (eta_m + 1.0))
                    delta_q = val ** mut_pow - 1.0
                else:
                    xy = 1.0 - delta2
                    val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * (xy ** (eta_m + 1.0))
                    delta_q = 1.0 - val ** mut_pow

                particle.position[i] += delta_q * (self.problem.xu[i] - self.problem.xl[i])
                particle.position[i] = max(self.problem.xl[i], min(self.problem.xu[i], particle.position[i]))

    def _update_archive(self):
        """更新外部存档"""
        # 将当前粒子的个体最优位置添加到存档中
        for particle in self.particles:
            is_dominated = False
            archive_copy = self.archive.copy()

            # 检查是否被存档中的解支配
            for solution in archive_copy:
                if self._dominates(solution.best_fitness, particle.best_fitness):
                    is_dominated = True
                    break
                # 检查是否支配存档中的解
                elif self._dominates(particle.best_fitness, solution.best_fitness):
                    self.archive.remove(solution)

            # 如果不被支配，添加到存档
            if not is_dominated and not any(np.array_equal(particle.best_position, a.best_position) for a in self.archive):
                # 深拷贝粒子
                archive_particle = Particle(particle.dimensions, particle.bounds)
                archive_particle.position = particle.best_position.copy()
                archive_particle.best_position = particle.best_position.copy()
                archive_particle.fitness = particle.best_fitness.copy() if hasattr(particle.best_fitness, 'copy') else particle.best_fitness
                archive_particle.best_fitness = particle.best_fitness.copy() if hasattr(particle.best_fitness, 'copy') else particle.best_fitness

                self.archive.append(archive_particle)

        # 如果存档超过大小限制，使用拥挤度排序保留多样性
        if len(self.archive) > self.archive_size:
            self._prune_archive()

    def _prune_archive(self):
        """标准存档修剪方法"""
        # 计算拥挤度
        crowding_distances = self._calculate_crowding_distance([a.best_fitness for a in self.archive])

        # 按拥挤度排序并保留前archive_size个
        sorted_indices = np.argsort(crowding_distances)[::-1]
        self.archive = [self.archive[i] for i in sorted_indices[:self.archive_size]]

    def _calculate_crowding_distance(self, fitnesses):
        """计算拥挤度"""
        n = len(fitnesses)
        if n <= 2:
            return [float('inf')] * n

        # 将fitnesses转换为numpy数组
        points = np.array(fitnesses)

        # 初始化距离
        distances = np.zeros(n)

        # 对每个目标
        for i in range(self.problem.n_obj):
            # 按该目标排序
            idx = np.argsort(points[:, i])

            # 边界点设为无穷
            distances[idx[0]] = float('inf')
            distances[idx[-1]] = float('inf')

            # 计算中间点
            if n > 2:
                # 目标范围
                f_range = points[idx[-1], i] - points[idx[0], i]

                # 避免除以零
                if f_range > 0:
                    for j in range(1, n - 1):
                        distances[idx[j]] += (points[idx[j + 1], i] - points[idx[j - 1], i]) / f_range

        return distances

    def _dominates(self, fitness1, fitness2):
        """判断fitness1是否支配fitness2"""
        # 确保转换为numpy数组以便比较
        if not isinstance(fitness1, np.ndarray):
            f1 = np.array(fitness1)
        else:
            f1 = fitness1

        if not isinstance(fitness2, np.ndarray):
            f2 = np.array(fitness2)
        else:
            f2 = fitness2

        # 至少一个目标更好，其他不差
        better = False
        for i in range(len(f1)):
            if f1[i] > f2[i]:  # 假设最小化
                return False
            if f1[i] < f2[i]:
                better = True

        return better

    def _get_pareto_front(self):
        """获取算法生成的Pareto前沿"""
        if self.use_archive and self.archive:
            return np.array([p.best_fitness for p in self.archive])
        else:
            # 从粒子群中提取非支配解
            non_dominated = []
            for p in self.particles:
                if not any(self._dominates(other.best_fitness, p.best_fitness) for other in self.particles):
                    non_dominated.append(p.best_fitness)
            return np.array(non_dominated)

    def _track_performance(self, iteration):
        """跟踪性能指标"""
        # 获取当前Pareto前沿
        front = self._get_pareto_front()

        # 保存迭代次数和前沿
        self.tracking['iterations'].append(iteration)
        self.tracking['fronts'].append(front)

        # 计算性能指标
        true_front = self.problem.get_pareto_front()

        # 均匀性指标SP
        if len(front) > 1:
            sp = self._spacing(front)
            self.tracking['metrics']['sp'].append(sp)
        else:
            self.tracking['metrics']['sp'].append(float('nan'))

        # IGD指标
        if true_front is not None and len(front) > 0:
            igd = self._igd(front, true_front)
            self.tracking['metrics']['igd'].append(igd)
        else:
            self.tracking['metrics']['igd'].append(float('nan'))

        # 超体积指标HV
        if self.problem.n_obj == 2 and len(front) > 0:
            # 设置参考点
            ref_point = np.max(front, axis=0) * 1.1

            hv = self._hypervolume(front, ref_point)
            self.tracking['metrics']['hv'].append(hv)
        else:
            self.tracking['metrics']['hv'].append(float('nan'))

    def _spacing(self, front):
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

    def _igd(self, approximation_front, true_front):
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

    def _hypervolume(self, front, reference_point):
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


# ====================== 性能评估与可视化函数 ======================

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


def compare_algorithms(problem, results, metrics, save_path, title="算法性能对比"):
    """比较不同算法的性能"""
    # 计算每个算法的平均性能
    n_algorithms = len(results)
    n_metrics = len(metrics)

    # 创建包含所有指标的子图
    fig, axes = plt.subplots(1, n_metrics, figsize=(6 * n_metrics, 8))
    if n_metrics == 1:
        axes = [axes]

    # 给图表添加总标题
    fig.suptitle(title, fontsize=16)

    # 为每个指标绘制箱线图
    for i, metric in enumerate(metrics):
        # 准备数据
        data = []
        labels = []

        for alg_name, alg_results in results.items():
            if metric in alg_results:
                values = [v for v in alg_results[metric] if not np.isnan(v)]
                if values:
                    data.append(values)
                    labels.append(alg_name)

        # 绘制箱线图
        if data:
            axes[i].boxplot(data, labels=labels)
            axes[i].set_title(f'{metric.upper()} 指标对比')
            axes[i].grid(True)

            # 旋转x轴标签以防止重叠
            plt.setp(axes[i].get_xticklabels(), rotation=30, ha='right')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def visualize_pareto_fronts(problem, fronts, algorithm_names, save_path, title="Pareto前沿对比"):
    """可视化Pareto前沿"""
    plt.figure(figsize=(10, 8))

    # 为每个算法选择一个颜色
    colors = plt.cm.tab10(np.linspace(0, 1, len(fronts)))

    # 绘制每个算法的Pareto前沿
    for (front, alg_name), color in zip(zip(fronts, algorithm_names), colors):
        if len(front) > 0:
            # 二维问题
            if problem.n_obj == 2:
                plt.scatter(front[:, 0], front[:, 1],
                           s=30, color=color, label=alg_name, alpha=0.7)

    plt.title(title)
    plt.xlabel('系统成本')
    plt.ylabel('水头均方差')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def run_comparison(problem, algorithms, algorithm_params, n_runs=5,
                   metrics=['sp', 'igd', 'hv'], results_dir='results'):
    """
    运行算法比较实验

    参数:
    problem: 优化问题实例
    algorithms: 算法类列表
    algorithm_params: 算法参数字典
    n_runs: 每个算法运行次数
    metrics: 性能指标列表
    results_dir: 结果保存目录
    """
    # 配置字体
    configure_fonts()

    # 创建结果目录
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # 结果存储
    results = {}

    # 对每个算法
    for algorithm_class in algorithms:
        algorithm_name = algorithm_class.__name__
        print(f"运行算法: {algorithm_name}")

        # 获取算法参数
        params = algorithm_params.get(algorithm_name, {})

        # 初始化结果存储
        results[algorithm_name] = {
            'fronts': [],
            'runtimes': []
        }

        # 对每个指标初始化存储
        for metric in metrics:
            results[algorithm_name][metric] = []

        # 运行n_runs次
        for run in range(n_runs):
            print(f"  运行 {run+1}/{n_runs}")

            # 创建算法实例
            algorithm = algorithm_class(problem, **params)

            # 运行算法
            start_time = time.time()
            pareto_front = algorithm.optimize(verbose=False)
            run_time = time.time() - start_time

            # 存储结果
            results[algorithm_name]['fronts'].append(pareto_front)
            results[algorithm_name]['runtimes'].append(run_time)

            # 计算性能指标
            for metric in metrics:
                if metric == 'sp' and len(pareto_front) > 1:
                    value = algorithm._spacing(pareto_front)
                elif metric == 'igd' and problem.get_pareto_front() is not None:
                    true_front = problem.get_pareto_front()
                    value = algorithm._igd(pareto_front, true_front)
                elif metric == 'hv' and problem.n_obj == 2:
                    ref_point = np.max(pareto_front, axis=0) * 1.1
                    value = algorithm._hypervolume(pareto_front, ref_point)
                else:
                    value = float('nan')

                results[algorithm_name][metric].append(value)

    # 计算最佳前沿
    best_fronts = {}
    algorithm_names = []

    for algorithm_name, algorithm_results in results.items():
        algorithm_names.append(algorithm_name)

        if 'igd' in metrics and problem.get_pareto_front() is not None:
            # 使用IGD选择最佳前沿
            valid_indices = [i for i, v in enumerate(algorithm_results['igd']) if not np.isnan(v)]
            if valid_indices:
                best_idx = valid_indices[np.argmin([algorithm_results['igd'][i] for i in valid_indices])]
                best_fronts[algorithm_name] = algorithm_results['fronts'][best_idx]
            elif algorithm_results['fronts']:
                best_fronts[algorithm_name] = algorithm_results['fronts'][0]
        elif 'hv' in metrics:
            # 使用HV选择最佳前沿
            valid_indices = [i for i, v in enumerate(algorithm_results['hv']) if not np.isnan(v)]
            if valid_indices:
                best_idx = valid_indices[np.argmax([algorithm_results['hv'][i] for i in valid_indices])]
                best_fronts[algorithm_name] = algorithm_results['fronts'][best_idx]
            elif algorithm_results['fronts']:
                best_fronts[algorithm_name] = algorithm_results['fronts'][0]
        elif algorithm_results['fronts']:
            # 默认使用第一个前沿
            best_fronts[algorithm_name] = algorithm_results['fronts'][0]

    # 生成比较图表
    compare_algorithms(
        problem,
        {alg: {metric: results[alg][metric] for metric in metrics} for alg in results},
        metrics,
        os.path.join(results_dir, "algorithm_comparison.png"),
        title=f"{problem.name} 问题算法性能对比"
    )

    # 可视化Pareto前沿
    if best_fronts:
        visualize_pareto_fronts(
            problem,
            [best_fronts[alg] for alg in algorithm_names],
            algorithm_names,
            os.path.join(results_dir, "pareto_fronts_comparison.png"),
            title=f"{problem.name} 问题Pareto前沿对比"
        )

    # 保存指标统计
    with open(os.path.join(results_dir, "metrics_statistics.txt"), 'w') as f:
        f.write(f"{problem.name} 问题性能指标统计\n")
        f.write("=" * 50 + "\n\n")

        for metric in metrics:
            f.write(f"{metric.upper()} 指标:\n")
            f.write("-" * 30 + "\n")

            for alg in results:
                values = [v for v in results[alg][metric] if not np.isnan(v)]
                if values:
                    mean = np.mean(values)
                    std = np.std(values)
                    min_val = np.min(values)
                    max_val = np.max(values)

                    f.write(f"{alg}:\n")
                    f.write(f"  平均值: {mean:.6f}\n")
                    f.write(f"  标准差: {std:.6f}\n")
                    f.write(f"  最小值: {min_val:.6f}\n")
                    f.write(f"  最大值: {max_val:.6f}\n")
                    f.write(f"  有效样本数: {len(values)}/{n_runs}\n\n")
                else:
                    f.write(f"{alg}: 无有效结果\n\n")

            f.write("\n")

        # 运行时间统计
        f.write("运行时间统计(秒):\n")
        f.write("-" * 30 + "\n")

        for alg in results:
            times = results[alg]['runtimes']
            f.write(f"{alg}:\n")
            f.write(f"  平均值: {np.mean(times):.4f}\n")
            f.write(f"  标准差: {np.std(times):.4f}\n")
            f.write(f"  最小值: {np.min(times):.4f}\n")
            f.write(f"  最大值: {np.max(times):.4f}\n\n")

    return results


# ====================== 主函数 ======================

def main():
    """主函数"""
    # 设置随机种子以确保结果可重复
    random.seed(42)
    np.random.seed(42)

    # 创建灌溉系统优化问题
    irrigation_problem = IrrigationProblem(
        node_count=23,          # 节点数量
        first_pressure=49.62,   # 首节点压力
        first_diameter=500,     # 首管径
        lgz1=8,                 # 轮灌组参数1
        lgz2=4,                 # 轮灌组参数2
        baseline_pressure=23.8, # 基准压力
        max_variance=5.0        # 最大水头均方差
    )

    # 定义要比较的算法
    algorithms = [
        EnhancedMOPSO,
        MOPSO
    ]

    # 定义算法参数
    algorithm_params = {
        'EnhancedMOPSO': {
            'pop_size': 30,
            'max_iterations': 50,
            'w_init': 0.9,
            'w_end': 0.4,
            'c1_init': 2.5,
            'c1_end': 0.5,
            'c2_init': 0.5,
            'c2_end': 2.5,
            'archive_size': 100,
            'mutation_rate': 0.1
        },
        'MOPSO': {
            'pop_size': 30,
            'max_iterations': 50,
            'w': 0.7,
            'c1': 1.5,
            'c2': 1.5
        }
    }

    # 创建结果目录
    results_dir = 'irrigation_optimization_results'

    # 运行比较实验
    results = run_comparison(
        irrigation_problem,
        algorithms,
        algorithm_params,
        n_runs=3,
        metrics=['sp', 'hv'],
        results_dir=results_dir
    )

    print(f"优化完成，结果已保存至 {results_dir}")


if __name__ == "__main__":
    main()