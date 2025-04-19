import numpy as np
import matplotlib.pyplot as plt
import random
import time
from matplotlib import rcParams
import platform
import os
from scipy.spatial.distance import cdist, pdist
import logging


# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def merge_pareto_fronts(algorithm_name, results_dir='pareto_comparison_results'):
    """
    合并多次运行的帕累托前沿结果为单个文件

    参数:
    algorithm_name: 算法名称
    results_dir: 结果保存目录

    返回:
    合并后的帕累托前沿文件路径
    """
    import os
    import pandas as pd
    import numpy as np
    from datetime import datetime

    # 确保目录存在
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # 寻找所有匹配的CSV文件
    files = [f for f in os.listdir(results_dir)
             if f.startswith(f"{algorithm_name}_pareto_front_") and f.endswith(".csv")]

    if not files:
        print(f"未找到{algorithm_name}的帕累托前沿文件")
        return None

    # 读取并合并所有文件
    all_solutions = []
    for file in files:
        file_path = os.path.join(results_dir, file)
        try:
            df = pd.read_csv(file_path)
            all_solutions.append(df)
        except Exception as e:
            print(f"读取文件{file_path}失败: {str(e)}")

    if not all_solutions:
        print(f"没有成功读取任何{algorithm_name}的帕累托前沿文件")
        return None

    # 合并所有解集
    merged_df = pd.concat(all_solutions, ignore_index=True)

    # 去除重复解
    merged_df = merged_df.drop_duplicates()

    # 提取非支配解集
    solutions = merged_df.values
    non_dominated = []

    for i, sol1 in enumerate(solutions):
        is_dominated = False
        for j, sol2 in enumerate(solutions):
            if i != j:
                # 检查sol2是否支配sol1
                if np.all(sol2 <= sol1) and np.any(sol2 < sol1):
                    is_dominated = True
                    break
        if not is_dominated:
            non_dominated.append(sol1)

    # 创建新的DataFrame
    final_df = pd.DataFrame(non_dominated, columns=merged_df.columns)

    # 保存合并后的结果
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = f"{algorithm_name}_merged_pareto_front_{timestamp}.csv"
    filepath = os.path.join(results_dir, filename)
    final_df.to_csv(filepath, index=False)

    print(f"{algorithm_name}的多次运行帕累托前沿已合并，共{len(final_df)}个非支配解，保存到: {filepath}")

    # 删除原始文件
    for file in files:
        try:
            os.remove(os.path.join(results_dir, file))
            print(f"已删除原始文件: {file}")
        except Exception as e:
            print(f"删除文件{file}失败: {str(e)}")

    return filepath


def save_algorithm_results(algorithm, algorithm_name, pareto_dir='pareto_comparison_results'):
    """
    统一保存算法的帕累托前沿结果
    """
    try:
        from pareto_show import save_pareto_front, save_pareto_solutions

        # 确保目录存在
        if not os.path.exists(pareto_dir):
            os.makedirs(pareto_dir)

        # 根据不同算法类型提取帕累托前沿和解集
        if algorithm_name == "MOPSO":
            # MOPSO基于存档的算法
            if not algorithm.archive:
                print(f"{algorithm_name}没有找到有效的帕累托解集")
                return

            pareto_front = np.array([p.best_fitness for p in algorithm.archive if hasattr(p, 'best_fitness')])
            pareto_solutions = algorithm.archive

        elif algorithm_name == "NSGAII":
            # NSGAII特定处理
            front = algorithm._get_pareto_front()
            if len(front) == 0:
                print(f"{algorithm_name}没有找到有效的帕累托解集")
                return

            pareto_front = front
            pareto_solutions = []
            fronts = algorithm._fast_non_dominated_sort(algorithm.population)

            for individual in fronts[0]:
                solution = type('Solution', (), {})
                solution.position = individual['x']
                solution.best_position = individual['x']
                solution.fitness = individual['objectives']
                solution.best_fitness = individual['objectives']
                pareto_solutions.append(solution)

        elif algorithm_name == "MOEAD":
            # MOEAD特定处理
            # 手动提取非支配解集
            population = algorithm.population
            objectives = np.array([ind['objectives'] for ind in population])

            # 找出非支配解
            non_dominated_indices = []
            for i in range(len(population)):
                is_dominated = False
                for j in range(len(population)):
                    if i != j:
                        if algorithm._dominates(objectives[j], objectives[i]):
                            is_dominated = True
                            break
                if not is_dominated:
                    non_dominated_indices.append(i)

            # 确保至少有一个解
            if not non_dominated_indices:
                # 如果没有非支配解，选择weighted sum最小的解
                weighted_sum = np.sum(objectives, axis=1)
                non_dominated_indices = [np.argmin(weighted_sum)]

            # 构建帕累托前沿和解集
            pareto_front = objectives[non_dominated_indices]
            pareto_solutions = []

            for idx in non_dominated_indices:
                solution = type('Solution', (), {})
                solution.position = population[idx]['x']
                solution.best_position = population[idx]['x']
                solution.fitness = population[idx]['objectives']
                solution.best_fitness = population[idx]['objectives']
                pareto_solutions.append(solution)

        elif algorithm_name == "SPEA2":
            # SPEA2特定处理
            # 直接从存档中提取适应度小于1的个体
            non_dominated = [ind for ind in algorithm.archive if ind['fitness'] < 1.0]

            # 如果存档中没有适应度小于1的解，则检查内部种群
            if not non_dominated:
                # 重新计算种群适应度
                algorithm._calculate_fitness(algorithm.population)
                non_dominated = [ind for ind in algorithm.population if ind['fitness'] < 1.0]

                # 如果仍然没有找到非支配解，选择适应度最小的几个解
                if not non_dominated:
                    sorted_pop = sorted(algorithm.population, key=lambda x: x['fitness'])
                    non_dominated = sorted_pop[:min(10, len(sorted_pop))]

            if not non_dominated:
                print(f"{algorithm_name}没有找到有效的帕累托解集")
                return

            pareto_front = np.array([ind['objectives'] for ind in non_dominated])

            # 创建解对象
            pareto_solutions = []
            for individual in non_dominated:
                solution = type('Solution', (), {})
                solution.position = individual['x']
                solution.best_position = individual['x']
                solution.fitness = individual['objectives']
                solution.best_fitness = individual['objectives']
                pareto_solutions.append(solution)
        else:
            print(f"未知的算法类型: {algorithm_name}")
            return

        # 保存帕累托前沿和解集
        if len(pareto_front) > 0:
            save_pareto_front(pareto_front, algorithm_name)
            save_pareto_solutions(pareto_solutions, algorithm_name)
            print(f"{algorithm_name}帕累托解集已成功保存")
        else:
            print(f"{algorithm_name}没有找到有效的帕累托解集")

    except Exception as e:
        print(f"保存{algorithm_name}帕累托解集时出错: {str(e)}")
        import traceback
        traceback.print_exc()


# ====================== 灌溉系统问题适配 ======================

class IrrigationProblem:
    """灌溉系统优化问题包装类"""

    def __init__(self, node_count=23, first_pressure=49.62, first_diameter=500,
                 lgz1=8, lgz2=4, baseline_pressure=23.8, max_variance=10):
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
            available_first_diameters = [d for d in self.pipe_specs["submain"]["diameters"] if
                                         d <= main_connection_diameter]

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
    """基础多目标粒子群优化算法 (支持动态参数版本)"""

    # 修改 __init__ 方法以接受动态参数范围
    def __init__(self, problem, pop_size=100, max_iterations=100,
                 w_init=0.9, w_end=0.4,  # 惯性权重初始/结束值
                 c1_init=1.5, c1_end=1.5,  # 个体学习因子初始/结束值 (默认保持不变)
                 c2_init=1.5, c2_end=1.5,  # 社会学习因子初始/结束值 (默认保持不变)
                 use_archive=True, archive_size=100):  # 添加 archive_size
        """
        初始化MOPSO算法
        problem: 优化问题实例
        pop_size: 种群大小
        max_iterations: 最大迭代次数
        w_init, w_end: 惯性权重的初始和结束值
        c1_init, c1_end: 个体学习因子的初始和结束值
        c2_init, c2_end: 社会学习因子的初始和结束值
        use_archive: 是否使用外部存档
        archive_size: 存档大小限制
        """
        self.problem = problem
        self.pop_size = pop_size
        self.max_iterations = max_iterations
        # 存储动态参数范围
        self.w_init = w_init
        self.w_end = w_end
        self.c1_init = c1_init
        self.c1_end = c1_end
        self.c2_init = c2_init
        self.c2_end = c2_end
        # 其他参数
        self.use_archive = use_archive
        self.archive_size = archive_size  # 确保处理 archive_size

        # 粒子群和外部存档
        self.particles = []
        self.archive = []
        # 保持与原MOPSO相同的领导者选择和存档修剪逻辑
        self.leader_selector = self._crowding_distance_leader
        # self.archive_size = 100 # archive_size 从参数传入

        # 性能指标跟踪
        self.tracking = {
            'iterations': [],
            'fronts': [],
            'metrics': {
                'igdf': [], 'igdx': [], 'rpsp': [], 'hv': [], 'sp': []
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
            # --- 修改 TP9/TP10 后 evaluate 的返回值 ---
            evaluation_result = self.problem.evaluate(particle.position)
            if isinstance(evaluation_result, tuple) and len(evaluation_result) == 2:
                # 如果返回的是 (objectives, constraints)，只取 objectives
                objectives = evaluation_result[0]
            else:
                # 否则，假设只返回 objectives
                objectives = evaluation_result
            particle.fitness = np.array(objectives)  # 使用 numpy 数组存储适应度
            # --- 修改结束 ---
            particle.best_position = particle.position.copy()  # 初始化 best_position
            particle.best_fitness = particle.fitness.copy()  # 使用 fitness 初始化 best_fitness

        # 初始化外部存档
        if self.use_archive:
            self._update_archive()  # 使用初始 pbest 更新存档

        # 优化迭代
        for iteration in range(self.max_iterations):
            if verbose and iteration % 10 == 0:
                print(f"迭代 {iteration}/{self.max_iterations}，当前存档大小: {len(self.archive)}")

            # --- 计算当前迭代的动态参数 ---
            progress = iteration / self.max_iterations
            current_w = self.w_init - (self.w_init - self.w_end) * progress
            current_c1 = self.c1_init - (self.c1_init - self.c1_end) * progress
            current_c2 = self.c2_init + (self.c2_end - self.c2_init) * progress
            # --- 参数计算结束 ---

            # 对每个粒子
            for particle in self.particles:
                # 选择领导者
                if self.archive and self.use_archive:
                    leader = self.leader_selector(particle)
                else:
                    # 如果没有存档或不使用存档，需要一个备选策略
                    # 可以从种群本身的非支配解中选，或随机选一个粒子
                    # 这里我们调用与 CASMOPSO 类似的内部选择函数 (如果需要，可单独实现)
                    leader = self._select_leader_from_swarm(particle)  # 确保这个方法存在或被正确调用

                # 如果没有领导者可选 (例如初始时存档为空且无法从种群选)
                if leader is None:
                    # 可以让粒子使用自己的 pbest 作为引导，或者跳过更新
                    # 这里选择让粒子使用自己的 pbest
                    leader = particle  # 让它飞向自己的历史最优

                # 更新速度和位置 (使用当前计算出的 w, c1, c2)
                particle.update_velocity(leader.best_position, current_w, current_c1, current_c2)
                particle.update_position()

                # 评估新位置
                # --- 同样处理 evaluate 的返回值 ---
                evaluation_result = self.problem.evaluate(particle.position)
                if isinstance(evaluation_result, tuple) and len(evaluation_result) == 2:
                    objectives = evaluation_result[0]
                else:
                    objectives = evaluation_result
                particle.fitness = np.array(objectives)  # 更新 fitness
                # --- 修改结束 ---

                # 更新个体最优 (pbest)
                # 需要比较 fitness 和 best_fitness
                if self._dominates(particle.fitness, particle.best_fitness):
                    particle.best_position = particle.position.copy()
                    particle.best_fitness = particle.fitness.copy()
                # 如果是非支配关系，可以考虑随机更新或不更新
                elif not self._dominates(particle.best_fitness, particle.fitness):
                    # 如果两个解互不支配，可以随机选择是否更新 pbest
                    if random.random() < 0.5:
                        particle.best_position = particle.position.copy()
                        particle.best_fitness = particle.fitness.copy()

            # 更新外部存档 (使用更新后的 pbest)
            if self.use_archive:
                self._update_archive()

            # 跟踪性能指标
            if tracking and iteration % 10 == 0:
                # 确保 _track_performance 使用的是存档或种群的 pbest
                self._track_performance(iteration)

        # 最终评估
        if tracking:
            self._track_performance(self.max_iterations - 1)

        # 返回Pareto前沿
        return self._get_pareto_front()  # 基于最终存档或种群 pbest

    # --- _update_archive 方法保持不变，但确保它使用 best_fitness ---
    def _update_archive(self):
        """更新外部存档"""
        # 将当前粒子的个体最优位置添加到存档中
        current_pbest_positions = [p.best_position for p in self.particles]
        current_pbest_fitness = [p.best_fitness for p in self.particles]

        combined_solutions = []
        # 添加当前存档
        if self.archive:
            combined_solutions.extend([(p.best_position, p.best_fitness) for p in self.archive])
        # 添加当前种群的 pbest
        combined_solutions.extend(zip(current_pbest_positions, current_pbest_fitness))

        # 提取非支配解来构建新存档
        new_archive_solutions = []
        if combined_solutions:
            positions = np.array([s[0] for s in combined_solutions])
            fitnesses = np.array([s[1] for s in combined_solutions])

            # 查找非支配解的索引
            is_dominated = np.zeros(len(fitnesses), dtype=bool)
            for i in range(len(fitnesses)):
                if is_dominated[i]: continue
                for j in range(i + 1, len(fitnesses)):
                    if is_dominated[j]: continue
                    if self._dominates(fitnesses[i], fitnesses[j]):
                        is_dominated[j] = True
                    elif self._dominates(fitnesses[j], fitnesses[i]):
                        is_dominated[i] = True
                        break  # i被支配，跳出内层循环

            non_dominated_indices = np.where(~is_dominated)[0]

            # 重新创建存档粒子列表
            self.archive = []
            unique_positions = set()  # 用于去重
            for idx in non_dominated_indices:
                pos_tuple = tuple(positions[idx])
                if pos_tuple not in unique_positions:
                    archive_particle = Particle(self.problem.n_var, list(zip(self.problem.xl, self.problem.xu)))
                    archive_particle.position = positions[idx].copy()  # 当前位置设为最优位置
                    archive_particle.best_position = positions[idx].copy()
                    archive_particle.fitness = fitnesses[idx].copy()  # 当前适应度设为最优适应度
                    archive_particle.best_fitness = fitnesses[idx].copy()
                    self.archive.append(archive_particle)
                    unique_positions.add(pos_tuple)

        # 如果存档超过大小限制，使用拥挤度排序保留多样性
        if len(self.archive) > self.archive_size:
            self._prune_archive()

    # --- _prune_archive 方法保持不变 ---
    def _prune_archive(self):
        """使用拥挤度排序修剪存档"""
        if len(self.archive) <= self.archive_size:
            return
        # 使用拥挤度排序保留前N个解
        fitnesses = [a.best_fitness for a in self.archive]
        crowding_distances = self._calculate_crowding_distance(fitnesses)

        # 按拥挤度降序排序
        sorted_indices = np.argsort(crowding_distances)[::-1]
        # 保留前 archive_size 个
        self.archive = [self.archive[i] for i in sorted_indices[:self.archive_size]]

    # --- _crowding_distance_leader 方法保持不变 ---
    def _crowding_distance_leader(self, particle):
        """基于拥挤度选择领导者"""
        if not self.archive:  # 如果存档为空，返回 None 或其他策略
            return None  # 或者返回粒子自身？ particle
        if len(self.archive) == 1:
            return self.archive[0]

        # 随机选择候选 (锦标赛选择)
        tournament_size = min(3, len(self.archive))  # 锦标赛大小
        candidates_idx = np.random.choice(len(self.archive), tournament_size, replace=False)
        candidates = [self.archive[i] for i in candidates_idx]

        # 计算候选的拥挤度
        fitnesses = [c.best_fitness for c in candidates]
        crowding_distances = self._calculate_crowding_distance(fitnesses)

        # 选择拥挤度最大的
        best_idx_in_candidates = np.argmax(crowding_distances)
        return candidates[best_idx_in_candidates]

    # --- 添加 _select_leader_from_swarm (如果需要) ---
    def _select_leader_from_swarm(self, particle):
        """从粒子群的pbest中选择领导者 (如果存档为空或不使用)"""
        # 提取当前种群的所有 pbest fitness
        pbest_fitnesses = [p.best_fitness for p in self.particles]
        pbest_positions = [p.best_position for p in self.particles]

        # 找出非支配的 pbest
        non_dominated_indices = []
        is_dominated = np.zeros(len(pbest_fitnesses), dtype=bool)
        for i in range(len(pbest_fitnesses)):
            if is_dominated[i]: continue
            for j in range(i + 1, len(pbest_fitnesses)):
                if is_dominated[j]: continue
                if self._dominates(pbest_fitnesses[i], pbest_fitnesses[j]):
                    is_dominated[j] = True
                elif self._dominates(pbest_fitnesses[j], pbest_fitnesses[i]):
                    is_dominated[i] = True
                    break
            if not is_dominated[i]:
                non_dominated_indices.append(i)

        if not non_dominated_indices:
            # 如果没有非支配解 (不太可能发生，除非所有解都相同)
            # 返回粒子自身或者随机选一个
            return particle  # 让它飞向自己的历史最优

        # 从非支配的 pbest 中随机选择一个作为领导者
        leader_idx = random.choice(non_dominated_indices)
        # 返回一个临时的 "leader" 对象，包含 best_position
        # 或者直接返回 pbest_position? update_velocity 需要 best_position
        temp_leader = Particle(self.problem.n_var, [])  # 临时对象
        temp_leader.best_position = pbest_positions[leader_idx]
        return temp_leader

    # --- _calculate_crowding_distance 方法保持不变 ---
    def _calculate_crowding_distance(self, fitnesses):
        n = len(fitnesses)
        if n <= 2:
            return [float('inf')] * n
        points = np.array(fitnesses)
        distances = np.zeros(n)
        for i in range(self.problem.n_obj):
            idx = np.argsort(points[:, i])
            distances[idx[0]] = float('inf')
            distances[idx[-1]] = float('inf')
            if n > 2:
                f_range = points[idx[-1], i] - points[idx[0], i]
                if f_range > 1e-8:  # 避免除零
                    for j in range(1, n - 1):
                        distances[idx[j]] += (points[idx[j + 1], i] - points[idx[j - 1], i]) / f_range
        return distances

    # --- _dominates 方法保持不变 ---
    def _dominates(self, fitness1, fitness2):
        """判断fitness1是否支配fitness2"""
        f1 = np.asarray(fitness1)  # 确保是数组
        f2 = np.asarray(fitness2)  # 确保是数组
        # 检查维度是否匹配，以防万一
        if f1.shape != f2.shape:
            print(f"警告: 支配比较时维度不匹配: {f1.shape} vs {f2.shape}")
            return False  # 或者抛出错误
        # 至少一个目标严格更好，且没有目标更差
        return np.all(f1 <= f2) and np.any(f1 < f2)

    # --- _get_pareto_front 方法保持不变 ---
    def _get_pareto_front(self):
        """获取算法生成的Pareto前沿"""
        if self.use_archive and self.archive:
            # 确保返回的是 best_fitness
            return np.array([p.best_fitness for p in self.archive])
        else:
            # 从粒子群的 pbest 中提取非支配解
            pbest_fitnesses = [p.best_fitness for p in self.particles]
            if not pbest_fitnesses: return np.array([])  # 处理空种群

            non_dominated = []
            is_dominated = np.zeros(len(pbest_fitnesses), dtype=bool)
            for i in range(len(pbest_fitnesses)):
                if is_dominated[i]: continue
                for j in range(i + 1, len(pbest_fitnesses)):
                    if is_dominated[j]: continue
                    if self._dominates(pbest_fitnesses[i], pbest_fitnesses[j]):
                        is_dominated[j] = True
                    elif self._dominates(pbest_fitnesses[j], pbest_fitnesses[i]):
                        is_dominated[i] = True
                        break
                if not is_dominated[i]:
                    non_dominated.append(pbest_fitnesses[i])
            return np.array(non_dominated)

    # --- _get_pareto_set 方法保持不变 ---
    def _get_pareto_set(self):
        """获取算法生成的Pareto解集"""
        if self.use_archive and self.archive:
            # 确保返回的是 best_position
            return np.array([p.best_position for p in self.archive])
        else:
            # 从粒子群的 pbest 中提取非支配解
            pbest_fitnesses = [p.best_fitness for p in self.particles]
            pbest_positions = [p.best_position for p in self.particles]
            if not pbest_fitnesses: return np.array([])

            non_dominated_indices = []
            is_dominated = np.zeros(len(pbest_fitnesses), dtype=bool)
            for i in range(len(pbest_fitnesses)):
                if is_dominated[i]: continue
                for j in range(i + 1, len(pbest_fitnesses)):
                    if is_dominated[j]: continue
                    if self._dominates(pbest_fitnesses[i], pbest_fitnesses[j]):
                        is_dominated[j] = True
                    elif self._dominates(pbest_fitnesses[j], pbest_fitnesses[i]):
                        is_dominated[i] = True
                        break
                if not is_dominated[i]:
                    non_dominated_indices.append(i)
            return np.array([pbest_positions[i] for i in non_dominated_indices])

    # MOPSO _track_performance函数
    # MOPSO _track_performance函数基本不需要修改
    def _track_performance(self, iteration):
        """跟踪性能指标"""
        front = self._get_pareto_front()
        self.tracking['iterations'].append(iteration)
        self.tracking['fronts'].append(front)

        # 灌溉问题没有真实前沿
        true_front = self.problem.get_pareto_front()

        # SP指标计算保持不变
        if len(front) > 1:
            sp = self._spacing(front)
            self.tracking['metrics']['sp'].append(sp)
        else:
            self.tracking['metrics']['sp'].append(float('nan'))

        # 超体积指标HV - 确保适用于二维问题
        if len(front) > 0:
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
        sp = np.sqrt(np.sum((distances - d_mean) ** 2) / len(distances))

        return sp

    def _hypervolume(self, front, reference_point):
        """
        计算超体积指标(HV) - 适用于二维问题
        前沿与参考点构成的超体积
        值越大表示质量越高
        """
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


class NSGAII:
    """NSGA-II算法实现"""

    def __init__(self, problem, pop_size=100, max_generations=100,
                 pc=0.9,  # 交叉概率 (Crossover probability)
                 eta_c=20,  # SBX 交叉分布指数 (Distribution index for SBX)
                 pm_ratio=1.0,  # 变异概率因子 (pm = pm_ratio / n_var)
                 eta_m=20):  # 多项式变异分布指数 (Distribution index for polynomial mutation)
        """
        初始化NSGA-II算法
        problem: 优化问题实例
        pop_size: 种群大小
        max_generations: 最大代数
        pc: 模拟二进制交叉 (SBX) 的概率
        eta_c: SBX 的分布指数
        pm_ratio: 变异概率 pm = pm_ratio / n_var (n_var 是变量数)
        eta_m: 多项式变异的分布指数
        """
        self.problem = problem
        self.pop_size = pop_size
        self.max_generations = max_generations
        # --- 存储交叉和变异参数 ---
        self.pc = pc
        self.eta_c = eta_c
        # 计算实际的变异概率 pm (每个变量独立变异的概率)
        self.pm = pm_ratio / self.problem.n_var
        self.eta_m = eta_m
        # --- 参数存储结束 ---

        # 种群
        self.population = None

        # 性能指标跟踪
        self.tracking = {
            'iterations': [], 'fronts': [],
            'metrics': {'igdf': [], 'igdx': [], 'rpsp': [], 'hv': [], 'sp': []}
        }

    def optimize(self, tracking=True, verbose=True):
        """执行优化过程"""
        # 初始化种群
        self.population = self._initialize_population()

        # 评估种群
        self._evaluate_population(self.population)

        # 非支配排序
        fronts = self._fast_non_dominated_sort(self.population)

        # 分配拥挤度 - 添加空前沿检查
        for front in fronts:
            if front:  # 确保前沿不为空
                self._crowding_distance_assignment(front)

        # 迭代优化
        for generation in range(self.max_generations):
            if verbose and generation % 10 == 0:
                print(f"迭代 {generation}/{self.max_generations}")

            # 选择
            parents = self._tournament_selection(self.population)

            # 交叉和变异
            offspring = self._crossover_and_mutation(parents)

            # 评估子代
            self._evaluate_population(offspring)

            # 合并种群
            combined = self.population + offspring

            # 非支配排序
            fronts = self._fast_non_dominated_sort(combined)

            # 分配拥挤度
            for front in fronts:
                if front:  # 确保前沿不为空
                    self._crowding_distance_assignment(front)

            # 环境选择
            self.population = self._environmental_selection(fronts)

            # 跟踪性能指标
            if tracking and generation % 10 == 0:
                self._track_performance(generation)

        # 最终评估
        if tracking:
            self._track_performance(self.max_generations - 1)

        # 返回Pareto前沿
        return self._get_pareto_front()

    def _initialize_population(self):
        """初始化种群"""
        population = []
        for _ in range(self.pop_size):
            # 随机生成个体
            individual = {}
            individual['x'] = np.array(
                [np.random.uniform(low, up) for low, up in zip(self.problem.xl, self.problem.xu)])
            individual['rank'] = None
            individual['crowding_distance'] = None
            individual['objectives'] = None
            population.append(individual)

        return population

    def _evaluate_population(self, population):
        """评估种群"""
        for individual in population:
            if individual['objectives'] is None:
                individual['objectives'] = np.array(self.problem.evaluate(individual['x']))

    def _fast_non_dominated_sort(self, population):
        """快速非支配排序 - 改进版"""
        # 初始化
        fronts = [[]]  # 存储不同等级的前沿
        for p in population:
            p['domination_count'] = 0  # 被多少个体支配
            p['dominated_solutions'] = []  # 支配的个体

            for q in population:
                if self._dominates(p['objectives'], q['objectives']):
                    # p支配q
                    p['dominated_solutions'].append(q)
                elif self._dominates(q['objectives'], p['objectives']):
                    # q支配p
                    p['domination_count'] += 1

            if p['domination_count'] == 0:
                p['rank'] = 0
                fronts[0].append(p)

        # 生成其他前沿
        i = 0
        # 修复：添加边界检查确保i不会超出fronts的范围
        while i < len(fronts):
            next_front = []

            if not fronts[i]:  # 如果当前前沿为空，跳过
                i += 1
                continue

            for p in fronts[i]:
                for q in p['dominated_solutions']:
                    q['domination_count'] -= 1
                    if q['domination_count'] == 0:
                        q['rank'] = i + 1
                        next_front.append(q)

            i += 1
            if next_front:
                fronts.append(next_front)

        # 移除空前沿
        fronts = [front for front in fronts if front]

        return fronts

    def _crowding_distance_assignment(self, front):
        """分配拥挤度 (增强分母稳定性)"""
        if not front:  # 检查 front 是否为空
            return

        n = len(front)
        for p in front:
            p['crowding_distance'] = 0.0  # 确保初始化为浮点数

        # 提取 fitnesses
        fitnesses = np.array([ind['objectives'] for ind in front])

        # 对每个目标
        for m in range(self.problem.n_obj):
            # 按目标排序 (获取排序后的索引)
            sorted_indices = np.argsort(fitnesses[:, m])

            # 边界点设为无穷
            front[sorted_indices[0]]['crowding_distance'] = float('inf')
            front[sorted_indices[-1]]['crowding_distance'] = float('inf')

            # 计算中间点的拥挤度
            if n > 2:
                f_max = fitnesses[sorted_indices[-1], m]
                f_min = fitnesses[sorted_indices[0], m]

                # --- 修改: 为分母添加 epsilon ---
                # norm = f_max - f_min if f_max > f_min else 1.0 # 原来的方式
                epsilon = 1e-9  # 一个很小的值
                norm = (f_max - f_min) + epsilon  # 加上 epsilon 避免严格为0
                # --- 修改结束 ---

                for i in range(1, n - 1):
                    # 使用原始 front 列表中的索引来更新距离
                    prev_idx = sorted_indices[i - 1]
                    next_idx = sorted_indices[i + 1]
                    current_idx = sorted_indices[i]

                    numerator = fitnesses[next_idx, m] - fitnesses[prev_idx, m]
                    front[current_idx]['crowding_distance'] += numerator / norm

    def _tournament_selection(self, population):
        """锦标赛选择"""
        selected = []
        while len(selected) < self.pop_size:
            # 随机选择两个个体
            a = random.choice(population)
            b = random.choice(population)

            # 锦标赛比较
            if (a['rank'] < b['rank']) or \
                    (a['rank'] == b['rank'] and a['crowding_distance'] > b['crowding_distance']):
                selected.append(a.copy())
            else:
                selected.append(b.copy())

        return selected

    def _crossover_and_mutation(self, parents):
        """交叉和变异 - 使用 self 中的参数"""
        offspring = []
        n_var = self.problem.n_var
        xl = self.problem.xl
        xu = self.problem.xu

        # 确保进行偶数次交叉，生成 pop_size 个子代
        parent_indices = list(range(len(parents)))
        random.shuffle(parent_indices)  # 打乱父代顺序

        for i in range(0, self.pop_size, 2):
            # 选择父代索引，处理最后一个父代可能落单的情况
            idx1 = parent_indices[i]
            idx2 = parent_indices[i + 1] if (i + 1) < len(parents) else parent_indices[0]  # 落单则与第一个配对

            # 深拷贝父代以产生子代（避免修改原始父代）
            p1 = parents[idx1].copy()
            p2 = parents[idx2].copy()
            # 确保子代有独立的 'x' 副本
            p1['x'] = parents[idx1]['x'].copy()
            p2['x'] = parents[idx2]['x'].copy()

            # SBX交叉
            # 使用 self.pc 和 self.eta_c
            if random.random() < self.pc:
                for j in range(n_var):
                    if random.random() < 0.5:  # 对每个变量 50% 概率交叉
                        y1, y2 = p1['x'][j], p2['x'][j]
                        if abs(y1 - y2) > 1e-10:
                            if y1 > y2: y1, y2 = y2, y1  # 确保 y1 <= y2

                            rand = random.random()
                            beta = 1.0 + (2.0 * (y1 - xl[j]) / (y2 - y1)) if (y2 - y1) > 1e-10 else 1.0
                            alpha = 2.0 - beta ** -(self.eta_c + 1.0)
                            if rand <= (1.0 / alpha):
                                beta_q = (rand * alpha) ** (1.0 / (self.eta_c + 1.0))
                            else:
                                beta_q = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (self.eta_c + 1.0))

                            c1 = 0.5 * ((1.0 + beta_q) * y1 + (1.0 - beta_q) * y2)
                            c2 = 0.5 * ((1.0 - beta_q) * y1 + (1.0 + beta_q) * y2)

                            # 边界处理
                            c1 = np.clip(c1, xl[j], xu[j])
                            c2 = np.clip(c2, xl[j], xu[j])

                            # 随机分配给子代
                            if random.random() < 0.5:
                                p1['x'][j], p2['x'][j] = c1, c2
                            else:
                                p1['x'][j], p2['x'][j] = c2, c1

            # 多项式变异
            # 使用 self.pm 和 self.eta_m
            for child in [p1, p2]:
                for j in range(n_var):
                    if random.random() < self.pm:  # 使用 self.pm
                        y = child['x'][j]
                        delta1 = (y - xl[j]) / (xu[j] - xl[j]) if (xu[j] - xl[j]) > 1e-10 else 0.5
                        delta2 = (xu[j] - y) / (xu[j] - xl[j]) if (xu[j] - xl[j]) > 1e-10 else 0.5
                        delta1 = np.clip(delta1, 0, 1)  # 确保在[0,1]
                        delta2 = np.clip(delta2, 0, 1)  # 确保在[0,1]

                        rand = random.random()
                        mut_pow = 1.0 / (self.eta_m + 1.0)  # 使用 self.eta_m

                        if rand < 0.5:
                            xy = 1.0 - delta1
                            if xy < 0: xy = 0
                            val = 2.0 * rand + (1.0 - 2.0 * rand) * (xy ** (self.eta_m + 1.0))
                            delta_q = val ** mut_pow - 1.0
                        else:
                            xy = 1.0 - delta2
                            if xy < 0: xy = 0
                            val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * (xy ** (self.eta_m + 1.0))
                            delta_q = 1.0 - val ** mut_pow

                        y = y + delta_q * (xu[j] - xl[j])
                        child['x'][j] = np.clip(y, xl[j], xu[j])  # 边界处理

            # 重置子代的评估状态
            p1['objectives'] = None
            p1['rank'] = None
            p1['crowding_distance'] = None
            p2['objectives'] = None
            p2['rank'] = None
            p2['crowding_distance'] = None

            offspring.append(p1)
            # 确保只添加 pop_size 个子代
            if len(offspring) < self.pop_size:
                offspring.append(p2)

        return offspring[:self.pop_size]  # 返回精确 pop_size 个子代

    def _environmental_selection(self, fronts):
        """环境选择"""
        # 选择下一代种群
        next_population = []
        i = 0

        # 添加完整的前沿 - 增加额外的边界检查
        while i < len(fronts) and fronts[i] and len(next_population) + len(fronts[i]) <= self.pop_size:
            next_population.extend(fronts[i])
            i += 1

        # 处理最后一个前沿
        if len(next_population) < self.pop_size and i < len(fronts) and fronts[i]:
            # 按拥挤度排序
            last_front = sorted(fronts[i], key=lambda x: x['crowding_distance'], reverse=True)

            # 添加拥挤度最大的个体
            next_population.extend(last_front[:self.pop_size - len(next_population)])

        return next_population

    def _get_pareto_front(self):
        """获取算法生成的Pareto前沿"""
        # 提取非支配解
        fronts = self._fast_non_dominated_sort(self.population)
        return np.array([individual['objectives'] for individual in fronts[0]])

    def _get_pareto_set(self):
        """获取算法生成的Pareto解集"""
        # 提取非支配解
        fronts = self._fast_non_dominated_sort(self.population)
        return np.array([individual['x'] for individual in fronts[0]])

    def _dominates(self, obj1, obj2):
        """判断obj1是否支配obj2"""
        return np.all(obj1 <= obj2) and np.any(obj1 < obj2)

    def _track_performance(self, generation):
        """跟踪性能指标 - 修改版"""
        # 获取当前Pareto前沿
        front = self._get_pareto_front()

        # 保存迭代次数和前沿
        self.tracking['iterations'].append(generation)
        self.tracking['fronts'].append(front)

        # 灌溉问题没有真实前沿
        true_front = self.problem.get_pareto_front()

        # 均匀性指标SP
        sp = self._spacing(front)
        self.tracking['metrics']['sp'].append(sp)

        # 超体积指标HV - 适用于灌溉系统二维问题
        if len(front) > 0:
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
        sp = np.sqrt(np.sum((distances - d_mean) ** 2) / len(distances))

        return sp

    def _hypervolume(self, front, reference_point):
        """
        计算超体积指标(HV) - 适用于二维问题
        前沿与参考点构成的超体积
        值越大表示质量越高
        """
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


class SPEA2:
    """强度Pareto进化算法2"""

    def __init__(self, problem, pop_size=200, archive_size=300, max_generations=200):
        """
        大幅增加种群大小和存档大小
        """
        self.problem = problem
        self.pop_size = pop_size  # 增至200
        self.archive_size = archive_size  # 增至300
        self.max_generations = max_generations

        # 种群和存档
        self.population = []
        self.archive = []

        # 性能指标跟踪
        self.tracking = {
            'iterations': [],
            'fronts': [],
            'metrics': {
                'igdf': [], 'igdx': [], 'rpsp': [], 'hv': [], 'sp': []
            }
        }

    def optimize(self, tracking=True, verbose=True):
        """修改优化流程以增强收敛和多样性"""
        # 初始化种群
        self._initialize_population()

        # 初始化存档
        self.archive = []

        # 计算初始适应度
        self._calculate_fitness(self.population + self.archive)

        # 更新存档
        self._update_archive()

        # 迭代优化
        for gen in range(self.max_generations):
            if verbose and gen % 10 == 0:
                print(f"迭代 {gen}/{self.max_generations}，存档大小: {len(self.archive)}")

            # 环境选择
            mating_pool = self._environmental_selection()

            # 产生下一代
            offspring = self._generate_offspring(mating_pool)

            # 替换种群
            self.population = offspring

            # 计算适应度
            self._calculate_fitness(self.population + self.archive)

            # 更新存档
            self._update_archive()

            # 如果解的数量太少，添加更多多样性
            if gen % 20 == 0 and len(self.archive) < 50:
                self._inject_diversity()

            # 跟踪性能指标
            if tracking and gen % 10 == 0:
                self._track_performance(gen)

        # 最终评估
        if tracking:
            self._track_performance(self.max_generations - 1)

        # 返回Pareto前沿
        return self._get_pareto_front()

    def _inject_diversity(self):
        """注入多样性机制"""
        # 生成一些随机解
        random_solutions = []
        for _ in range(int(self.pop_size * 0.2)):  # 20%的种群大小
            x = np.array([np.random.uniform(low, up) for low, up in zip(self.problem.xl, self.problem.xu)])
            objectives = np.array(self.problem.evaluate(x))
            random_solutions.append({
                'x': x,
                'objectives': objectives,
                'fitness': 0.0,
                'strength': 0,
                'raw_fitness': 0.0,
                'distance': 0.0
            })

        # 替换种群中的一部分解
        if self.population:
            replace_indices = np.random.choice(len(self.population), min(len(random_solutions), len(self.population)),
                                               replace=False)
            for i, idx in enumerate(replace_indices):
                self.population[idx] = random_solutions[i]

    def _initialize_population(self):
        """初始化种群"""
        self.population = []

        for _ in range(self.pop_size):
            # 随机生成个体
            x = np.array([np.random.uniform(low, up) for low, up in zip(self.problem.xl, self.problem.xu)])

            # 评估个体
            objectives = np.array(self.problem.evaluate(x))  # 确保是numpy数组

            # 添加到种群
            self.population.append({
                'x': x,
                'objectives': objectives,
                'fitness': 0.0,
                'strength': 0,
                'raw_fitness': 0.0,
                'distance': 0.0
            })

    def _calculate_fitness(self, combined_pop):
        """完全重写的适应度计算方法"""
        n = len(combined_pop)

        # 计算支配关系矩阵
        is_dominated = np.zeros((n, n), dtype=bool)
        for i in range(n):
            for j in range(n):
                if i != j:
                    if self._dominates(combined_pop[i]['objectives'], combined_pop[j]['objectives']):
                        is_dominated[i, j] = True

        # 计算每个个体支配的个体数 (strength)
        for i, p in enumerate(combined_pop):
            p['strength'] = np.sum(is_dominated[i])

        # 计算raw fitness (被支配情况)
        for i, p in enumerate(combined_pop):
            dominated_by = np.where(is_dominated[:, i])[0]
            p['raw_fitness'] = sum(combined_pop[j]['strength'] for j in dominated_by)

        # 新的密度估计方法 - 使用K最近邻
        k = int(np.sqrt(n))  # k值

        # 计算所有个体之间的距离
        objectives = np.array([p['objectives'] for p in combined_pop])
        distances = cdist(objectives, objectives)

        # 对每个个体计算密度
        for i, p in enumerate(combined_pop):
            # 排除自己
            dist_to_others = np.delete(distances[i], i)
            # 找到第k个最近的距离
            if len(dist_to_others) >= k:
                kth_dist = np.sort(dist_to_others)[k - 1]
                # 改进的密度计算公式，调小密度惩罚
                p['distance'] = 1.0 / (kth_dist + 3.0)
            else:
                p['distance'] = 0.0

        # 最终适应度 = raw fitness + distance (降低密度影响)
        for p in combined_pop:
            p['fitness'] = p['raw_fitness'] + 0.5 * p['distance']  # 减小密度影响

    def _update_archive(self):
        """改进的存档更新方法"""
        # 合并种群和存档
        combined = self.population + self.archive

        # 重新计算适应度
        self._calculate_fitness(combined)

        # 选择适应度小于1的个体(非支配解)
        new_archive = [p for p in combined if p['fitness'] < 1.0]

        # 如果非支配解太少
        if len(new_archive) < self.archive_size:
            # 按适应度排序
            remaining = [p for p in combined if p['fitness'] >= 1.0]
            remaining.sort(key=lambda x: x['fitness'])

            # 添加适应度最小的个体
            new_archive.extend(remaining[:self.archive_size - len(new_archive)])

        # 如果非支配解太多
        elif len(new_archive) > self.archive_size:
            # 根据目标空间的多样性排序 - 优先保留边界解
            while len(new_archive) > self.archive_size:
                # 删除最拥挤的解
                self._remove_most_crowded_improved(new_archive)

        # 更新存档
        self.archive = new_archive

    def _remove_most_crowded_improved(self, archive):
        """改进的拥挤度计算和裁剪方法"""
        if len(archive) <= 1:
            return

        # 提取目标值
        objectives = np.array([p['objectives'] for p in archive])

        # 如果是二目标问题，特殊处理以保留边界点
        if objectives.shape[1] == 2:
            # 识别边界解 (每个目标的最小值)
            min_f1_idx = np.argmin(objectives[:, 0])
            min_f2_idx = np.argmin(objectives[:, 1])
            # 保护边界解
            protected_indices = {min_f1_idx, min_f2_idx}

            # 如果所有解都是边界解，随机删除一个
            if len(protected_indices) >= len(archive):
                to_remove = np.random.randint(0, len(archive))
                archive.pop(to_remove)
                return

            # 计算所有可删除解的拥挤度
            min_dist = float('inf')
            to_remove = -1

            for i in range(len(archive)):
                if i in protected_indices:
                    continue

                # 计算到其他解的最小距离
                min_dist_i = float('inf')
                for j in range(len(archive)):
                    if i == j:
                        continue

                    dist = np.sqrt(np.sum((objectives[i] - objectives[j]) ** 2))
                    min_dist_i = min(min_dist_i, dist)

                # 找到拥挤度最大(距离最小)的解
                if min_dist_i < min_dist:
                    min_dist = min_dist_i
                    to_remove = i

            # 删除拥挤度最大的解
            if to_remove >= 0:
                archive.pop(to_remove)
            else:
                # 如果没找到，随机删除
                non_protected = [i for i in range(len(archive)) if i not in protected_indices]
                to_remove = np.random.choice(non_protected)
                archive.pop(to_remove)
        else:
            # 多目标问题使用传统方法
            # 寻找距离最近的一对解
            min_dist = float('inf')
            min_i, min_j = 0, 0

            for i in range(len(archive)):
                for j in range(i + 1, len(archive)):
                    dist = np.sqrt(np.sum((objectives[i] - objectives[j]) ** 2))
                    if dist < min_dist:
                        min_dist = dist
                        min_i, min_j = i, j

            # 决定删除哪一个 - 选择距其他解更近的一个
            i_dist = 0.0
            j_dist = 0.0

            for k in range(len(archive)):
                if k != min_i and k != min_j:
                    i_dist += np.sqrt(np.sum((objectives[min_i] - objectives[k]) ** 2))
                    j_dist += np.sqrt(np.sum((objectives[min_j] - objectives[k]) ** 2))

            # 移除最拥挤的个体
            if i_dist < j_dist:
                archive.pop(min_i)
            else:
                archive.pop(min_j)
    def _remove_most_crowded(self, archive):
        """移除最拥挤的个体"""
        # 计算所有个体间的距离
        if len(archive) <= 1:
            return

        min_dist = float('inf')
        min_i = 0
        min_j = 0

        for i in range(len(archive)):
            i_obj = np.array(archive[i]['objectives'])  # 确保是numpy数组

            for j in range(i + 1, len(archive)):
                j_obj = np.array(archive[j]['objectives'])  # 确保是numpy数组
                dist = np.sqrt(np.sum((i_obj - j_obj) ** 2))
                if dist < min_dist:
                    min_dist = dist
                    min_i = i
                    min_j = j

        # 找到距离其他个体更近的那个
        i_dist = 0.0
        j_dist = 0.0

        for k in range(len(archive)):
            if k != min_i and k != min_j:
                k_obj = np.array(archive[k]['objectives'])  # 确保是numpy数组
                i_obj = np.array(archive[min_i]['objectives'])  # 确保是numpy数组
                j_obj = np.array(archive[min_j]['objectives'])  # 确保是numpy数组

                i_dist += np.sqrt(np.sum((i_obj - k_obj) ** 2))
                j_dist += np.sqrt(np.sum((j_obj - k_obj) ** 2))

        # 移除最拥挤的个体
        if i_dist < j_dist:
            archive.pop(min_i)
        else:
            archive.pop(min_j)

    def _environmental_selection(self):
        """环境选择，选择用于交配的个体"""
        # 创建交配池
        mating_pool = []

        # 二元锦标赛选择
        for _ in range(self.pop_size):
            # 随机选择两个个体
            if len(self.archive) > 0:
                idx1 = np.random.randint(0, len(self.archive))
                idx2 = np.random.randint(0, len(self.archive))

                # 选择适应度更好的个体
                if self.archive[idx1]['fitness'] < self.archive[idx2]['fitness']:
                    mating_pool.append(self.archive[idx1])
                else:
                    mating_pool.append(self.archive[idx2])
            else:
                # 如果存档为空，从种群中选择
                idx1 = np.random.randint(0, len(self.population))
                idx2 = np.random.randint(0, len(self.population))

                if self.population[idx1]['fitness'] < self.population[idx2]['fitness']:
                    mating_pool.append(self.population[idx1])
                else:
                    mating_pool.append(self.population[idx2])

        return mating_pool

    def _generate_offspring(self, mating_pool):
        """完全重写生成子代方法"""
        offspring = []

        # 增加多样性机制 - 注入一些随机解
        random_count = int(self.pop_size * 0.1)  # 10%的随机解
        for _ in range(random_count):
            # 完全随机解
            x = np.array([np.random.uniform(low, up) for low, up in zip(self.problem.xl, self.problem.xu)])
            objectives = np.array(self.problem.evaluate(x))
            offspring.append({
                'x': x,
                'objectives': objectives,
                'fitness': 0.0,
                'strength': 0,
                'raw_fitness': 0.0,
                'distance': 0.0
            })

        # 生成剩余的子代
        for _ in range(self.pop_size - random_count):
            # 选择父代 - 锦标赛选择
            parents = []
            for _ in range(2):
                if len(mating_pool) > 3:
                    # 锦标赛大小
                    tournament_size = 3
                    candidates = np.random.choice(len(mating_pool), tournament_size, replace=False)
                    tournament = [mating_pool[idx] for idx in candidates]
                    # 选择适应度最好的
                    best = min(tournament, key=lambda x: x['fitness'])
                    parents.append(best)
                else:
                    # 如果交配池太小，随机选择
                    parents.append(random.choice(mating_pool))

            # 确保有两个父代
            if len(parents) < 2:
                # 复制第一个父代
                parents.append(parents[0])

            parent1 = parents[0]['x']
            parent2 = parents[1]['x']

            # 模拟二进制交叉(SBX) - 参数调整
            cr = 0.95  # 高交叉率
            eta_c = 10  # 低分布指数 = 高多样性

            child_x = np.zeros(self.problem.n_var)

            for i in range(self.problem.n_var):
                if np.random.random() < cr:
                    # 执行交叉
                    if abs(parent1[i] - parent2[i]) > 1e-10:
                        y1, y2 = min(parent1[i], parent2[i]), max(parent1[i], parent2[i])

                        # 计算beta值
                        beta = 1.0 + 2.0 * (y1 - self.problem.xl[i]) / (y2 - y1)
                        alpha = 2.0 - beta ** (-eta_c - 1)
                        rand = np.random.random()

                        if rand <= 1.0 / alpha:
                            beta_q = (rand * alpha) ** (1.0 / (eta_c + 1))
                        else:
                            beta_q = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta_c + 1))

                        # 生成子代
                        child_x[i] = 0.5 * ((1 + beta_q) * y1 + (1 - beta_q) * y2)

                        # 边界处理
                        child_x[i] = max(self.problem.xl[i], min(self.problem.xu[i], child_x[i]))
                    else:
                        child_x[i] = parent1[i]
                else:
                    # 随机选择一个父代
                    child_x[i] = parent1[i] if np.random.random() < 0.5 else parent2[i]

            # 多项式变异 - 增强变异强度
            pm = 2.0 / self.problem.n_var  # 增加变异概率
            eta_m = 10  # 降低分布指数，增加变异量

            for i in range(self.problem.n_var):
                if np.random.random() < pm:
                    y = child_x[i]
                    delta1 = (y - self.problem.xl[i]) / (self.problem.xu[i] - self.problem.xl[i])
                    delta2 = (self.problem.xu[i] - y) / (self.problem.xu[i] - self.problem.xl[i])

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

                    child_x[i] = y + delta_q * (self.problem.xu[i] - self.problem.xl[i])
                    child_x[i] = max(self.problem.xl[i], min(self.problem.xu[i], child_x[i]))

            # 评估子代
            try:
                child_obj = np.array(self.problem.evaluate(child_x))

                # 添加到子代种群
                offspring.append({
                    'x': child_x,
                    'objectives': child_obj,
                    'fitness': 0.0,
                    'strength': 0,
                    'raw_fitness': 0.0,
                    'distance': 0.0
                })
            except Exception as e:
                print(f"评估子代时出错: {e}")
                # 生成随机解
                x = np.array([np.random.uniform(low, up) for low, up in zip(self.problem.xl, self.problem.xu)])
                objectives = np.array(self.problem.evaluate(x))
                offspring.append({
                    'x': x,
                    'objectives': objectives,
                    'fitness': 0.0,
                    'strength': 0,
                    'raw_fitness': 0.0,
                    'distance': 0.0
                })

        return offspring

    def _dominates(self, obj1, obj2):
        """判断obj1是否支配obj2"""
        return np.all(obj1 <= obj2) and np.any(obj1 < obj2)

    def _get_pareto_front(self):
        """获取Pareto前沿"""
        # 返回存档中的非支配解的目标值
        non_dominated = [ind for ind in self.archive if ind['fitness'] < 1.0]
        if not non_dominated and self.archive:
            # 如果没有严格非支配解，使用整个存档
            non_dominated = self.archive
        return np.array([ind['objectives'] for ind in non_dominated])

    def _get_pareto_set(self):
        """获取Pareto解集"""
        # 返回存档中的非支配解的决策变量
        non_dominated = [ind for ind in self.archive if ind['fitness'] < 1.0]
        if not non_dominated and self.archive:
            # 如果没有严格非支配解，使用整个存档
            non_dominated = self.archive
        return np.array([ind['x'] for ind in non_dominated])

    def _track_performance(self, iteration):
        """跟踪性能指标 - irr风格实现"""
        # 获取当前Pareto前沿
        front = self._get_pareto_front()

        # 保存迭代次数和前沿
        self.tracking['iterations'].append(iteration)
        self.tracking['fronts'].append(front)

        # 均匀性指标SP
        if len(front) > 1:
            sp = self._spacing(front)
            self.tracking['metrics']['sp'].append(sp)
        else:
            self.tracking['metrics']['sp'].append(float('nan'))

        # 超体积指标HV
        if len(front) > 0:
            # 设置参考点
            ref_point = np.max(front, axis=0) * 1.1

            try:
                hv = self._hypervolume(front, ref_point)
                self.tracking['metrics']['hv'].append(hv)
            except Exception as e:
                print(f"HV计算错误: {e}")
                self.tracking['metrics']['hv'].append(float('nan'))
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
        sp = np.sqrt(np.sum((distances - d_mean) ** 2) / len(distances))

        return sp

    def _hypervolume(self, front, reference_point):
        """
        计算超体积指标(HV) - 适用于二维问题
        前沿与参考点构成的超体积
        值越大表示质量越高
        """
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


class MOEAD:
    """基于分解的多目标进化算法(MOEA/D)"""

    def __init__(self, problem, pop_size=500, max_generations=300, T=10, delta=0.7, nr=20):
        """
        大幅增加种群大小、减少邻居数量、增加更新数量
        """
        self.problem = problem
        self.pop_size = pop_size  # 增至500
        self.max_generations = max_generations
        self.T = min(T, pop_size)  # 减少邻居大小以增加多样性
        self.delta = delta  # 降低从邻居选择的概率
        self.nr = nr  # 大幅增加可更新的解数量

        # 额外新增维护一个外部存档
        self.external_archive = []
        self.archive_size = 200  # 外部存档大小

        # 种群和权重等初始化保持不变
        self.population = []
        self.weights = []
        self.neighbors = []
        self.z = None

        # 性能指标跟踪
        self.tracking = {
            'iterations': [],
            'fronts': [],
            'metrics': {
                'igdf': [],
                'igdx': [],
                'rpsp': [],
                'hv': [],
                'sp': []
            }
        }

    def optimize(self, tracking=True, verbose=True):
        """修改优化流程以使用外部存档"""
        # 初始化权重向量和邻居
        self._initialize_weights()
        self._initialize_neighbors()

        # 初始化种群
        self._initialize_population()

        # 初始化理想点
        self.z = np.min([ind['objectives'] for ind in self.population], axis=0)

        # 初始化外部存档
        self._update_external_archive()

        # 迭代优化
        for gen in range(self.max_generations):
            if verbose and gen % 10 == 0:
                print(f"迭代 {gen}/{self.max_generations}，存档大小: {len(self.external_archive)}")

            # 对每个权重向量
            for i in range(self.pop_size):
                # 选择父代
                if np.random.random() < self.delta:
                    # 从邻居中选择
                    p_indices = np.random.choice(self.neighbors[i], 2, replace=False)
                else:
                    # 从整个种群中选择
                    p_indices = np.random.choice(self.pop_size, 2, replace=False)

                # 产生子代
                child = self._reproduction(p_indices)

                # 评估子代
                child_obj = np.array(self.problem.evaluate(child))

                # 更新理想点
                self.z = np.minimum(self.z, child_obj)

                # 更新邻居解
                self._update_neighbors(i, child, child_obj)

            # 周期性注入外部存档中的非支配解到主种群
            if gen % 10 == 0 and self.external_archive:
                # 随机选择一些当前种群个体
                replace_indices = np.random.choice(self.pop_size, min(20, len(self.external_archive)), replace=False)

                # 用存档中的非支配解替换
                for i, idx in enumerate(replace_indices):
                    if i < len(self.external_archive):
                        self.population[idx]['x'] = self.external_archive[i]['x'].copy()
                        self.population[idx]['objectives'] = self.external_archive[i]['objectives'].copy()

            # 更新外部存档
            self._update_external_archive()

            # 跟踪性能指标
            if tracking and gen % 10 == 0:
                self._track_performance(gen)

        # 最终评估
        if tracking:
            self._track_performance(self.max_generations - 1)

        # 从外部存档中返回Pareto前沿
        return np.array([ind['objectives'] for ind in self.external_archive])

    def _initialize_weights(self):
        """完全重写权重向量生成方法"""
        if self.problem.n_obj == 2:  # 二目标问题使用等间隔生成
            self.weights = np.zeros((self.pop_size, self.problem.n_obj))
            for i in range(self.pop_size):
                # 确保边界点也被包含
                if i == 0:
                    self.weights[i] = [0.0001, 0.9999]  # 接近(0,1)
                elif i == self.pop_size - 1:
                    self.weights[i] = [0.9999, 0.0001]  # 接近(1,0)
                else:
                    # 中间点均匀分布
                    weight = i / (self.pop_size - 1)
                    self.weights[i] = [weight, 1 - weight]
        else:
            # 多于两个目标时使用原有方法
            self.weights = self._generate_uniform_weights(self.problem.n_obj, self.pop_size)

    def _generate_uniform_weights(self, n_obj, pop_size):
        """改进的权重向量生成方法，确保生成足够的权重"""

        # 添加组合数计算函数
        def choose(n, k):
            """计算组合数C(n,k)"""
            if k < 0 or k > n:
                return 0
            if k == 0 or k == n:
                return 1

            result = 1
            for i in range(k):
                result = result * (n - i) // (i + 1)
            return result

        if n_obj == 3:
            # 计算合适的H值
            H = 1
            while choose(H + n_obj - 1, n_obj - 1) < pop_size:
                H += 1

            # 生成权重向量
            weights = []
            for i in range(H + 1):
                for j in range(H + 1 - i):
                    k = H - i - j
                    if k >= 0:  # 确保三个权重的和为H
                        weight = np.array([i, j, k], dtype=float) / H
                        weights.append(weight)

            # 如果生成的权重过多，随机选择
            if len(weights) > pop_size:
                indices = np.random.choice(len(weights), pop_size, replace=False)
                weights = [weights[i] for i in indices]

            return np.array(weights)
        else:
            # 对于其他维度，使用简单的均匀生成方法
            weights = []
            for _ in range(pop_size):
                weight = np.random.random(n_obj)
                weight = weight / np.sum(weight)  # 归一化
                weights.append(weight)

            return np.array(weights)

    def _generate_weight_vectors(self, n_obj, pop_size):
        """为三目标问题生成系统的权重向量"""
        # 确定每个维度上的点数
        h = int((pop_size * 2) ** (1.0 / n_obj))

        # 递归生成权重向量
        def _generate_recursive(n_remain, weights, depth, result):
            if depth == n_obj - 1:
                weights[depth] = n_remain / h
                result.append(weights.copy())
                return

            for i in range(n_remain + 1):
                weights[depth] = i / h
                _generate_recursive(n_remain - i, weights, depth + 1, result)

        weights_list = []
        _generate_recursive(h, np.zeros(n_obj), 0, weights_list)

        # 转换为numpy数组
        weights = np.array(weights_list)

        # 如果生成的权重向量过多，随机选择
        if len(weights) > pop_size:
            indices = np.random.choice(len(weights), pop_size, replace=False)
            weights = weights[indices]

        return weights

    def _initialize_neighbors(self):
        """初始化邻居关系，添加安全检查"""
        n = len(self.weights)
        self.neighbors = []

        # 计算权重向量之间的距离
        dist = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                dist[i, j] = np.sum((self.weights[i] - self.weights[j]) ** 2)

        # 调整邻居数量，确保不超过种群大小
        self.T = min(self.T, n - 1)

        # 对每个权重向量找到T个最近的邻居
        for i in range(n):
            self.neighbors.append(np.argsort(dist[i])[:self.T])

    def _initialize_population(self):
        """初始化种群"""
        self.population = []

        for i in range(self.pop_size):
            # 随机生成个体
            x = np.array([np.random.uniform(low, up) for low, up in zip(self.problem.xl, self.problem.xu)])

            # 评估个体
            objectives = np.array(self.problem.evaluate(x))

            # 添加到种群
            self.population.append({
                'x': x,
                'objectives': objectives
            })

    def _reproduction(self, parent_indices):
        """增强繁殖操作以产生更多样的后代"""
        # 获取父代
        parent1 = self.population[parent_indices[0]]['x']
        parent2 = self.population[parent_indices[1]]['x']

        # 模拟二进制交叉参数
        eta_c = 10  # 降低分布指数以增加多样性
        pc = 0.95  # 增加交叉概率

        # 变异参数
        eta_m = 10  # 降低分布指数以增加多样性
        pm = 1.0 / self.problem.n_var  # 每个变量的变异概率

        # 添加额外的多样性机制 - 小概率完全随机解
        if np.random.random() < 0.05:  # 5%的概率产生随机解
            return np.array([np.random.uniform(low, up) for low, up in zip(self.problem.xl, self.problem.xu)])

        # 正常交叉操作
        child = np.zeros(self.problem.n_var)
        for i in range(self.problem.n_var):
            if np.random.random() < pc:
                # 执行交叉
                if abs(parent1[i] - parent2[i]) > 1e-10:
                    y1, y2 = min(parent1[i], parent2[i]), max(parent1[i], parent2[i])

                    # 计算beta值
                    beta = 1.0 + 2.0 * (y1 - self.problem.xl[i]) / (y2 - y1)
                    alpha = 2.0 - beta ** (-eta_c - 1)
                    rand = np.random.random()

                    if rand <= 1.0 / alpha:
                        beta_q = (rand * alpha) ** (1.0 / (eta_c + 1))
                    else:
                        beta_q = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta_c + 1))

                    # 生成子代
                    child[i] = 0.5 * ((1 + beta_q) * y1 + (1 - beta_q) * y2)
                    child[i] = max(self.problem.xl[i], min(self.problem.xu[i], child[i]))
                else:
                    child[i] = parent1[i]
            else:
                # 随机从一个父代选择
                child[i] = parent1[i] if np.random.random() < 0.5 else parent2[i]

        # 变异操作
        for i in range(self.problem.n_var):
            if np.random.random() < pm:
                y = child[i]
                delta1 = (y - self.problem.xl[i]) / (self.problem.xu[i] - self.problem.xl[i])
                delta2 = (self.problem.xu[i] - y) / (self.problem.xu[i] - self.problem.xl[i])

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

                child[i] = y + delta_q * (self.problem.xu[i] - self.problem.xl[i])
                child[i] = max(self.problem.xl[i], min(self.problem.xu[i], child[i]))

        return child

    def _update_neighbors(self, idx, child_x, child_obj):
        """改进邻居更新策略"""
        # 计数更新次数
        count = 0

        # 随机排序邻居
        perm = np.random.permutation(self.neighbors[idx])

        # 对每个邻居
        for j in perm:
            # 计算切比雪夫距离
            old_fit = self._tchebycheff(self.population[j]['objectives'], self.weights[j])
            new_fit = self._tchebycheff(child_obj, self.weights[j])

            # 增加概率性接受机制 - 即使解稍差也有机会被接受
            if new_fit <= old_fit or np.random.random() < 0.05:  # 5%的概率接受较差的解
                self.population[j]['x'] = child_x.copy()
                self.population[j]['objectives'] = child_obj.copy()
                count += 1

            # 限制更新次数
            if count >= self.nr:
                break

    def _tchebycheff(self, objectives, weights):
        """计算切比雪夫距离"""
        return np.max(weights * np.abs(objectives - self.z))

    def _get_pareto_front(self):
        """获取Pareto前沿"""
        # 提取所有目标值
        objectives = np.array([ind['objectives'] for ind in self.population])

        # 提取非支配解
        is_dominated = np.full(self.pop_size, False)

        for i in range(self.pop_size):
            for j in range(self.pop_size):
                if i != j and not is_dominated[j]:
                    if self._dominates(objectives[j], objectives[i]):
                        is_dominated[i] = True
                        break

        # 返回非支配解的目标值
        return objectives[~is_dominated]

    def _get_pareto_set(self):
        """获取Pareto解集"""
        # 提取所有目标值和解
        objectives = np.array([ind['objectives'] for ind in self.population])
        solutions = np.array([ind['x'] for ind in self.population])

        # 提取非支配解
        is_dominated = np.full(self.pop_size, False)

        for i in range(self.pop_size):
            for j in range(self.pop_size):
                if i != j and not is_dominated[j]:
                    if self._dominates(objectives[j], objectives[i]):
                        is_dominated[i] = True
                        break

        # 返回非支配解的解集
        return solutions[~is_dominated]

    def _dominates(self, obj1, obj2):
        """判断obj1是否支配obj2"""
        return np.all(obj1 <= obj2) and np.any(obj1 < obj2)

    def _track_performance(self, iteration):
        """跟踪性能指标 - irr风格实现"""
        # 获取当前Pareto前沿
        front = self._get_pareto_front()

        # 保存迭代次数和前沿
        self.tracking['iterations'].append(iteration)
        self.tracking['fronts'].append(front)

        # 灌溉问题没有真实前沿
        true_front = self.problem.get_pareto_front()

        # 计算SP指标 (均匀性)
        if len(front) > 1:
            # 使用自定义spacing函数
            sp = self._spacing(front)
            self.tracking['metrics']['sp'].append(sp)
        else:
            self.tracking['metrics']['sp'].append(float('nan'))

        # 计算HV指标 - 适用于灌溉系统二维问题
        if len(front) > 0:
            # 设置参考点
            ref_point = np.max(front, axis=0) * 1.1

            try:
                # 使用自定义hypervolume函数
                hv = self._hypervolume(front, ref_point)
                self.tracking['metrics']['hv'].append(hv)
            except Exception as e:
                print(f"HV计算错误: {e}")
                self.tracking['metrics']['hv'].append(float('nan'))
        else:
            self.tracking['metrics']['hv'].append(float('nan'))

    def _update_external_archive(self):
        """新增：维护外部存档"""
        # 将当前种群的所有个体加入存档候选集
        candidates = self.population.copy()

        # 如果已有存档，加入候选集
        if self.external_archive:
            candidates.extend(self.external_archive)

        # 提取所有目标值
        objectives = np.array([ind['objectives'] for ind in candidates])

        # 找出非支配解
        non_dominated_indices = []
        for i in range(len(candidates)):
            is_dominated = False
            for j in range(len(candidates)):
                if i != j and self._dominates(objectives[j], objectives[i]):
                    is_dominated = True
                    break
            if not is_dominated:
                non_dominated_indices.append(i)

        # 创建新的存档
        self.external_archive = [candidates[i] for i in non_dominated_indices]

        # 如果存档过大，使用拥挤度排序修剪
        if len(self.external_archive) > self.archive_size:
            # 提取存档目标值
            archive_obj = np.array([ind['objectives'] for ind in self.external_archive])

            # 计算拥挤度
            crowding_distances = np.zeros(len(self.external_archive))
            # 对每个目标
            for m in range(self.problem.n_obj):
                # 按目标排序
                idx = np.argsort(archive_obj[:, m])
                # 边界点设为无穷
                crowding_distances[idx[0]] = float('inf')
                crowding_distances[idx[-1]] = float('inf')

                # 计算中间点拥挤度
                for i in range(1, len(idx) - 1):
                    crowding_distances[idx[i]] += (archive_obj[idx[i + 1], m] - archive_obj[idx[i - 1], m])

            # 按拥挤度排序
            sorted_indices = np.argsort(crowding_distances)[::-1]
            # 保留最不拥挤的解
            self.external_archive = [self.external_archive[i] for i in sorted_indices[:self.archive_size]]

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
        sp = np.sqrt(np.sum((distances - d_mean) ** 2) / len(distances))

        return sp

    def _hypervolume(self, front, reference_point):
        """
        计算超体积指标(HV) - 适用于二维问题
        前沿与参考点构成的超体积
        值越大表示质量越高
        """
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
                   metrics=['sp', 'hv'], results_dir='results',
                   pareto_dir='pareto_comparison_results'):
    """
    运行算法比较实验
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
            print(f"  运行 {run + 1}/{n_runs}")

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

            # 保存每次运行的帕累托前沿
            save_algorithm_results(algorithm, algorithm_name, pareto_dir)

    return results


# ====================== 主函数 ======================

def main():
    """主函数"""
    # 设置随机种子以确保结果可重复
    random.seed(42)
    np.random.seed(42)

    # 创建灌溉系统优化问题
    irrigation_problem = IrrigationProblem(
        node_count=23,  # 节点数量
        first_pressure=49.62,  # 首节点压力
        first_diameter=500,  # 首管径
        lgz1=8,  # 轮灌组参数1
        lgz2=4,  # 轮灌组参数2
        baseline_pressure=23.8,  # 基准压力
        max_variance=10  # 最大水头均方差
    )

    # 定义要比较的算法（更新为全部四种）
    algorithms = [
        #MOPSO,
        NSGAII,
        MOEAD,
        SPEA2
    ]

    # 定义算法参数（添加MOEAD和SPEA2的参数）
    algorithm_params = {
        "MOPSO": {  # 使用新的动态参数接口
            "pop_size": 200,
            "max_iterations": 200,
            "w_init": 0.9, "w_end": 0.4,  # 动态惯性权重
            "c1_init": 1.5, "c1_end": 1.5,  # (等效于 c1=1.5)
            "c2_init": 1.5, "c2_end": 1.5,  # (等效于 c2=1.5)
            "use_archive": True,
            "archive_size": 150  # 标准存档大小
        },
        "NSGAII": {
            "pop_size": 200,
            "max_generations": 200
        },
        "MOEAD": {
            "pop_size": 300,
            "max_generations": 200,
            "T": 15,
            "delta": 0.8,
            "nr": 10
        },
        "SPEA2": {
            "pop_size": 100,
            "archive_size": 100,
            "max_generations": 200
        }
    }

    # 创建结果目录
    results_dir = 'irrigation_optimization_results'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # 创建Pareto比较目录
    pareto_dir = 'pareto_comparison_results'
    if not os.path.exists(pareto_dir):
        os.makedirs(pareto_dir)

    # 运行比较实验
    results = run_comparison(
        irrigation_problem,
        algorithms,
        algorithm_params,
        n_runs=1,
        metrics=['sp', 'hv'],
        results_dir=results_dir,
        pareto_dir=pareto_dir
    )

    # 合并多次运行的帕累托前沿
    print("\n正在合并各算法的帕累托前沿...")
    for algorithm in algorithms:
        alg_name = algorithm.__name__
        try:
            merge_pareto_fronts(alg_name, pareto_dir)
        except Exception as e:
            print(f"合并{alg_name}帕累托前沿时出错: {str(e)}")

    print(f"优化和分析完成，结果已保存至 {results_dir} 和 {pareto_dir}")


if __name__ == "__main__":
    main()
