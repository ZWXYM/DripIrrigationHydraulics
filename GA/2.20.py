import math
import random
import logging
import time
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from deap import base, creator, tools, algorithms

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
DEFAULT_INLET_PRESSURE = 50  # 默认入口水头压力（米）
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
    def __init__(self, node_count, node_spacing=DEFAULT_NODE_SPACING,
                 first_segment_length=DEFAULT_FIRST_SEGMENT_LENGTH,
                 submain_length=DEFAULT_SUBMAIN_LENGTH,
                 lateral_layout="single"):
        """初始化灌溉系统"""
        self.node_count = node_count
        self.node_spacing = node_spacing
        self.first_segment_length = first_segment_length
        self.submain_length = submain_length
        self.lateral_layout = lateral_layout

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
            'pressure_variance': self._get_pressure_variance(),
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
                segment["pressure"] = DEFAULT_INLET_PRESSURE
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
        pressures = [submain["outlet_pressure"] - PRESSURE_BASELINE
                     for submain in self.submains if submain["flow_rate"] > 0]
        if not pressures:
            return 0
        mean_pressure = sum(pressures) / len(pressures)
        return sum((p - mean_pressure) ** 2 for p in pressures) / len(pressures)

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


class NSGAOptimizationTracker:
    def __init__(self, show_dynamic_plots=False, auto_save=False):
        self.generations = []
        self.best_costs = []
        self.best_variances = []
        self.all_costs = []
        self.all_variances = []
        self.all_generations = []

        # 动态图表显示设置
        self.show_dynamic_plots = show_dynamic_plots
        self.auto_save = auto_save  # 新增控制自动保存的选项
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

        # 初始化2D图表
        self.fig_2d, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        self.ax1.set_ylabel('系统成本 (元)', fontsize=12)
        self.ax1.set_title('NSGA-II算法优化迭代曲线', fontsize=14)
        self.ax1.grid(True, linestyle='--', alpha=0.7)

        self.ax2.set_xlabel('迭代代数', fontsize=12)
        self.ax2.set_ylabel('水头均方差', fontsize=12)
        self.ax2.grid(True, linestyle='--', alpha=0.7)

        # 创建空的线条对象
        self.line1, = self.ax1.plot([], [], 'b-o', linewidth=2)
        self.line2, = self.ax2.plot([], [], 'r-o', linewidth=2)

        # 初始化3D图表
        self.fig_3d = plt.figure(figsize=(12, 10))
        self.ax_3d = self.fig_3d.add_subplot(111, projection='3d')
        self.ax_3d.set_xlabel('系统成本 (元)', fontsize=12)
        self.ax_3d.set_ylabel('水头均方差', fontsize=12)
        self.ax_3d.set_zlabel('迭代代数', fontsize=12)
        self.ax_3d.set_title('NSGA-II算法优化3D进度图', fontsize=14)

        # 显示图表
        self.fig_2d.canvas.draw()
        self.fig_2d.canvas.flush_events()
        self.fig_3d.canvas.draw()
        self.fig_3d.canvas.flush_events()

    def select_best_solution_by_marginal_improvement(self, solutions):
        """选择在相同成本递减变化情况下节点水头均方差下降最快的解"""
        # 按成本升序排序解集
        sorted_solutions = sorted(solutions, key=lambda ind: ind.fitness.values[0])

        # 如果只有一个解，直接返回
        if len(sorted_solutions) <= 1:
            return sorted_solutions[0]

        # 计算每对相邻解之间的边际改进率
        marginal_improvements = []
        for i in range(1, len(sorted_solutions)):
            prev_cost = sorted_solutions[i - 1].fitness.values[0]
            prev_variance = sorted_solutions[i - 1].fitness.values[1]
            curr_cost = sorted_solutions[i].fitness.values[0]
            curr_variance = sorted_solutions[i].fitness.values[1]

            # 计算成本变化和方差变化
            cost_diff = curr_cost - prev_cost
            variance_diff = curr_variance - prev_variance

            # 如果成本增加，跳过
            if cost_diff >= 0:
                continue

            # 计算边际改进率
            if cost_diff < 0:
                marginal_improvement = variance_diff / abs(cost_diff)
                marginal_improvements.append((i, marginal_improvement))

        # 如果没有找到有效的边际改进，返回成本最低的解
        if not marginal_improvements:
            return sorted_solutions[0]

        # 找出边际改进率最大的解
        best_idx, _ = min(marginal_improvements, key=lambda x: x[1])

        return sorted_solutions[best_idx]

    def update(self, generation, population):
        """更新跟踪器的数据"""
        self.generations.append(generation)

        # 获取有效解
        valid_solutions = [ind for ind in population if np.all(np.isfinite(ind.fitness.values))]

        if valid_solutions:
            # 选择代表性解
            best_solution = self.select_best_solution_by_marginal_improvement(valid_solutions)

            # 记录代表性解的成本和方差
            cost = best_solution.fitness.values[0]
            variance = best_solution.fitness.values[1]

            self.best_costs.append(cost)
            self.best_variances.append(variance)

            # 收集所有解的数据用于3D可视化
            for solution in valid_solutions:
                self.all_costs.append(solution.fitness.values[0])
                self.all_variances.append(solution.fitness.values[1])
                self.all_generations.append(generation)

            # 如果启用了动态图表，更新图表
            if self.show_dynamic_plots and generation % 5 == 0:  # 每5代更新一次图表，可调整
                try:
                    self._update_plots()
                except Exception as e:
                    print(f"图表更新时出错: {e}")
    def _update_plots(self):
        """更新动态图表"""
        if not self.generations:
            return

        try:
            # 更新2D图表中的数据
            self.line1.set_data(self.generations, self.best_costs)
            self.line2.set_data(self.generations, self.best_variances)

            # 调整坐标轴范围
            self.ax1.relim()
            self.ax1.autoscale_view()
            self.ax2.relim()
            self.ax2.autoscale_view()

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

            self.ax_3d.set_xlabel('系统成本 (元)', fontsize=12)
            self.ax_3d.set_ylabel('水头均方差', fontsize=12)
            self.ax_3d.set_zlabel('迭代代数', fontsize=12)
            self.ax_3d.set_title('NSGA-II算法优化3D进度图', fontsize=14)

            # 刷新图表 - 使用更可靠的方法
            self.fig_2d.canvas.draw_idle()
            self.fig_3d.canvas.draw_idle()

            # 处理事件队列
            self.fig_2d.canvas.flush_events()
            self.fig_3d.canvas.flush_events()

            # 短暂暂停以确保图形更新
            plt.pause(0.001)

        except Exception as e:
            print(f"图表更新过程中出错: {e}")

    def finalize_plots(self):
        """优化结束后最终更新图表"""
        if not self.show_dynamic_plots:
            # 如果没有启用动态图表，创建新图表
            self.plot_2d_curves()
            self.plot_3d_progress()
            return

        # 更新已有的动态图表
        try:
            # 更新2D图表
            self.line1.set_data(self.generations, self.best_costs)
            self.line2.set_data(self.generations, self.best_variances)

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
            cbar.set_label('迭代代数', fontsize=12)

            self.ax_3d.set_xlabel('系统成本 (元)', fontsize=12)
            self.ax_3d.set_ylabel('水头均方差', fontsize=12)
            self.ax_3d.set_zlabel('迭代代数', fontsize=12)
            self.ax_3d.set_title('NSGA-II算法优化3D进度图', fontsize=14)

            # 如果需要，保存图表
            if self.auto_save:
                self.fig_2d.savefig('NSGA_2d_curves.png', dpi=300, bbox_inches='tight')
                self.fig_3d.savefig('NSGA_3d_progress.png', dpi=300, bbox_inches='tight')

            # 刷新图表
            self.fig_2d.canvas.draw()
            self.fig_3d.canvas.draw()

            # 切换到阻塞模式显示图表
            plt.ioff()  # 关闭交互模式
            plt.show(block=True)  # 阻塞直到所有窗口关闭

        except Exception as e:
            print(f"最终图表更新时出错: {e}")

    def plot_2d_curves(self):
        """绘制最终的2D迭代曲线"""
        if not self.generations:
            print("没有数据可供绘图")
            return

        # 确保在非交互模式下创建新图表
        plt.ioff()

        # 创建新图表
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

        # 成本曲线
        ax1.plot(self.generations, self.best_costs, 'b-o', linewidth=2)
        ax1.set_ylabel('系统成本 (元)', fontsize=12)
        ax1.set_title('NSGA-II算法优化迭代曲线', fontsize=14)
        ax1.grid(True, linestyle='--', alpha=0.7)

        # 方差曲线
        ax2.plot(self.generations, self.best_variances, 'r-o', linewidth=2)
        ax2.set_xlabel('迭代代数', fontsize=12)
        ax2.set_ylabel('水头均方差', fontsize=12)
        ax2.grid(True, linestyle='--', alpha=0.7)

        plt.tight_layout()

        # 如果需要，保存图表
        if self.auto_save:
            plt.savefig('NSGA_2d_curves.png', dpi=300, bbox_inches='tight')

        # 显示图表
        plt.ion()  # 重新开启交互模式以便能打开多个图表
        plt.show(block=False)

    def plot_3d_progress(self):
        """绘制最终的3D进度图"""
        if not self.all_generations:
            print("没有数据可供绘图")
            return

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
            c=self.all_generations,  # 按代数上色
            cmap='viridis',
            s=50,
            alpha=0.6
        )

        # 添加颜色条
        cbar = fig.colorbar(scatter, ax=ax, pad=0.1)
        cbar.set_label('迭代代数', fontsize=12)

        # 设置轴标签
        ax.set_xlabel('系统成本 (元)', fontsize=12)
        ax.set_ylabel('水头均方差', fontsize=12)
        ax.set_zlabel('迭代代数', fontsize=12)

        # 设置图表标题
        ax.set_title('NSGA-II算法优化3D进度图', fontsize=14)

        # 调整视角
        ax.view_init(elev=30, azim=45)

        # 如果需要，保存图表
        if self.auto_save:
            plt.savefig('NSGA_3d_progress.png', dpi=300, bbox_inches='tight')

        # 显示图表
        plt.ion()  # 重新开启交互模式以便能打开多个图表
        plt.show(block=False)


def multi_objective_optimization(irrigation_system, lgz1, lgz2):
    """多目标优化函数"""
    if hasattr(creator, "FitnessMulti"):
        del creator.FitnessMulti
    if hasattr(creator, "Individual"):
        del creator.Individual

    # 创建跟踪器
    tracker = NSGAOptimizationTracker(show_dynamic_plots=True, auto_save=False)  # 启用动态图表、关闭自动保存

    # 初始化轮灌组配置
    group_count = irrigation_system.initialize_irrigation_groups(lgz1, lgz2)

    def adjust_pipe_diameters(individual, active_nodes):
        """根据水力要求调整管径"""
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
                    if submain["outlet_pressure"] < DEFAULT_DRIP_LINE_INLET_PRESSURE:
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
                return True, individual

            if not diameter_increased:
                return False, individual

            attempt += 1

        return False, individual

    def evaluate(individual):
        """评估函数"""
        try:
            total_cost = 0
            pressure_variances = []

            for group_idx in range(group_count):
                active_nodes = irrigation_system.irrigation_groups[group_idx]

                # 调整管径直到满足要求或无法继续调整
                success, adjusted_individual = adjust_pipe_diameters(individual.copy(), active_nodes)
                if success:
                    individual = adjusted_individual
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
    population = toolbox.population(n=100)
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

    # 更新跟踪器
    tracker.update(0, population)

    # 演化过程
    for gen in range(1, 50):
        offspring = algorithms.varOr(population, toolbox, 100, 0.7, 0.2)
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        population = toolbox.select(population + offspring, 100)
        record = stats.compile(population)
        logbook.record(gen=gen, **record)

        # 更新跟踪器
        tracker.update(gen, population)

        # 输出进度
        if gen % 10 == 0:
            logging.info(f"Generation {gen}: {record}")

    # 绘制图表
    tracker.finalize_plots()

    return tools.sortNondominated(population, len(population), first_front_only=True)[0], logbook

def print_detailed_results(irrigation_system, best_individual, lgz1, lgz2,
                           output_file="optimization_results_NSGAⅡ_DAN.txt"):
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

        # 存储所有压力差值用于全局统计
        all_pressure_margins = []

        # 对每个轮灌组输出信息
        for group_idx, nodes in enumerate(irrigation_system.irrigation_groups):
            write_line(f"\n=== 轮灌组 {group_idx + 1} ===")
            write_line(f"启用节点: {nodes}")

            # 更新水力计算
            irrigation_system._update_flow_rates(nodes)
            irrigation_system._calculate_hydraulics()
            irrigation_system._calculate_pressures()

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

        # 计算并输出全局统计指标
        if all_pressure_margins:
            global_avg_margin = sum(all_pressure_margins) / len(all_pressure_margins)
            global_variance = sum((p - global_avg_margin) ** 2 for p in all_pressure_margins) / len(
                all_pressure_margins)
            global_std_dev = global_variance ** 0.5

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


def visualize_pareto_front(pareto_front):
    """可视化Pareto前沿"""
    try:
        if not pareto_front:
            print("没有可视化的解集")
            return

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

        plt.title('多目标梳齿NSGAⅡ管网优化Pareto前沿', fontsize=12, pad=15)
        plt.xlabel('系统成本', fontsize=10)
        plt.ylabel('压力方差', fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.ticklabel_format(style='sci', scilimits=(-2, 3), axis='both')
        plt.legend(loc='upper right')
        plt.tight_layout()
        plt.show()
        plt.close()

    except Exception as e:
        logging.error(f"可视化过程中发生错误: {str(e)}")
        print(f"可视化失败: {str(e)}")


def select_best_solution_by_marginal_improvement(solutions):
    """
    选择在相同成本递减变化情况下节点水头均方差下降最快的解

    参数:
    solutions: 有效的Pareto解列表，每个元素是NSGA-II算法生成的带有fitness.values属性的对象

    返回:
    选中的最佳解决方案
    """
    # 按成本升序排序解集
    sorted_solutions = sorted(solutions, key=lambda ind: ind.fitness.values[0])

    # 如果只有一个解，直接返回
    if len(sorted_solutions) <= 1:
        return sorted_solutions[0]

    # 计算每对相邻解之间的边际改进率
    marginal_improvements = []
    for i in range(1, len(sorted_solutions)):
        prev_cost = sorted_solutions[i - 1].fitness.values[0]
        prev_variance = sorted_solutions[i - 1].fitness.values[1]
        curr_cost = sorted_solutions[i].fitness.values[0]
        curr_variance = sorted_solutions[i].fitness.values[1]

        # 计算成本变化和方差变化
        cost_diff = curr_cost - prev_cost
        variance_diff = curr_variance - prev_variance

        # 如果成本增加，跳过
        if cost_diff >= 0:
            continue

        # 计算边际改进率：方差减少量/成本增加量的绝对值
        if cost_diff < 0:
            # 我们希望方差减少（负值）除以成本增加（负值）得到正值
            marginal_improvement = variance_diff / abs(cost_diff)
            marginal_improvements.append((i, marginal_improvement))

    # 如果没有找到有效的边际改进，返回成本最低的解
    if not marginal_improvements:
        return sorted_solutions[0]

    # 找出边际改进率最大的解（因为方差变化可能为负，我们寻找最小值）
    best_idx, _ = min(marginal_improvements, key=lambda x: x[1])

    # 返回该解
    return sorted_solutions[best_idx]



def main():
    """主函数"""
    try:
        # 创建灌溉系统
        irrigation_system = IrrigationSystem(
            node_count=23,
        )

        # 设置轮灌参数
        best_lgz1, best_lgz2 = 6, 4
        logging.info("开始进行多目标优化...")

        # 执行优化
        start_time = time.time()
        pareto_front, logbook = multi_objective_optimization(irrigation_system, best_lgz1, best_lgz2)
        end_time = time.time()

        logging.info(f"优化完成，耗时: {end_time - start_time:.2f}秒")

        # 输出结果
        if pareto_front:
            valid_solutions = [ind for ind in pareto_front if np.all(np.isfinite(ind.fitness.values))]
            if valid_solutions:
                best_solution = select_best_solution_by_marginal_improvement(valid_solutions)
                print_detailed_results(irrigation_system, best_solution, best_lgz1, best_lgz2)
                visualize_pareto_front(pareto_front)
                logging.info("结果已保存并可视化完成")

            else:
                logging.error("未找到有效的解决方案")
        else:
            logging.error("多目标优化未能产生有效的Pareto前沿")

    except Exception as e:
        logging.error(f"程序执行出错: {str(e)}")
        raise


if __name__ == "__main__":
    # 设置随机种子以确保结果可重复
    random.seed(42)
    np.random.seed(42)
    # 执行主程序
    main()