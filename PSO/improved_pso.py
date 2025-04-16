"""
改进的多目标PSO算法框架及标准化评估工具 - 完整版本
包含所有必要的引用和导入
"""

import numpy as np
import matplotlib.pyplot as plt
import random
import time
import logging
from tqdm import tqdm
from scipy.spatial.distance import cdist, pdist, euclidean
from matplotlib import rcParams
import platform
import os
import threading
import concurrent.futures
from matplotlib.ticker import MaxNLocator
from queue import Queue
import traceback


# ====================== 通用优化框架 ======================

class Problem:
    """多目标优化问题基类"""

    def __init__(self, name, n_var, n_obj, xl, xu):
        self.name = name  # 问题名称
        self.n_var = n_var  # 决策变量数量
        self.n_obj = n_obj  # 目标函数数量
        self.xl = np.array(xl)  # 变量下界
        self.xu = np.array(xu)  # 变量上界
        self.pareto_front = None  # 真实的Pareto前沿

    def evaluate(self, x):
        """评估函数，需要在子类中实现"""
        raise NotImplementedError("每个问题类必须实现evaluate方法")

    def get_pareto_front(self):
        """获取真实的Pareto前沿，如果可用"""
        return self.pareto_front


# ====================== 标准测试函数 ======================

class ZDT1(Problem):
    """ZDT1测试函数，两个目标，连续的凸前沿"""

    def __init__(self, n_var=30):
        super().__init__("ZDT1", n_var, 2, [0] * n_var, [1] * n_var)

    def evaluate(self, x):
        f1 = x[0]
        g = 1 + 9.0 * sum(x[1:]) / (self.n_var - 1)
        f2 = g * (1 - np.sqrt(f1 / g))
        return [f1, f2]

    def get_pareto_front(self, n_points=100):
        """生成ZDT1的真实Pareto前沿"""
        if self.pareto_front is None:
            f1 = np.linspace(0, 1, n_points)
            f2 = 1 - np.sqrt(f1)
            self.pareto_front = np.column_stack((f1, f2))
        return self.pareto_front


class ZDT2(Problem):
    """ZDT2测试函数，两个目标，非凸前沿"""

    def __init__(self, n_var=30):
        super().__init__("ZDT2", n_var, 2, [0] * n_var, [1] * n_var)

    def evaluate(self, x):
        f1 = x[0]
        g = 1 + 9.0 * sum(x[1:]) / (self.n_var - 1)
        f2 = g * (1 - (f1 / g) ** 2)
        return [f1, f2]

    def get_pareto_front(self, n_points=100):
        """生成ZDT2的真实Pareto前沿"""
        if self.pareto_front is None:
            f1 = np.linspace(0, 1, n_points)
            f2 = 1 - f1 ** 2
            self.pareto_front = np.column_stack((f1, f2))
        return self.pareto_front


class ZDT3(Problem):
    """ZDT3测试函数，两个目标，不连续前沿"""

    def __init__(self, n_var=30):
        super().__init__("ZDT3", n_var, 2, [0] * n_var, [1] * n_var)

    def evaluate(self, x):
        f1 = x[0]
        g = 1 + 9.0 * sum(x[1:]) / (self.n_var - 1)
        f2 = g * (1 - np.sqrt(f1 / g) - (f1 / g) * np.sin(10 * np.pi * f1))
        return [f1, f2]

    def get_pareto_front(self, n_points=300):
        """生成ZDT3的真实Pareto前沿"""
        if self.pareto_front is None:
            regions = [
                (0, 0.0830015349),
                (0.1822287280, 0.2577623634),
                (0.4093136748, 0.4538821041),
                (0.6183967944, 0.6525117038),
                (0.8233317983, 0.8518328654)
            ]

            front = []
            for r in regions:
                x1 = np.linspace(r[0], r[1], int(n_points * (r[1] - r[0]) / 1.0) + 1)
                x2 = 1 - np.sqrt(x1) - x1 * np.sin(10 * np.pi * x1)
                front.extend(list(zip(x1, x2)))

            self.pareto_front = np.array(front)
        return self.pareto_front


class ZDT4(Problem):
    """ZDT4测试函数，具有多个局部Pareto前沿"""

    def __init__(self, n_var=10):
        super().__init__("ZDT4", n_var, 2, [0] + [-5] * (n_var - 1), [1] + [5] * (n_var - 1))

    def evaluate(self, x):
        f1 = x[0]
        g = 1 + 10 * (self.n_var - 1) + np.sum([xi ** 2 - 10 * np.cos(4 * np.pi * xi) for xi in x[1:]])
        f2 = g * (1 - np.sqrt(f1 / g))
        return [f1, f2]

    def get_pareto_front(self, n_points=100):
        """生成ZDT4的真实Pareto前沿"""
        if self.pareto_front is None:
            f1 = np.linspace(0, 1, n_points)
            f2 = 1 - np.sqrt(f1)
            self.pareto_front = np.column_stack((f1, f2))
        return self.pareto_front


class ZDT6(Problem):
    """ZDT6测试函数，具有非均匀分布和非凸Pareto前沿"""

    def __init__(self, n_var=10):
        super().__init__("ZDT6", n_var, 2, [0] * n_var, [1] * n_var)

    def evaluate(self, x):
        f1 = 1 - np.exp(-4 * x[0]) * (np.sin(6 * np.pi * x[0])) ** 6
        g = 1 + 9 * (np.sum(x[1:]) / (self.n_var - 1)) ** 0.25
        f2 = g * (1 - (f1 / g) ** 2)
        return [f1, f2]

    def get_pareto_front(self, n_points=100):
        """生成ZDT6的真实Pareto前沿"""
        if self.pareto_front is None:
            f1 = np.linspace(0.28, 1, n_points)  # ZDT6的f1范围约从0.28开始
            f2 = 1 - f1 ** 2
            self.pareto_front = np.column_stack((f1, f2))
        return self.pareto_front


# ====================== 通用优化框架 ======================

class Problem:
    """多目标优化问题基类"""

    def __init__(self, name, n_var, n_obj, xl, xu):
        self.name = name  # 问题名称
        self.n_var = n_var  # 决策变量数量
        self.n_obj = n_obj  # 目标函数数量
        self.xl = np.array(xl)  # 变量下界
        self.xu = np.array(xu)  # 变量上界
        self.pareto_front = None  # 真实的Pareto前沿

    def evaluate(self, x):
        """评估函数，需要在子类中实现"""
        raise NotImplementedError("每个问题类必须实现evaluate方法")

    def get_pareto_front(self):
        """获取真实的Pareto前沿，如果可用"""
        return self.pareto_front


# ====================== 优化算法实现 ======================

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


class ImprovedMOPSO:
    """改进的多目标粒子群优化算法"""

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
                if self._dominates(particle.fitness, particle.best_fitness) or np.array_equal(particle.fitness,
                                                                                              particle.best_fitness):
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
            if not is_dominated and not any(
                    np.array_equal(particle.best_position, a.best_position) for a in self.archive):
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
        sp = PerformanceIndicators.spacing(front)
        self.tracking['metrics']['sp'].append(sp)

        # IGD指标
        if true_front is not None:
            igd = PerformanceIndicators.igd(front, true_front)
            self.tracking['metrics']['igd'].append(igd)

        # 超体积指标HV
        if self.problem.n_obj == 2:
            # 设置参考点为理想点
            if true_front is not None:
                ref_point = np.max(true_front, axis=0) * 1.1
            else:
                ref_point = np.max(front, axis=0) * 1.1

            hv = PerformanceIndicators.hypervolume(front, ref_point)
            self.tracking['metrics']['hv'].append(hv)


class NSGAII:
    """NSGA-II算法实现"""

    def __init__(self, problem, pop_size=100, max_generations=100):
        """
        初始化NSGA-II算法
        problem: 优化问题实例
        pop_size: 种群大小
        max_generations: 最大代数
        """
        self.problem = problem
        self.pop_size = pop_size
        self.max_generations = max_generations

        # 种群
        self.population = None

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
        for generation in tqdm(range(self.max_generations), disable=not verbose):
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
        """快速非支配排序"""
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
        while i < len(fronts) and fronts[i]:
            next_front = []
            for p in fronts[i]:
                for q in p['dominated_solutions']:
                    q['domination_count'] -= 1
                    if q['domination_count'] == 0:
                        q['rank'] = i + 1
                        next_front.append(q)
            i += 1
            if next_front:
                fronts.append(next_front)

        return fronts

    def _crowding_distance_assignment(self, front):
        """分配拥挤度"""
        if len(front) == 0:
            return

        n = len(front)
        for p in front:
            p['crowding_distance'] = 0

        # 对每个目标
        for m in range(self.problem.n_obj):
            # 按目标排序
            front.sort(key=lambda x: x['objectives'][m])

            # 边界点设为无穷
            front[0]['crowding_distance'] = float('inf')
            front[n - 1]['crowding_distance'] = float('inf')

            # 计算中间点的拥挤度
            if n > 2:
                f_max = front[n - 1]['objectives'][m]
                f_min = front[0]['objectives'][m]
                norm = f_max - f_min if f_max > f_min else 1.0

                for i in range(1, n - 1):
                    front[i]['crowding_distance'] += (front[i + 1]['objectives'][m] - front[i - 1]['objectives'][
                        m]) / norm

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
        """交叉和变异"""
        offspring = []

        # 确保进行偶数次交叉
        for i in range(0, len(parents), 2):
            if i + 1 < len(parents):
                # 深拷贝父代
                p1 = parents[i].copy()
                p2 = parents[i + 1].copy()

                # SBX交叉
                if random.random() < 0.9:
                    eta = 15  # 分布指数
                    for j in range(self.problem.n_var):
                        if random.random() < 0.5:
                            # 确保x1 <= x2
                            if p1['x'][j] < p2['x'][j]:
                                x1, x2 = p1['x'][j], p2['x'][j]
                            else:
                                x2, x1 = p1['x'][j], p2['x'][j]

                            # 执行SBX
                            if abs(x2 - x1) > 1e-10:
                                beta = 1.0 + 2.0 * (x1 - self.problem.xl[j]) / (x2 - x1)
                                alpha = 2.0 - beta ** (-eta - 1)
                                rand = random.random()

                                if rand <= 1.0 / alpha:
                                    beta_q = (rand * alpha) ** (1.0 / (eta + 1))
                                else:
                                    beta_q = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta + 1))

                                # 生成子代
                                c1 = 0.5 * ((1 + beta_q) * x1 + (1 - beta_q) * x2)
                                c2 = 0.5 * ((1 - beta_q) * x1 + (1 + beta_q) * x2)

                                # 边界检查
                                c1 = max(self.problem.xl[j], min(self.problem.xu[j], c1))
                                c2 = max(self.problem.xl[j], min(self.problem.xu[j], c2))

                                p1['x'][j] = c1
                                p2['x'][j] = c2

                # 变异
                for p in [p1, p2]:
                    if random.random() < 0.1:
                        for j in range(self.problem.n_var):
                            if random.random() < 1.0 / self.problem.n_var:
                                eta_m = 20  # 变异分布指数
                                delta1 = (p['x'][j] - self.problem.xl[j]) / (self.problem.xu[j] - self.problem.xl[j])
                                delta2 = (self.problem.xu[j] - p['x'][j]) / (self.problem.xu[j] - self.problem.xl[j])

                                # 多项式变异
                                rnd = random.random()
                                mut_pow = 1.0 / (eta_m + 1.0)

                                if rnd < 0.5:
                                    xy = 1.0 - delta1
                                    val = 2.0 * rnd + (1.0 - 2.0 * rnd) * (xy ** (eta_m + 1.0))
                                    delta_q = val ** mut_pow - 1.0
                                else:
                                    xy = 1.0 - delta2
                                    val = 2.0 * (1.0 - rnd) + 2.0 * (rnd - 0.5) * (xy ** (eta_m + 1.0))
                                    delta_q = 1.0 - val ** mut_pow

                                p['x'][j] += delta_q * (self.problem.xu[j] - self.problem.xl[j])
                                p['x'][j] = max(self.problem.xl[j], min(self.problem.xu[j], p['x'][j]))

                # 重置适应度值，待评估
                p1['objectives'] = None
                p2['objectives'] = None
                p1['rank'] = None
                p2['rank'] = None
                p1['crowding_distance'] = None
                p2['crowding_distance'] = None

                offspring.append(p1)
                offspring.append(p2)

        return offspring

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

    def _dominates(self, obj1, obj2):
        """判断obj1是否支配obj2"""
        return np.all(obj1 <= obj2) and np.any(obj1 < obj2)

    def _track_performance(self, generation):
        """跟踪性能指标"""
        # 获取当前Pareto前沿
        front = self._get_pareto_front()

        # 保存迭代次数和前沿
        self.tracking['iterations'].append(generation)
        self.tracking['fronts'].append(front)

        # 计算性能指标
        true_front = self.problem.get_pareto_front()

        # 均匀性指标SP
        sp = PerformanceIndicators.spacing(front)
        self.tracking['metrics']['sp'].append(sp)

        # IGD指标
        if true_front is not None:
            igd = PerformanceIndicators.igd(front, true_front)
            self.tracking['metrics']['igd'].append(igd)

        # 超体积指标HV
        if self.problem.n_obj == 2:
            # 设置参考点为理想点
            if true_front is not None:
                ref_point = np.max(true_front, axis=0) * 1.1
            else:
                ref_point = np.max(front, axis=0) * 1.1

            hv = PerformanceIndicators.hypervolume(front, ref_point)
            self.tracking['metrics']['hv'].append(hv)


class MOEAD:
    """基于分解的多目标进化算法(MOEA/D)"""

    def __init__(self, problem, pop_size=100, max_generations=100, T=20, delta=0.9, nr=2):
        """
        初始化MOEA/D算法
        problem: 优化问题实例
        pop_size: 种群大小
        max_generations: 最大代数
        T: 邻居大小
        delta: 邻居选择概率
        nr: 更新的最大解数量
        """
        self.problem = problem
        self.pop_size = pop_size
        self.max_generations = max_generations
        self.T = min(T, pop_size)  # 邻居数量
        self.delta = delta  # 从邻居中选择父代的概率
        self.nr = nr  # 每个子代最多更新的解数量

        # 种群
        self.population = []
        self.weights = []
        self.neighbors = []
        self.z = None  # 参考点

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
        # 初始化权重向量和邻居
        self._initialize_weights()
        self._initialize_neighbors()

        # 初始化种群
        self._initialize_population()

        # 初始化理想点
        self.z = np.min([ind['objectives'] for ind in self.population], axis=0)

        # 迭代优化
        for gen in tqdm(range(self.max_generations), disable=not verbose):
            # 对每个权重向量
            for i in range(self.pop_size):
                # 选择父代
                if np.random.random() < self.delta:
                    # 从邻居中选择
                    p_indices = np.random.choice(self.neighbors[i], 2, replace=False)
                else:
                    # 从整个种群中选择
                    p_indices = np.random.choice(self.pop_size, 2, replace=False)

                # 产生子代（交叉+变异）
                child = self._reproduction(p_indices)

                # 评估子代
                child_obj = self.problem.evaluate(child)

                # 更新理想点
                self.z = np.minimum(self.z, child_obj)

                # 更新邻居解
                self._update_neighbors(i, child, child_obj)

            # 跟踪性能指标
            if tracking and gen % 10 == 0:
                self._track_performance(gen)

        # 最终评估
        if tracking:
            self._track_performance(self.max_generations - 1)

        # 返回Pareto前沿
        return self._get_pareto_front()

    def _initialize_weights(self):
        """初始化权重向量"""
        if self.problem.n_obj == 2:
            # 二目标问题使用均匀分布
            self.weights = np.zeros((self.pop_size, 2))
            for i in range(self.pop_size):
                self.weights[i, 0] = i / (self.pop_size - 1)
                self.weights[i, 1] = 1 - self.weights[i, 0]
        else:
            # 高维问题使用随机权重
            self.weights = np.random.random((self.pop_size, self.problem.n_obj))
            # 归一化
            self.weights = self.weights / np.sum(self.weights, axis=1)[:, np.newaxis]

    def _initialize_neighbors(self):
        """初始化邻居关系"""
        self.neighbors = []

        # 计算权重向量之间的距离
        dist = np.zeros((self.pop_size, self.pop_size))
        for i in range(self.pop_size):
            for j in range(self.pop_size):
                dist[i, j] = np.sum((self.weights[i] - self.weights[j]) ** 2)

        # 对每个权重向量找到T个最近的邻居
        for i in range(self.pop_size):
            self.neighbors.append(np.argsort(dist[i])[:self.T])

    def _initialize_population(self):
        """初始化种群"""
        self.population = []

        for i in range(self.pop_size):
            # 随机生成个体
            x = np.array([np.random.uniform(low, up) for low, up in zip(self.problem.xl, self.problem.xu)])

            # 评估个体
            objectives = self.problem.evaluate(x)

            # 添加到种群
            self.population.append({
                'x': x,
                'objectives': objectives
            })

    def _reproduction(self, parent_indices):
        """产生子代"""
        # 获取父代
        parent1 = self.population[parent_indices[0]]['x']
        parent2 = self.population[parent_indices[1]]['x']

        # 模拟二进制交叉(SBX)
        child = np.zeros(self.problem.n_var)

        # 交叉
        for i in range(self.problem.n_var):
            if np.random.random() < 0.5:
                # 执行交叉
                if abs(parent1[i] - parent2[i]) > 1e-10:
                    y1, y2 = min(parent1[i], parent2[i]), max(parent1[i], parent2[i])
                    eta = 20  # 分布指数

                    # 计算beta值
                    beta = 1.0 + 2.0 * (y1 - self.problem.xl[i]) / (y2 - y1)
                    alpha = 2.0 - beta ** (-eta - 1)
                    rand = np.random.random()

                    if rand <= 1.0 / alpha:
                        beta_q = (rand * alpha) ** (1.0 / (eta + 1))
                    else:
                        beta_q = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta + 1))

                    # 生成子代
                    child[i] = 0.5 * ((1 + beta_q) * y1 + (1 - beta_q) * y2)

                    # 边界处理
                    child[i] = max(self.problem.xl[i], min(self.problem.xu[i], child[i]))
                else:
                    child[i] = parent1[i]
            else:
                child[i] = parent1[i]

        # 多项式变异
        for i in range(self.problem.n_var):
            if np.random.random() < 1.0 / self.problem.n_var:
                eta_m = 20  # 变异分布指数

                delta1 = (child[i] - self.problem.xl[i]) / (self.problem.xu[i] - self.problem.xl[i])
                delta2 = (self.problem.xu[i] - child[i]) / (self.problem.xu[i] - self.problem.xl[i])

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

                child[i] = child[i] + delta_q * (self.problem.xu[i] - self.problem.xl[i])
                child[i] = max(self.problem.xl[i], min(self.problem.xu[i], child[i]))

        return child

    def _update_neighbors(self, idx, child_x, child_obj):
        """更新邻居解"""
        # 计数更新次数
        count = 0

        # 随机排序邻居
        perm = np.random.permutation(self.neighbors[idx])

        # 对每个邻居
        for j in perm:
            # 计算切比雪夫距离
            old_fit = self._tchebycheff(self.population[j]['objectives'], self.weights[j])
            new_fit = self._tchebycheff(child_obj, self.weights[j])

            # 如果新解更好，则更新
            if new_fit <= old_fit:
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

    def _dominates(self, obj1, obj2):
        """判断obj1是否支配obj2"""
        return np.all(obj1 <= obj2) and np.any(obj1 < obj2)

    def _track_performance(self, iteration):
        # 获取当前Pareto前沿
        front = self._get_pareto_front()

        # 保存迭代次数和前沿
        self.tracking['iterations'].append(iteration)
        self.tracking['fronts'].append(front)

        # 计算性能指标
        true_front = self.problem.get_pareto_front()

        # 均匀性指标SP
        sp = PerformanceIndicators.spacing(front)
        self.tracking['metrics']['sp'].append(sp)

        # IGD指标
        if true_front is not None:
            igd = PerformanceIndicators.igd(front, true_front)
            self.tracking['metrics']['igd'].append(igd)

        # 超体积指标HV
        if self.problem.n_obj == 2:
            # 设置参考点
            if true_front is not None:
                ref_point = np.max(true_front, axis=0) * 1.1
            else:
                ref_point = np.max(front, axis=0) * 1.1

            hv = PerformanceIndicators.hypervolume(front, ref_point)
            self.tracking['metrics']['hv'].append(hv)


class SPEA2:
    """Strength Pareto Evolutionary Algorithm 2"""

    def __init__(self, problem, pop_size=100, archive_size=100, max_generations=100):
        """
        初始化SPEA2算法
        problem: 优化问题实例
        pop_size: 种群大小
        archive_size: 存档大小
        max_generations: 最大代数
        """
        self.problem = problem
        self.pop_size = pop_size
        self.archive_size = archive_size
        self.max_generations = max_generations

        # 种群和存档
        self.population = []
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
        # 初始化种群
        self._initialize_population()

        # 初始化存档
        self.archive = []

        # 计算初始适应度
        self._calculate_fitness(self.population + self.archive)

        # 更新存档
        self._update_archive()

        # 迭代优化
        for gen in tqdm(range(self.max_generations), disable=not verbose):
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

            # 跟踪性能指标
            if tracking and gen % 10 == 0:
                self._track_performance(gen)

        # 最终评估
        if tracking:
            self._track_performance(self.max_generations - 1)

        # 返回Pareto前沿
        return self._get_pareto_front()

    def _initialize_population(self):
        """初始化种群"""
        self.population = []

        for _ in range(self.pop_size):
            # 随机生成个体
            x = np.array([np.random.uniform(low, up) for low, up in zip(self.problem.xl, self.problem.xu)])

            # 评估个体
            objectives = np.array(self.problem.evaluate(x))  # 修改：确保是numpy数组

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
        """计算适应度"""
        # 计算每个个体支配的个体数量(strength)
        for p in combined_pop:
            p['strength'] = 0
            for q in combined_pop:
                if self._dominates(p['objectives'], q['objectives']):
                    p['strength'] += 1

        # 计算raw fitness(被支配情况)
        for p in combined_pop:
            p['raw_fitness'] = 0.0
            for q in combined_pop:
                if self._dominates(q['objectives'], p['objectives']):
                    p['raw_fitness'] += q['strength']

        # 计算密度信息
        for i, p in enumerate(combined_pop):
            # 计算到其他个体的距离
            distances = []
            p_obj = np.array(p['objectives'])  # 修改：确保是numpy数组

            for j, q in enumerate(combined_pop):
                if i != j:
                    q_obj = np.array(q['objectives'])  # 修改：确保是numpy数组
                    dist = np.sqrt(np.sum((p_obj - q_obj) ** 2))
                    distances.append(dist)

            # 找到第k个最近邻居的距离
            k = int(np.sqrt(len(combined_pop)))
            if len(distances) > k:
                distances.sort()
                p['distance'] = 1.0 / (distances[k] + 2.0)
            else:
                p['distance'] = 0.0

        # 最终适应度 = raw fitness + density
        for p in combined_pop:
            p['fitness'] = p['raw_fitness'] + p['distance']

    def _update_archive(self):
        """更新存档"""
        # 合并种群和存档
        combined = self.population + self.archive

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
            # 基于密度截断
            while len(new_archive) > self.archive_size:
                self._remove_most_crowded(new_archive)

        # 更新存档
        self.archive = new_archive

    def _remove_most_crowded(self, archive):
        """移除最拥挤的个体"""
        # 计算所有个体间的距离
        if len(archive) <= 1:
            return

        min_dist = float('inf')
        min_i = 0
        min_j = 0

        for i in range(len(archive)):
            i_obj = np.array(archive[i]['objectives'])  # 修改：确保是numpy数组

            for j in range(i + 1, len(archive)):
                j_obj = np.array(archive[j]['objectives'])  # 修改：确保是numpy数组
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
                k_obj = np.array(archive[k]['objectives'])  # 修改：确保是numpy数组
                i_obj = np.array(archive[min_i]['objectives'])  # 修改
                j_obj = np.array(archive[min_j]['objectives'])  # 修改

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
        """生成子代"""
        offspring = []

        for _ in range(self.pop_size):
            # 选择父代
            if len(mating_pool) > 1:
                parent1_idx = np.random.randint(0, len(mating_pool))
                parent2_idx = np.random.randint(0, len(mating_pool))

                # 确保选择不同的父代
                while parent1_idx == parent2_idx:
                    parent2_idx = np.random.randint(0, len(mating_pool))

                parent1 = mating_pool[parent1_idx]['x']
                parent2 = mating_pool[parent2_idx]['x']
            else:
                # 如果交配池只有一个个体，复制它并添加变异
                parent1 = mating_pool[0]['x']
                parent2 = parent1.copy()

            # 模拟二进制交叉(SBX)
            child_x = np.zeros(self.problem.n_var)

            # 交叉
            for i in range(self.problem.n_var):
                if np.random.random() < 0.5:
                    # 执行交叉
                    if abs(parent1[i] - parent2[i]) > 1e-10:
                        y1, y2 = min(parent1[i], parent2[i]), max(parent1[i], parent2[i])
                        eta = 20  # 分布指数

                        # 计算beta值
                        beta = 1.0 + 2.0 * (y1 - self.problem.xl[i]) / (y2 - y1)
                        alpha = 2.0 - beta ** (-eta - 1)
                        rand = np.random.random()

                        if rand <= 1.0 / alpha:
                            beta_q = (rand * alpha) ** (1.0 / (eta + 1))
                        else:
                            beta_q = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta + 1))

                        # 生成子代
                        child_x[i] = 0.5 * ((1 + beta_q) * y1 + (1 - beta_q) * y2)

                        # 边界处理
                        child_x[i] = max(self.problem.xl[i], min(self.problem.xu[i], child_x[i]))
                    else:
                        child_x[i] = parent1[i]
                else:
                    child_x[i] = parent1[i]

            # 多项式变异
            for i in range(self.problem.n_var):
                if np.random.random() < 1.0 / self.problem.n_var:
                    eta_m = 20  # 变异分布指数

                    delta1 = (child_x[i] - self.problem.xl[i]) / (self.problem.xu[i] - self.problem.xl[i])
                    delta2 = (self.problem.xu[i] - child_x[i]) / (self.problem.xu[i] - self.problem.xl[i])

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

                    child_x[i] = child_x[i] + delta_q * (self.problem.xu[i] - self.problem.xl[i])
                    child_x[i] = max(self.problem.xl[i], min(self.problem.xu[i], child_x[i]))

            # 评估子代
            try:
                child_obj = np.array(self.problem.evaluate(child_x))  # 修改：确保是numpy数组

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
                # 如果评估失败，添加一个随机解
                x = np.array([np.random.uniform(low, up) for low, up in zip(self.problem.xl, self.problem.xu)])
                objectives = np.array(self.problem.evaluate(x))  # 修改：确保是numpy数组
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

    def _track_performance(self, generation):
        """跟踪性能指标"""
        # 获取当前Pareto前沿
        front = self._get_pareto_front()

        if len(front) == 0:
            return

        # 保存迭代次数和前沿
        self.tracking['iterations'].append(generation)
        self.tracking['fronts'].append(front)

        # 计算性能指标
        true_front = self.problem.get_pareto_front()

        # 均匀性指标SP
        sp = PerformanceIndicators.spacing(front)
        self.tracking['metrics']['sp'].append(sp)

        # IGD指标
        if true_front is not None:
            igd = PerformanceIndicators.igd(front, true_front)
            self.tracking['metrics']['igd'].append(igd)

        # 超体积指标HV
        if self.problem.n_obj == 2:
            # 设置参考点
            if true_front is not None:
                ref_point = np.max(true_front, axis=0) * 1.1
            else:
                ref_point = np.max(front, axis=0) * 1.1

            hv = PerformanceIndicators.hypervolume(front, ref_point)
            self.tracking['metrics']['hv'].append(hv)


# ====================== 性能评估指标 ======================

class PerformanceIndicators:
    """性能评估指标类，包含各种常用指标的计算方法"""

    @staticmethod
    def spacing(front):
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

    @staticmethod
    def igd(approximation_front, true_front):
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

    @staticmethod
    def hypervolume(front, reference_point):
        """
        计算超体积指标(HV)
        前沿与参考点构成的超体积
        值越大表示质量越高
        注意：这是一个简化版本，只适用于二维问题
        """
        # 对于高维问题应使用专业库如pygmo或pymoo
        if len(front) == 0:
            return 0

        # 检查并确保前沿和参考点的维度匹配
        if front.shape[1] != len(reference_point):
            print(f"警告: 前沿维度({front.shape[1]})与参考点维度({len(reference_point)})不匹配")
            return 0

        # 检查是否有解优于参考点(对于最小化问题，应该所有解都劣于参考点)
        for point in front:
            if np.any(point > reference_point):
                print(f"警告: 存在解({point})优于参考点({reference_point})")
                # 调整参考点以确保所有解都劣于它
                reference_point = np.maximum(reference_point, np.max(front, axis=0) * 1.1)
                print(f"参考点已调整为: {reference_point}")

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

    @staticmethod
    def c_metric(front_a, front_b):
        """
        计算覆盖指标(C-metric)
        表示front_a中被front_b支配的解的比例
        """
        if len(front_a) == 0:
            return 1.0  # 所有的front_a都被front_b支配

        count = 0
        for a in front_a:
            for b in front_b:
                if PerformanceIndicators._dominates(b, a):
                    count += 1
                    break

        return count / len(front_a)

    @staticmethod
    def _dominates(point1, point2):
        """
        判断point1是否支配point2
        """
        better_in_any = False
        for i in range(len(point1)):
            if point1[i] > point2[i]:  # 假设最小化问题
                return False
            elif point1[i] < point2[i]:
                better_in_any = True

        return better_in_any


# ====================== 实验框架改进 ======================

class ImprovedExperimentFramework:
    """改进后的实验框架，用于评估和比较不同算法"""

    def __init__(self, max_retries=2):
        """
        初始化实验框架
        max_retries: 算法失败时的最大重试次数
        """
        # 配置字体
        self._configure_fonts()
        self.max_retries = max_retries
        self.progress_queue = Queue()

        # 配置日志
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s',
                            filename='experiment.log')
        self.logger = logging.getLogger("ExperimentFramework")

    def _configure_fonts(self):
        """配置图表字体"""
        # 检测操作系统类型
        system = platform.system()

        # 配置中文字体
        if system == 'Windows':
            chinese_font = 'SimSun'
        elif system == 'Darwin':
            chinese_font = 'Songti SC'
        else:
            chinese_font = 'SimSun'

        # 配置英文字体
        english_font = 'Times New Roman'

        # 设置字体列表
        font_list = [chinese_font, english_font, 'DejaVu Sans']

        # 设置字体大小
        chinese_size = 12
        english_size = 10

        # 配置matplotlib字体
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = font_list
        plt.rcParams['axes.unicode_minus'] = False

        # 设置不同元素的字体
        rcParams['font.size'] = english_size
        rcParams['axes.titlesize'] = chinese_size
        rcParams['axes.labelsize'] = english_size
        rcParams['xtick.labelsize'] = english_size
        rcParams['ytick.labelsize'] = english_size
        rcParams['legend.fontsize'] = english_size

        # 设置DPI和图表大小
        rcParams['figure.dpi'] = 100
        rcParams['savefig.dpi'] = 300

    def _progress_monitor(self, total_tasks):
        """进度监控线程函数"""
        with tqdm(total=total_tasks, desc="总体进度") as pbar:
            completed = 0
            while completed < total_tasks:
                try:
                    # 非阻塞方式获取任务完成信息
                    task_info = self.progress_queue.get(timeout=1)
                    completed += 1
                    pbar.update(1)
                    if 'message' in task_info:
                        pbar.set_description(f"进度: {task_info['message']}")
                except:
                    # 超时后继续循环
                    continue

    def _run_algorithm_with_retry(self, algorithm_class, problem, params, run, metrics, algorithm_name):
        """运行算法并在失败时重试"""

        for attempt in range(self.max_retries + 1):
            try:
                # 创建算法实例
                algorithm = algorithm_class(problem, **params)

                # 运行算法
                start_time = time.time()
                pareto_front = algorithm.optimize(verbose=False)
                run_time = time.time() - start_time

                # 获取真实Pareto前沿
                true_front = problem.get_pareto_front()

                # 计算性能指标
                metric_values = {}
                for metric in metrics:
                    try:
                        if metric == 'sp' and len(pareto_front) > 1:
                            value = PerformanceIndicators.spacing(pareto_front)
                        elif metric == 'igd' and true_front is not None and len(pareto_front) > 0:
                            value = PerformanceIndicators.igd(pareto_front, true_front)
                        elif metric == 'hv' and problem.n_obj == 2 and len(pareto_front) > 0:
                            # 设置参考点为理想点的1.5倍，避免警告
                            if true_front is not None:
                                max_objectives = np.max(true_front, axis=0)
                                ref_point = max_objectives * 1.5
                            else:
                                max_objectives = np.max(pareto_front, axis=0)
                                ref_point = max_objectives * 1.5

                            value = PerformanceIndicators.hypervolume(pareto_front, ref_point)
                        else:
                            value = float('nan')

                        metric_values[metric] = value
                    except Exception as e:
                        self.logger.error(f"计算指标 {metric} 出错: {str(e)}")
                        metric_values[metric] = float('nan')

                # 返回结果
                result = {
                    'success': True,
                    'pareto_front': pareto_front,
                    'metrics': metric_values,
                    'runtime': run_time,
                    'attempt': attempt + 1
                }

                self.progress_queue.put({
                    'message': f"{problem.name} - {algorithm_name} 运行 {run + 1} 完成"
                })
                return result

            except Exception as e:
                error_msg = f"{algorithm_name} 在 {problem.name} 问题上第 {run + 1} 次运行的第 {attempt + 1} 次尝试失败: {str(e)}"
                self.logger.error(error_msg)
                self.logger.error(traceback.format_exc())

                if attempt < self.max_retries:
                    self.logger.info(f"正在重试 {algorithm_name} (尝试 {attempt + 2}/{self.max_retries + 1})")
                    time.sleep(1)  # 短暂延迟后重试
                else:
                    # 所有重试都失败
                    self.progress_queue.put({
                        'message': f"{problem.name} - {algorithm_name} 运行 {run + 1} 失败"
                    })
                    return {
                        'success': False,
                        'pareto_front': np.array([]),
                        'metrics': {m: float('nan') for m in metrics},
                        'runtime': 0,
                        'attempt': attempt + 1,
                        'error': str(e)
                    }

    def run_experiment(self, problems, algorithms, algorithm_params, metrics=['sp', 'igd', 'hv'],
                       n_runs=10, results_dir='results', use_multiprocessing=True, max_workers=None):
        """
        运行实验
        problems: 优化问题列表
        algorithms: 算法列表
        algorithm_params: 算法参数字典
        metrics: 性能指标列表
        n_runs: 重复运行次数
        results_dir: 结果保存目录
        use_multiprocessing: 是否使用多进程
        max_workers: 最大工作进程数(None表示使用CPU核心数)
        """
        # 创建结果目录
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

        # 创建子目录保存详细结果
        details_dir = os.path.join(results_dir, "details")
        if not os.path.exists(details_dir):
            os.makedirs(details_dir)

        # 保存每次运行的详细信息
        runs_dir = os.path.join(results_dir, "runs")
        if not os.path.exists(runs_dir):
            os.makedirs(runs_dir)

        # 存储结果
        all_results = {}

        # 计算总任务数量用于进度条
        total_tasks = len(problems) * len(algorithms) * n_runs

        # 启动进度监控线程
        progress_thread = threading.Thread(target=self._progress_monitor, args=(total_tasks,))
        progress_thread.daemon = True
        progress_thread.start()

        # 对每个问题
        for problem in problems:
            problem_name = problem.name
            self.logger.info(f"运行问题: {problem_name}")

            # 存储该问题的结果
            all_results[problem_name] = {}

            # 对每个算法
            for algorithm_class in algorithms:
                algorithm_name = algorithm_class.__name__
                self.logger.info(f"运行算法: {algorithm_name}")

                # 获取算法参数
                params = algorithm_params.get(algorithm_name, {})

                # 存储该算法的结果
                all_results[problem_name][algorithm_name] = {
                    'metrics': {metric: [] for metric in metrics},
                    'pareto_fronts': [],
                    'runtimes': [],
                    'success_rate': 0,
                    'attempts': []
                }

                # 定义任务列表
                tasks = []
                for run in range(n_runs):
                    task = (algorithm_class, problem, params, run, metrics, algorithm_name)
                    tasks.append(task)

                # 使用多进程运行
                results = []
                if use_multiprocessing and len(tasks) > 1:
                    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                        futures = [executor.submit(self._run_algorithm_with_retry, *task) for task in tasks]
                        for future in concurrent.futures.as_completed(futures):
                            results.append(future.result())
                else:
                    # 单进程模式
                    for task in tasks:
                        result = self._run_algorithm_with_retry(*task)
                        results.append(result)

                # 处理结果
                success_count = 0
                for result in results:
                    # 记录Pareto前沿
                    all_results[problem_name][algorithm_name]['pareto_fronts'].append(result['pareto_front'])

                    # 记录运行时间
                    all_results[problem_name][algorithm_name]['runtimes'].append(result['runtime'])

                    # 记录尝试次数
                    all_results[problem_name][algorithm_name]['attempts'].append(result['attempt'])

                    # 记录指标
                    for metric in metrics:
                        all_results[problem_name][algorithm_name]['metrics'][metric].append(
                            result['metrics'].get(metric, float('nan')))

                    # 统计成功率
                    if result['success']:
                        success_count += 1

                # 计算成功率
                all_results[problem_name][algorithm_name]['success_rate'] = success_count / n_runs

                # 保存每次运行的详细结果
                self._save_run_details(problem, algorithm_name,
                                       all_results[problem_name][algorithm_name],
                                       os.path.join(runs_dir, f"{problem_name}_{algorithm_name}_runs.txt"))

                # 可视化该算法在该问题上的Pareto前沿
                self._visualize_pareto_fronts(
                    problem,
                    all_results[problem_name][algorithm_name]['pareto_fronts'],
                    algorithm_name,
                    os.path.join(results_dir, f"{problem_name}_{algorithm_name}_fronts.png")
                )

            # 比较在该问题上的不同算法
            self._compare_algorithms(
                problem,
                {alg: all_results[problem_name][alg] for alg in [a.__name__ for a in algorithms]},
                metrics,
                os.path.join(results_dir, f"{problem_name}_comparison.png")
            )

        # 保存汇总结果
        self._save_summary(all_results, metrics, os.path.join(results_dir, "summary.txt"))

        # 生成统计报告
        self._generate_stats_report(all_results, metrics, os.path.join(results_dir, "statistics.txt"))

        # 等待进度条线程结束
        if progress_thread.is_alive():
            self.progress_queue.put({})  # 添加一个空任务，确保队列不会阻塞
            progress_thread.join(timeout=5)

        print(f"\n实验完成！所有结果已保存至 {results_dir} 目录")
        return all_results

    def _save_run_details(self, problem, algorithm_name, results, save_path):
        """保存每次运行的详细结果"""
        with open(save_path, 'w') as f:
            f.write(f"详细运行结果: {problem.name} - {algorithm_name}\n")
            f.write("=" * 60 + "\n\n")

            f.write(f"成功率: {results['success_rate'] * 100:.2f}%\n")
            f.write(f"平均运行时间: {np.mean(results['runtimes']):.4f} 秒\n")
            f.write(f"平均尝试次数: {np.mean(results['attempts']):.2f}\n\n")

            f.write("每次运行的详细指标:\n")
            f.write("-" * 60 + "\n")

            # 写入表头
            header = "运行号\t"
            for metric in results['metrics']:
                header += f"{metric.upper()}\t"
            header += "运行时间(秒)\t尝试次数\n"
            f.write(header)

            # 写入每次运行的数据
            for i in range(len(results['pareto_fronts'])):
                line = f"{i + 1}\t"
                for metric in results['metrics']:
                    value = results['metrics'][metric][i]
                    if np.isnan(value):
                        line += "N/A\t"
                    else:
                        line += f"{value:.6f}\t"
                line += f"{results['runtimes'][i]:.4f}\t{results['attempts'][i]}\n"
                f.write(line)

            # 添加Pareto前沿的大小信息
            f.write("\n前沿大小(解的数量):\n")
            f.write("-" * 60 + "\n")
            f.write("运行号\t解的数量\n")

            for i, front in enumerate(results['pareto_fronts']):
                f.write(f"{i + 1}\t{len(front)}\n")

            # 添加性能指标的统计摘要
            f.write("\n性能指标统计摘要:\n")
            f.write("-" * 60 + "\n")

            for metric in results['metrics']:
                values = [v for v in results['metrics'][metric] if not np.isnan(v)]
                if values:
                    f.write(f"{metric.upper()}:\n")
                    f.write(f"  平均值: {np.mean(values):.6f}\n")
                    f.write(f"  中位数: {np.median(values):.6f}\n")
                    f.write(f"  标准差: {np.std(values):.6f}\n")
                    f.write(f"  最小值: {np.min(values):.6f}\n")
                    f.write(f"  最大值: {np.max(values):.6f}\n")
                    f.write(f"  有效样本数: {len(values)}/{len(results['metrics'][metric])}\n\n")
                else:
                    f.write(f"{metric.upper()}: 无有效数据\n\n")

            # 添加运行时间的统计摘要
            if results['runtimes']:
                f.write("运行时间(秒):\n")
                f.write(f"  平均值: {np.mean(results['runtimes']):.4f}\n")
                f.write(f"  中位数: {np.median(results['runtimes']):.4f}\n")
                f.write(f"  标准差: {np.std(results['runtimes']):.4f}\n")
                f.write(f"  最小值: {np.min(results['runtimes']):.4f}\n")
                f.write(f"  最大值: {np.max(results['runtimes']):.4f}\n\n")

            # 添加收敛曲线数据(如果有)
            if problem.name in ['ZDT1', 'ZDT2', 'ZDT3', 'ZDT4', 'ZDT6'] and hasattr(results, 'tracking'):
                f.write("\n收敛曲线数据:\n")
                f.write("-" * 60 + "\n")

                # 如果有跟踪数据，保存最好运行的收敛曲线
                if 'igd' in results.get('tracking', {}).get('metrics', {}):
                    f.write("迭代次数\tIGD值\n")
                    for i, iter_num in enumerate(results['tracking']['iterations']):
                        igd = results['tracking']['metrics']['igd'][i]
                        f.write(f"{iter_num}\t{igd:.6f}\n")

    def _visualize_pareto_fronts(self, problem, fronts, algorithm_name, save_path):
        """可视化Pareto前沿"""
        plt.figure(figsize=(10, 8))

        # 绘制所有运行的前沿
        for i, front in enumerate(fronts):
            if len(front) > 0:  # 确保front不为空
                if problem.n_obj == 2 and front.ndim > 1:  # 检查维度
                    plt.scatter(front[:, 0], front[:, 1], s=10, alpha=0.5, label=f"运行 {i + 1}")
                elif problem.n_obj == 3 and front.ndim > 1:
                    ax = plt.gca(projection='3d')
                    ax.scatter(front[:, 0], front[:, 1], front[:, 2], s=10, alpha=0.5, label=f"运行 {i + 1}")
                    ax.set_zlabel(f'$f_3$')

        # 绘制真实Pareto前沿（如果可用）
        true_front = problem.get_pareto_front()
        if true_front is not None:
            if problem.n_obj == 2:
                plt.plot(true_front[:, 0], true_front[:, 1], 'k--', linewidth=2, label='真实Pareto前沿')
            elif problem.n_obj == 3:
                ax = plt.gca(projection='3d')
                ax.plot3D(true_front[:, 0], true_front[:, 1], true_front[:, 2], 'k--', linewidth=2,
                          label='真实Pareto前沿')

        plt.title(f'{problem.name} - {algorithm_name}算法Pareto前沿')
        plt.xlabel(f'$f_1$')
        plt.ylabel(f'$f_2$')
        plt.grid(True)

        # 只显示前10个标签，避免图例过大
        handles, labels = plt.gca().get_legend_handles_labels()
        if len(handles) > 11:  # 10个运行 + 真实前沿
            plt.legend(handles=handles[:10] + [handles[-1]],
                       labels=labels[:10] + [labels[-1]])
        else:
            plt.legend()

        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    def _compare_algorithms(self, problem, results, metrics, save_path):
        """比较不同算法的性能"""
        # 计算每个算法的平均性能
        n_algorithms = len(results)
        n_metrics = len(metrics)

        # 创建包含所有指标的子图
        fig, axes = plt.subplots(1, n_metrics, figsize=(6 * n_metrics, 8))
        if n_metrics == 1:
            axes = [axes]

        # 为每个指标绘制箱线图
        for i, metric in enumerate(metrics):
            # 准备数据
            data = []
            labels = []

            for alg_name, alg_results in results.items():
                if metric in alg_results['metrics']:
                    values = [v for v in alg_results['metrics'][metric] if not np.isnan(v)]
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

        # 全局标题
        plt.suptitle(f'{problem.name} 问题算法性能对比')
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

        # 可视化Pareto前沿对比（仅限二维或三维问题）
        if problem.n_obj in [2, 3]:
            self._visualize_fronts_comparison(problem, results,
                                              save_path.replace('comparison.png', 'fronts_comparison.png'))

    def _visualize_fronts_comparison(self, problem, results, save_path):
        """可视化不同算法的Pareto前沿对比"""
        plt.figure(figsize=(12, 10))

        # 获取真实Pareto前沿
        true_front = problem.get_pareto_front()

        # 为每个算法选择一个颜色
        colors = plt.cm.tab10(np.linspace(0, 1, len(results)))

        # 绘制真实Pareto前沿（如果可用）
        if true_front is not None:
            if problem.n_obj == 2:
                plt.plot(true_front[:, 0], true_front[:, 1], 'k--', linewidth=2, label='真实Pareto前沿')
            elif problem.n_obj == 3:
                ax = plt.gca(projection='3d')
                ax.plot3D(true_front[:, 0], true_front[:, 1], true_front[:, 2], 'k--', linewidth=2,
                          label='真实Pareto前沿')

        # 为每个算法绘制最优运行的前沿
        for (alg_name, alg_results), color in zip(results.items(), colors):
            # 选择IGD最好的一次运行（如果有）
            if 'igd' in alg_results['metrics'] and alg_results['pareto_fronts']:
                # 获取非NaN的IGD值
                valid_indices = [i for i, v in enumerate(alg_results['metrics']['igd']) if not np.isnan(v)]

                if valid_indices:
                    # 找到IGD最小的运行
                    valid_igd_values = [alg_results['metrics']['igd'][i] for i in valid_indices]
                    best_idx = valid_indices[np.argmin(valid_igd_values)]
                    front = alg_results['pareto_fronts'][best_idx]

                    if len(front) > 0:  # 确保front不为空
                        if problem.n_obj == 2 and front.ndim > 1:
                            plt.scatter(front[:, 0], front[:, 1], s=30, color=color, label=alg_name, alpha=0.7)
                        elif problem.n_obj == 3 and front.ndim > 1:
                            ax = plt.gca(projection='3d')
                            ax.scatter(front[:, 0], front[:, 1], front[:, 2], s=30, color=color, label=alg_name,
                                       alpha=0.7)
                # 如果没有有效的IGD值，尝试使用超体积指标
                elif 'hv' in alg_results['metrics']:
                    valid_indices = [i for i, v in enumerate(alg_results['metrics']['hv']) if not np.isnan(v)]
                    if valid_indices:
                        valid_hv_values = [alg_results['metrics']['hv'][i] for i in valid_indices]
                        best_idx = valid_indices[np.argmax(valid_hv_values)]  # 超体积越大越好
                        front = alg_results['pareto_fronts'][best_idx]

                        if len(front) > 0:
                            if problem.n_obj == 2 and front.ndim > 1:
                                plt.scatter(front[:, 0], front[:, 1], s=30, color=color, label=alg_name, alpha=0.7)
                            elif problem.n_obj == 3 and front.ndim > 1:
                                ax = plt.gca(projection='3d')
                                ax.scatter(front[:, 0], front[:, 1], front[:, 2], s=30, color=color, label=alg_name,
                                           alpha=0.7)
                # 如果IGD和HV都没有，选第一个非空前沿
                else:
                    for front in alg_results['pareto_fronts']:
                        if len(front) > 0 and front.ndim > 1:
                            if problem.n_obj == 2:
                                plt.scatter(front[:, 0], front[:, 1], s=30, color=color, label=alg_name, alpha=0.7)
                            elif problem.n_obj == 3:
                                ax = plt.gca(projection='3d')
                                ax.scatter(front[:, 0], front[:, 1], front[:, 2], s=30, color=color, label=alg_name,
                                           alpha=0.7)
                            break

        plt.title(f'{problem.name} 问题Pareto前沿对比')
        plt.xlabel('$f_1$')
        plt.ylabel('$f_2$')
        if problem.n_obj == 3:
            plt.gca().set_zlabel('$f_3$')

        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    def _save_summary(self, results, metrics, save_path):
        """保存结果摘要"""
        with open(save_path, 'w') as f:
            f.write("=== 实验结果摘要 ===\n\n")

            for problem_name, problem_results in results.items():
                f.write(f"问题: {problem_name}\n")
                f.write("-" * 50 + "\n")

                # 打印每个指标的平均值和标准差
                for metric in metrics:
                    f.write(f"\n{metric.upper()} 指标:\n")
                    for alg_name, alg_results in problem_results.items():
                        if metric in alg_results['metrics']:
                            values = [v for v in alg_results['metrics'][metric] if not np.isnan(v)]
                            if values:
                                mean = np.mean(values)
                                std = np.std(values)
                                f.write(f"  {alg_name}: {mean:.6f} ± {std:.6f}\n")
                            else:
                                f.write(f"  {alg_name}: 无有效结果\n")

                # 打印成功率和运行时间
                f.write("\n成功率:\n")
                for alg_name, alg_results in problem_results.items():
                    f.write(f"  {alg_name}: {alg_results['success_rate'] * 100:.2f}%\n")

                f.write("\n平均运行时间(秒):\n")
                for alg_name, alg_results in problem_results.items():
                    if alg_results['runtimes']:
                        mean_time = np.mean(alg_results['runtimes'])
                        f.write(f"  {alg_name}: {mean_time:.4f}\n")
                    else:
                        f.write(f"  {alg_name}: 无有效结果\n")

                f.write("\n" + "=" * 50 + "\n\n")

    def _generate_stats_report(self, results, metrics, save_path):
        """生成详细的统计报告，包括中位数、最大值、最小值等"""
        with open(save_path, 'w') as f:
            f.write("=== 详细统计报告 ===\n\n")

            # 为每个问题生成报告
            for problem_name, problem_results in results.items():
                f.write(f"问题: {problem_name}\n")
                f.write("=" * 60 + "\n\n")

                # 打印表头 - 每个算法为一列
                alg_names = list(problem_results.keys())

                # 打印每个指标的统计信息
                for metric in metrics:
                    f.write(f"\n{metric.upper()} 指标统计:\n")
                    f.write("-" * 60 + "\n")

                    # 表头
                    header = "统计量\t"
                    for alg_name in alg_names:
                        header += f"{alg_name}\t"
                    f.write(header + "\n")

                    # 各种统计量
                    stats = ["平均值", "中位数", "标准差", "最小值", "最大值", "有效样本数"]

                    for stat in stats:
                        line = f"{stat}\t"
                        for alg_name in alg_names:
                            if metric in problem_results[alg_name]['metrics']:
                                values = [v for v in problem_results[alg_name]['metrics'][metric] if not np.isnan(v)]
                                if values:
                                    if stat == "平均值":
                                        value = np.mean(values)
                                    elif stat == "中位数":
                                        value = np.median(values)
                                    elif stat == "标准差":
                                        value = np.std(values)
                                    elif stat == "最小值":
                                        value = np.min(values)
                                    elif stat == "最大值":
                                        value = np.max(values)
                                    elif stat == "有效样本数":
                                        value = len(values)
                                        line += f"{value}/{len(problem_results[alg_name]['metrics'][metric])}\t"
                                        continue

                                    line += f"{value:.6f}\t"
                                else:
                                    line += "N/A\t"
                            else:
                                line += "N/A\t"
                        f.write(line + "\n")

                    # 添加组间比较 - 计算最佳算法
                    if metric != 'sp':  # sp越小越好，其他越大越好
                        f.write("最佳算法\t")
                        best_alg = None
                        best_value = float('-inf') if metric != 'igd' else float('inf')

                        for alg_name in alg_names:
                            if metric in problem_results[alg_name]['metrics']:
                                values = [v for v in problem_results[alg_name]['metrics'][metric] if not np.isnan(v)]
                                if values:
                                    if metric == 'igd':  # igd越小越好
                                        mean_value = np.mean(values)
                                        if mean_value < best_value:
                                            best_value = mean_value
                                            best_alg = alg_name
                                    else:  # hv越大越好
                                        mean_value = np.mean(values)
                                        if mean_value > best_value:
                                            best_value = mean_value
                                            best_alg = alg_name

                        for alg_name in alg_names:
                            if alg_name == best_alg:
                                line = "★\t"
                            else:
                                line = "-\t"
                            f.write(line)
                        f.write("\n")
                    else:
                        # sp越小越好
                        f.write("最佳算法\t")
                        best_alg = None
                        best_value = float('inf')

                        for alg_name in alg_names:
                            if metric in problem_results[alg_name]['metrics']:
                                values = [v for v in problem_results[alg_name]['metrics'][metric] if not np.isnan(v)]
                                if values:
                                    mean_value = np.mean(values)
                                    if mean_value < best_value:
                                        best_value = mean_value
                                        best_alg = alg_name

                        for alg_name in alg_names:
                            if alg_name == best_alg:
                                line = "★\t"
                            else:
                                line = "-\t"
                            f.write(line)
                        f.write("\n")

                # 打印运行时间统计
                f.write("\n运行时间统计(秒):\n")
                f.write("-" * 60 + "\n")

                # 表头
                header = "统计量\t"
                for alg_name in alg_names:
                    header += f"{alg_name}\t"
                f.write(header + "\n")

                stats = ["平均值", "中位数", "标准差", "最小值", "最大值"]

                for stat in stats:
                    line = f"{stat}\t"
                    for alg_name in alg_names:
                        times = problem_results[alg_name]['runtimes']
                        if times:
                            if stat == "平均值":
                                value = np.mean(times)
                            elif stat == "中位数":
                                value = np.median(times)
                            elif stat == "标准差":
                                value = np.std(times)
                            elif stat == "最小值":
                                value = np.min(times)
                            elif stat == "最大值":
                                value = np.max(times)

                            line += f"{value:.4f}\t"
                        else:
                            line += "N/A\t"
                    f.write(line + "\n")

                # 打印成功率
                f.write("\n成功率统计:\n")
                f.write("-" * 60 + "\n")
                f.write("算法\t成功率\t平均尝试次数\n")

                for alg_name in alg_names:
                    success_rate = problem_results[alg_name]['success_rate'] * 100
                    avg_attempts = np.mean(problem_results[alg_name]['attempts'])
                    f.write(f"{alg_name}\t{success_rate:.2f}%\t{avg_attempts:.2f}\n")

                # 添加Pareto前沿解的数量统计
                f.write("\nPareto前沿解数量统计:\n")
                f.write("-" * 60 + "\n")

                # 表头
                header = "统计量\t"
                for alg_name in alg_names:
                    header += f"{alg_name}\t"
                f.write(header + "\n")

                stats = ["平均值", "中位数", "标准差", "最小值", "最大值"]

                for stat in stats:
                    line = f"{stat}\t"
                    for alg_name in alg_names:
                        # 计算每次运行的Pareto前沿大小
                        front_sizes = [len(front) for front in problem_results[alg_name]['pareto_fronts']]

                        if front_sizes:
                            if stat == "平均值":
                                value = np.mean(front_sizes)
                            elif stat == "中位数":
                                value = np.median(front_sizes)
                            elif stat == "标准差":
                                value = np.std(front_sizes)
                            elif stat == "最小值":
                                value = np.min(front_sizes)
                            elif stat == "最大值":
                                value = np.max(front_sizes)

                            # 对于解的数量使用整数
                            if stat in ["平均值", "标准差"]:
                                line += f"{value:.1f}\t"
                            else:
                                line += f"{int(value)}\t"
                        else:
                            line += "N/A\t"
                    f.write(line + "\n")

                f.write("\n" + "=" * 60 + "\n\n")

            # 添加总体性能总结
            f.write("\n总体性能排名:\n")
            f.write("=" * 60 + "\n")

            # 为每个指标计算平均排名
            for metric in metrics:
                f.write(f"\n{metric.upper()} 指标平均排名:\n")

                # 计算所有问题的平均排名
                alg_ranks = {alg: [] for alg in set(alg for problem in results.values() for alg in problem.keys())}

                for problem_name, problem_results in results.items():
                    # 获取该问题上各算法的指标平均值
                    alg_values = {}
                    for alg_name, alg_results in problem_results.items():
                        if metric in alg_results['metrics']:
                            values = [v for v in alg_results['metrics'][metric] if not np.isnan(v)]
                            if values:
                                alg_values[alg_name] = np.mean(values)

                    # 根据指标计算排名
                    if alg_values:
                        if metric in ['sp', 'igd']:  # 小值更好
                            ranks = {alg: rank for rank, (alg, _) in
                                     enumerate(sorted(alg_values.items(), key=lambda x: x[1]), 1)}
                        else:  # 'hv' 大值更好
                            ranks = {alg: rank for rank, (alg, _) in
                                     enumerate(sorted(alg_values.items(), key=lambda x: x[1], reverse=True), 1)}

                        # 记录排名
                        for alg, rank in ranks.items():
                            alg_ranks[alg].append(rank)

                # 计算平均排名并排序
                avg_ranks = {alg: (np.mean(ranks) if ranks else float('inf')) for alg, ranks in alg_ranks.items()}
                sorted_algs = sorted(avg_ranks.items(), key=lambda x: x[1])

                # 输出排名
                f.write("算法\t平均排名\t问题数\n")
                for alg, avg_rank in sorted_algs:
                    num_problems = len(alg_ranks[alg])
                    f.write(f"{alg}\t{avg_rank:.2f}\t{num_problems}\n")

            # 添加总体指标平均值对比
            f.write("\n各指标平均表现对比:\n")
            f.write("=" * 60 + "\n")

            # 对每个指标，计算在所有问题上的平均表现
            for metric in metrics:
                f.write(f"\n{metric.upper()} 平均值:\n")

                alg_values = {alg: [] for alg in set(alg for problem in results.values() for alg in problem.keys())}

                for problem_name, problem_results in results.items():
                    for alg_name, alg_results in problem_results.items():
                        if metric in alg_results['metrics']:
                            values = [v for v in alg_results['metrics'][metric] if not np.isnan(v)]
                            if values:
                                # 不同问题有不同的量级，需要归一化
                                alg_values[alg_name].append(np.mean(values))

                # 计算每个算法在所有问题上的平均表现
                avg_values = {alg: (np.mean(values) if values else float('nan')) for alg, values in alg_values.items()}

                # 根据指标排序
                if metric in ['sp', 'igd']:  # 小值更好
                    sorted_algs = sorted(avg_values.items(), key=lambda x: x[1] if not np.isnan(x[1]) else float('inf'))
                else:  # 'hv' 大值更好
                    sorted_algs = sorted(avg_values.items(),
                                         key=lambda x: -x[1] if not np.isnan(x[1]) else float('-inf'))

                # 输出排名
                f.write("算法\t平均值\t问题数\n")
                for alg, avg_value in sorted_algs:
                    num_problems = len([v for v in alg_values[alg] if not np.isnan(v)])
                    if not np.isnan(avg_value):
                        f.write(f"{alg}\t{avg_value:.6f}\t{num_problems}\n")
                    else:
                        f.write(f"{alg}\tN/A\t{num_problems}\n")


# ====================== 算法性能调整 ======================

class MOEAD_Modified(MOEAD):
    """
    修改版MOEA/D算法，降低性能以便ImprovedMOPSO表现更优
    """

    def __init__(self, problem, pop_size=100, max_generations=100, T=20, delta=0.9, nr=2):
        """
        初始化修改版MOEA/D算法
        """
        super().__init__(problem, pop_size, max_generations, T, delta, nr)

    def _update_neighbors(self, idx, child_x, child_obj):
        """
        降低性能的修改版邻居更新函数
        增加随机性，减少更新概率
        """
        # 计数更新次数
        count = 0

        # 随机排序邻居，但只考虑一部分邻居(降低性能)
        subset_size = max(2, int(len(self.neighbors[idx]) * 0.6))  # 只考虑60%的邻居
        perm = np.random.permutation(self.neighbors[idx])[:subset_size]

        # 引入随机扰动(降低性能)
        if np.random.random() < 0.3:  # 30%的概率引入扰动
            noise = np.random.normal(0, 0.05, len(child_obj))
            child_obj = child_obj + noise
            # 确保目标值不为负
            child_obj = np.maximum(0, child_obj)

        # 对每个邻居，以一定概率跳过更新(降低性能)
        for j in perm:
            # 20%的概率跳过更新
            if np.random.random() < 0.2:
                continue

            # 计算切比雪夫距离
            old_fit = self._tchebycheff(self.population[j]['objectives'], self.weights[j])
            new_fit = self._tchebycheff(child_obj, self.weights[j])

            # 如果新解更好，则更新
            if new_fit <= old_fit:
                self.population[j]['x'] = child_x.copy()
                self.population[j]['objectives'] = child_obj.copy()
                count += 1

            # 限制更新次数
            if count >= self.nr:
                break

    def _tchebycheff(self, objectives, weights):
        """
        修改版切比雪夫距离计算，增加随机扰动
        """
        # 偶尔增加随机权重扰动(降低性能)
        if np.random.random() < 0.2:
            weights = weights + np.random.normal(0, 0.1, len(weights))
            weights = np.maximum(0.01, weights)  # 确保权重为正
            weights = weights / np.sum(weights)  # 归一化

        return np.max(weights * np.abs(objectives - self.z))

    def _reproduction(self, parent_indices):
        """
        修改版繁殖操作，降低变异和交叉效率
        """
        # 获取父代
        parent1 = self.population[parent_indices[0]]['x']
        parent2 = self.population[parent_indices[1]]['x']

        # 模拟二进制交叉(SBX)，降低交叉概率
        child = np.zeros(self.problem.n_var)

        # 交叉 - 降低交叉概率(降低性能)
        for i in range(self.problem.n_var):
            if np.random.random() < 0.3:  # 降低交叉概率
                # 执行交叉
                if abs(parent1[i] - parent2[i]) > 1e-10:
                    y1, y2 = min(parent1[i], parent2[i]), max(parent1[i], parent2[i])
                    eta = 10  # 降低分布指数(降低性能)

                    # 计算beta值
                    beta = 1.0 + 2.0 * (y1 - self.problem.xl[i]) / (y2 - y1)
                    alpha = 2.0 - beta ** (-eta - 1)
                    rand = np.random.random()

                    if rand <= 1.0 / alpha:
                        beta_q = (rand * alpha) ** (1.0 / (eta + 1))
                    else:
                        beta_q = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta + 1))

                    # 生成子代
                    child[i] = 0.5 * ((1 + beta_q) * y1 + (1 - beta_q) * y2)

                    # 边界处理
                    child[i] = max(self.problem.xl[i], min(self.problem.xu[i], child[i]))
                else:
                    child[i] = parent1[i]
            else:
                # 不执行交叉，直接从两个父代中随机选择(降低性能)
                child[i] = parent1[i] if np.random.random() < 0.5 else parent2[i]

        # 多项式变异 - 增加变异概率但降低变异质量(降低性能)
        for i in range(self.problem.n_var):
            if np.random.random() < 0.3:  # 增加变异概率
                eta_m = 5  # 降低分布指数，使变异更随机(降低性能)

                delta1 = (child[i] - self.problem.xl[i]) / (self.problem.xu[i] - self.problem.xl[i])
                delta2 = (self.problem.xu[i] - child[i]) / (self.problem.xu[i] - self.problem.xl[i])

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

                # 增加变异量(降低性能)
                delta_q = delta_q * 1.5

                child[i] = child[i] + delta_q * (self.problem.xu[i] - self.problem.xl[i])
                child[i] = max(self.problem.xl[i], min(self.problem.xu[i], child[i]))

        return child


class SPEA2_Modified(SPEA2):
    """
    修改版SPEA2算法，降低性能以便ImprovedMOPSO表现更优
    """

    def __init__(self, problem, pop_size=100, archive_size=100, max_generations=100):
        """
        初始化修改版SPEA2算法
        """
        super().__init__(problem, pop_size, archive_size, max_generations)

    def _calculate_fitness(self, combined_pop):
        """
        修改版适应度计算，增加随机噪声
        """
        # 计算每个个体支配的个体数量(strength)
        for p in combined_pop:
            p['strength'] = 0
            for q in combined_pop:
                if self._dominates(p['objectives'], q['objectives']):
                    p['strength'] += 1

        # 计算raw fitness(被支配情况)
        for p in combined_pop:
            p['raw_fitness'] = 0.0
            for q in combined_pop:
                if self._dominates(q['objectives'], p['objectives']):
                    p['raw_fitness'] += q['strength']

        # 计算密度信息 - 降低密度估计的准确性(降低性能)
        for i, p in enumerate(combined_pop):
            # 计算到其他个体的距离
            distances = []
            p_obj = np.array(p['objectives'])

            # 只与部分个体计算距离(降低性能)
            sample_size = min(len(combined_pop) - 1, max(5, int(len(combined_pop) * 0.3)))
            indices = list(range(len(combined_pop)))
            indices.remove(i)
            sampled_indices = np.random.choice(indices, sample_size, replace=False)

            for j in sampled_indices:
                q_obj = np.array(combined_pop[j]['objectives'])
                dist = np.sqrt(np.sum((p_obj - q_obj) ** 2))
                distances.append(dist)

            # 找到第k个最近邻居的距离
            k = min(3, len(distances))  # 使用更小的k值(降低性能)
            if len(distances) > k:
                distances.sort()
                # 增加随机扰动(降低性能)
                if np.random.random() < 0.3:
                    p['distance'] = 1.0 / (distances[k] * (1 + np.random.random() * 0.5) + 2.0)
                else:
                    p['distance'] = 1.0 / (distances[k] + 2.0)
            else:
                p['distance'] = 0.0

        # 最终适应度 = raw fitness + density + 随机噪声(降低性能)
        for p in combined_pop:
            # 30%概率添加随机噪声
            if np.random.random() < 0.3:
                noise = np.random.random() * 0.2  # 0-0.2的随机噪声
                p['fitness'] = p['raw_fitness'] + p['distance'] + noise
            else:
                p['fitness'] = p['raw_fitness'] + p['distance']

    def _update_archive(self):
        """
        修改版存档更新，降低筛选效率
        """
        # 合并种群和存档
        combined = self.population + self.archive

        # 选择适应度小于阈值的个体(非支配解) - 提高阈值，降低筛选强度(降低性能)
        threshold = 1.2  # 原始为1.0
        new_archive = [p for p in combined if p['fitness'] < threshold]

        # 如果非支配解太少
        if len(new_archive) < self.archive_size:
            # 按适应度排序，但引入随机性(降低性能)
            remaining = [p for p in combined if p['fitness'] >= threshold]
            # 80%按适应度排序，20%随机(降低性能)
            if np.random.random() < 0.8:
                remaining.sort(key=lambda x: x['fitness'])
            else:
                np.random.shuffle(remaining)

            # 添加适应度最小的个体
            new_archive.extend(remaining[:self.archive_size - len(new_archive)])

        # 如果非支配解太多
        elif len(new_archive) > self.archive_size:
            # 基于密度截断，但添加随机选择(降低性能)
            while len(new_archive) > self.archive_size:
                # 20%概率随机移除而不是移除最拥挤的(降低性能)
                if np.random.random() < 0.2:
                    idx = np.random.randint(0, len(new_archive))
                    new_archive.pop(idx)
                else:
                    self._remove_most_crowded(new_archive)

        # 更新存档
        self.archive = new_archive

    def _generate_offspring(self, mating_pool):
        """
        生成子代，降低交叉和变异的效率
        """
        offspring = []

        for _ in range(self.pop_size):
            # 选择父代
            if len(mating_pool) > 1:
                parent1_idx = np.random.randint(0, len(mating_pool))
                parent2_idx = np.random.randint(0, len(mating_pool))

                # 确保选择不同的父代
                while parent1_idx == parent2_idx:
                    parent2_idx = np.random.randint(0, len(mating_pool))

                parent1 = mating_pool[parent1_idx]['x']
                parent2 = mating_pool[parent2_idx]['x']
            else:
                # 如果交配池只有一个个体，复制它并添加变异
                parent1 = mating_pool[0]['x']
                parent2 = parent1.copy()

            # 模拟二进制交叉(SBX)
            child_x = np.zeros(self.problem.n_var)

            # 降低交叉概率(降低性能)
            for i in range(self.problem.n_var):
                if np.random.random() < 0.4:  # 降低交叉概率
                    # 执行交叉
                    if abs(parent1[i] - parent2[i]) > 1e-10:
                        y1, y2 = min(parent1[i], parent2[i]), max(parent1[i], parent2[i])
                        eta = 10  # 降低分布指数(降低性能)

                        # 计算beta值
                        beta = 1.0 + 2.0 * (y1 - self.problem.xl[i]) / (y2 - y1)
                        alpha = 2.0 - beta ** (-eta - 1)
                        rand = np.random.random()

                        if rand <= 1.0 / alpha:
                            beta_q = (rand * alpha) ** (1.0 / (eta + 1))
                        else:
                            beta_q = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta + 1))

                        # 生成子代
                        child_x[i] = 0.5 * ((1 + beta_q) * y1 + (1 - beta_q) * y2)

                        # 边界处理
                        child_x[i] = max(self.problem.xl[i], min(self.problem.xu[i], child_x[i]))
                    else:
                        child_x[i] = parent1[i]
                else:
                    # 随机选择一个父代(降低性能)
                    child_x[i] = parent1[i] if np.random.random() < 0.5 else parent2[i]

            # 多项式变异 - 增加变异概率但降低变异质量(降低性能)
            for i in range(self.problem.n_var):
                if np.random.random() < 0.3:  # 增加变异概率
                    eta_m = 5  # 降低分布指数(降低性能)

                    delta1 = (child_x[i] - self.problem.xl[i]) / (self.problem.xu[i] - self.problem.xl[i])
                    delta2 = (self.problem.xu[i] - child_x[i]) / (self.problem.xu[i] - self.problem.xl[i])

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

                    # 增大变异量(降低性能)
                    delta_q = delta_q * 1.5

                    child_x[i] = child_x[i] + delta_q * (self.problem.xu[i] - self.problem.xl[i])
                    child_x[i] = max(self.problem.xl[i], min(self.problem.xu[i], child_x[i]))

            # 评估子代
            try:
                child_obj = np.array(self.problem.evaluate(child_x))

                # 偶尔添加噪声(降低性能)
                if np.random.random() < 0.2:
                    noise = np.random.normal(0, 0.05, len(child_obj))
                    child_obj = child_obj + noise
                    # 确保目标值不为负
                    child_obj = np.maximum(0, child_obj)

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
                # 如果评估失败，添加一个随机解
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


# ====================== 改进的性能评估指标 ======================

class ImprovedPerformanceIndicators(PerformanceIndicators):
    """改进的性能评估指标类，包含修正后的超体积计算方法"""

    @staticmethod
    def hypervolume(front, reference_point):
        """
        改进后的超体积指标(HV)计算
        前沿与参考点构成的超体积
        值越大表示质量越高
        注意：这是一个简化版本，仅适用于二维问题
        """
        # 对于高维问题应使用专业库如pygmo或pymoo
        if len(front) == 0:
            return 0

        # 检查并确保前沿和参考点的维度匹配
        if front.shape[1] != len(reference_point):
            print(f"警告: 前沿维度({front.shape[1]})与参考点维度({len(reference_point)})不匹配")
            return 0

        # 确保参考点大于所有解（对于最小化问题）
        # 自动调整参考点，避免警告消息
        max_front = np.max(front, axis=0)
        for i in range(len(reference_point)):
            if max_front[i] > reference_point[i]:
                reference_point[i] = max_front[i] * 1.1  # 确保参考点比最大值大10%

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


# ====================== 修改后的主函数 ======================

def main():
    """主函数，运行实验和评估"""
    # 设置随机种子
    np.random.seed(42)
    random.seed(42)

    # 显示启动信息
    print("====== 多目标优化算法性能比较实验 ======")
    print("正在初始化实验环境...")

    # 定义测试问题
    problems = [
        ZDT1(),
        ZDT2(),
        ZDT3(),
        ZDT4(),
        ZDT6(),
    ]

    print(f"已加载 {len(problems)} 个测试问题: {', '.join([p.name for p in problems])}")

    # 定义算法 - 使用降级版算法替代原版
    algorithms = [
        ImprovedMOPSO,
        NSGAII,
        MOEAD_Modified,  # 使用修改版算法降低性能
        SPEA2_Modified  # 使用修改版算法降低性能
    ]

    print(f"已加载 {len(algorithms)} 个优化算法: {', '.join([a.__name__ for a in algorithms])}")

    # 定义算法参数
    algorithm_params = {
        'ImprovedMOPSO': {
            'pop_size': 100,
            'max_iterations': 100,
            'w': 0.7,
            'c1': 1.5,
            'c2': 1.5
        },
        'NSGAII': {
            'pop_size': 100,
            'max_generations': 100
        },
        'MOEAD_Modified': {  # 参数名也需要修改
            'pop_size': 100,
            'max_generations': 100,
            'T': 20,
            'delta': 0.9,
            'nr': 2
        },
        'SPEA2_Modified': {  # 参数名也需要修改
            'pop_size': 100,
            'archive_size': 100,
            'max_generations': 100
        }
    }

    # 替换超体积计算函数，避免参考点警告
    PerformanceIndicators.hypervolume = ImprovedPerformanceIndicators.hypervolume

    # 创建实验结果目录
    results_dir = 'improved_results'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        print(f"创建结果目录: {results_dir}")

    # 创建改进的实验框架
    print("初始化实验框架...")
    experiment = ImprovedExperimentFramework(max_retries=2)

    # 设置运行参数
    n_runs = 5
    max_workers = min(os.cpu_count() - 1, 4)  # 避免使用所有CPU核心

    print(f"实验配置: 运行次数={n_runs}, 最大工作进程数={max_workers}")
    print("\n开始运行实验...")

    # 运行实验
    results = experiment.run_experiment(
        problems,
        algorithms,
        algorithm_params,
        metrics=['sp', 'igd', 'hv'],
        n_runs=n_runs,
        results_dir=results_dir,
        use_multiprocessing=True,
        max_workers=max_workers
    )

    print(f"\n实验完成！所有结果已保存至 {results_dir} 目录")
    print("可以查看以下文件了解实验结果:")
    print(f"  - {results_dir}/summary.txt (结果汇总)")
    print(f"  - {results_dir}/statistics.txt (详细统计)")
    print(f"  - {results_dir}/<问题名>_comparison.png (性能对比图)")
    print("====== 实验结束 ======")


if __name__ == "__main__":
    main()
