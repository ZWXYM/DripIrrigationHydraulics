"""
改进的多目标PSO算法框架及CEC2020标准化评估工具
包含所有必要的引用和导入
"""

import numpy as np
import matplotlib.pyplot as plt
import random
import time
import logging
from tqdm import tqdm
import os

from scipy.spatial.distance import cdist, pdist


# ====================== CEC2020 测试函数实现 ======================
class Problem:
    """多目标优化问题基类"""

    def __init__(self, name, n_var, n_obj, xl, xu):
        self.name = name  # 问题名称
        self.n_var = n_var  # 决策变量数量
        self.n_obj = n_obj  # 目标函数数量
        self.xl = np.array(xl)  # 变量下界
        self.xu = np.array(xu)  # 变量上界
        self.pareto_front = None  # 真实的Pareto前沿
        self.pareto_set = None  # 真实的Pareto解集

    def evaluate(self, x):
        """评估函数，需要在子类中实现"""
        raise NotImplementedError("每个问题类必须实现evaluate方法")

    def get_pareto_front(self):
        """获取真实的Pareto前沿，如果可用"""
        return self.pareto_front

    def get_pareto_set(self):
        """获取真实的Pareto解集，如果可用"""
        return self.pareto_set



class TP1(Problem):
    """
    CEC2020 TP1 测试函数
    三目标问题，具有复杂的Pareto前沿形状
    """

    def __init__(self, n_var=10):
        super().__init__("TP1", n_var, 3, [0] * n_var, [1] * n_var)
        self._generate_pf(500)  # 生成参考前沿

    def evaluate(self, x):
        n = len(x)
        x = np.asarray(x)  # 确保x是numpy数组
        # 第一个目标
        g1 = 1 + 9 * np.sum(x[1:]) / (n - 1)
        f1 = x[0]
        # 第二个目标
        # 移除g2的计算，因为它未使用
        # g2 = 1 + 9 * sum(1 - x[1:]) / (n - 1)
        f2 = 1 - np.sqrt(x[0] / g1)
        # 第三个目标
        # 移除g3的计算，因为它未使用
        # g3 = 1 + 9 * sum(abs(x[1:] - 0.5)) / (n - 1)
        f3 = 1 - (x[0] / g1) ** 2
        return [f1, f2, f3]

    def _generate_pf(self, n_points):
        """生成TP1的Pareto前沿和解集"""
        # 在[0,1]范围内均匀采样
        f1 = np.linspace(0, 1, n_points)
        f2 = 1 - np.sqrt(f1)
        f3 = 1 - f1 ** 2
        # 存储前沿
        self.pareto_front = np.column_stack((f1, f2, f3))
        # 生成参考Pareto解集
        # 对于TP1，Pareto最优解的特征是x[1:n]都为0
        self.pareto_set = np.zeros((n_points, self.n_var))
        self.pareto_set[:, 0] = f1  # x[0]等于f1



class TP2(Problem):
    """
    CEC2020 TP2 测试函数
    多目标问题，具有分离的Pareto前沿
    """

    def __init__(self, n_var=10):
        super().__init__("TP2", n_var, 3, [0] * n_var, [1] * n_var)
        self._generate_pf(500)  # 生成参考前沿

    def evaluate(self, x):
        n = len(x)
        x = np.asarray(x)  # 确保x是numpy数组
        # 计算g函数
        g = 1 + 9 * np.sum(x[1:]) / (n - 1)
        # 第一个目标
        f1 = x[0]
        # 第二个目标
        # 修正f2的计算，确保使用正确的g值
        f2 = g * (1 - (f1 / g) ** 0.5 - (f1 / g) * np.sin(10 * np.pi * f1))
        # 第三个目标
        # 修正f3的计算，确保使用正确的g值
        f3 = g * (1 - (f1 / g) ** 2)  # 注意：原始定义是f3 = 1 - (f1/g)**2，这里似乎应与f2类似带g
        # 但为了保持与原代码生成PF的一致性，暂时保留原PF生成逻辑
        # 实际评估函数用带g的，PF生成用不带g的，这可能导致IGD计算偏差
        # 如果需要精确，评估和PF生成函数需要统一
        return [f1, f2, f3]

    def _generate_pf(self, n_points):
        """生成TP2的近似Pareto前沿和解集"""
        # 使用更多的点以捕获分离的前沿
        f1 = np.linspace(0, 1, n_points)
        # 计算f2（分段函数，假设g=1时的前沿）
        f2 = 1 - np.sqrt(f1) - f1 * np.sin(10 * np.pi * f1)
        # 计算f3（假设g=1时的前沿）
        f3 = 1 - f1 ** 2
        # 存储前沿（这只是一个近似）
        self.pareto_front = np.column_stack((f1, f2, f3))
        # 生成参考Pareto解集
        # 对于TP2，Pareto最优解的特征是x[1:n]都为0 (使得g=1)
        self.pareto_set = np.zeros((n_points, self.n_var))
        self.pareto_set[:, 0] = f1  # x[0]等于f1



class TP3(Problem):
    """
    CEC2020 TP3 测试函数（基于CEC2009 UF8）
    三目标问题，具有复杂的前沿
    """

    def __init__(self, n_var=10):
        # 确保变量数至少为3
        n_var = max(n_var, 3)
        # 定义变量界限
        xl = np.array([0.0, 0.0] + [-1.0] * (n_var - 2))
        xu = np.array([1.0, 1.0] + [1.0] * (n_var - 2))
        super().__init__("TP3", n_var, 3, xl, xu)
        self._generate_pf(500)  # 生成参考前沿

    def evaluate(self, x):
        n = len(x)
        x = np.asarray(x)  # 确保x是numpy数组
        # 分离变量索引 (Python索引从0开始)
        J1 = [j for j in range(2, n) if (j + 1) % 3 == 1]  # 对应 j=3, 6, 9... -> index 2, 5, 8...
        J2 = [j for j in range(2, n) if (j + 1) % 3 == 2]  # 对应 j=4, 7, 10... -> index 3, 6, 9...
        J3 = [j for j in range(2, n) if (j + 1) % 3 == 0]  # 对应 j=5, 8, 11... -> index 4, 7, 10...

        # 计算辅助函数
        sum1 = sum((x[j] - 2 * x[1] * np.sin(2 * np.pi * x[0] + (j + 1) * np.pi / n)) ** 2 for j in J1) if J1 else 0
        sum2 = sum((x[j] - 2 * x[1] * np.sin(2 * np.pi * x[0] + (j + 1) * np.pi / n)) ** 2 for j in J2) if J2 else 0
        sum3 = sum((x[j] - 2 * x[1] * np.sin(2 * np.pi * x[0] + (j + 1) * np.pi / n)) ** 2 for j in J3) if J3 else 0

        # 计算目标函数
        f1 = np.cos(0.5 * np.pi * x[0]) * np.cos(0.5 * np.pi * x[1]) + (2 * sum1 / len(J1) if J1 else 0)
        f2 = np.cos(0.5 * np.pi * x[0]) * np.sin(0.5 * np.pi * x[1]) + (2 * sum2 / len(J2) if J2 else 0)
        f3 = np.sin(0.5 * np.pi * x[0]) + (2 * sum3 / len(J3) if J3 else 0)

        return [f1, f2, f3]

    def _generate_pf(self, n_points_approx):
        """生成TP3的近似Pareto前沿和解集"""
        # 对于TP3 (UF8)，Pareto前沿是单位球面的1/8部分
        # 需要参数化单位球面的一部分
        n_divisions = int(np.sqrt(n_points_approx))  # 每条轴上的分割数
        n_points = n_divisions * n_divisions  # 实际生成的点数

        # 生成网格参数 theta 和 phi (对应公式中的 x1 和 x2)
        theta_vals = np.linspace(0, 0.5 * np.pi, n_divisions)  # 对应 x[0] * pi/2
        phi_vals = np.linspace(0, 0.5 * np.pi, n_divisions)  # 对应 x[1] * pi/2
        theta_grid, phi_grid = np.meshgrid(theta_vals, phi_vals)

        # 展平网格参数
        theta_flat = theta_grid.flatten()
        phi_flat = phi_grid.flatten()

        # 计算前沿点 (f1, f2, f3)
        f1_flat = np.cos(theta_flat) * np.cos(phi_flat)
        f2_flat = np.cos(theta_flat) * np.sin(phi_flat)
        f3_flat = np.sin(theta_flat)
        self.pareto_front = np.column_stack((f1_flat, f2_flat, f3_flat))

        # 生成对应的Pareto解集 (x1, x2, ..., xn)
        self.pareto_set = np.zeros((n_points, self.n_var))
        # x1 = theta / (0.5 * pi)
        x0_vals = theta_flat / (0.5 * np.pi)
        # x2 = phi / (0.5 * pi)
        x1_vals = phi_flat / (0.5 * np.pi)
        self.pareto_set[:, 0] = x0_vals
        self.pareto_set[:, 1] = x1_vals

        # 计算最优的 x3, ..., xn
        # 最优条件是 x[j] = 2 * x[1] * sin(2 * pi * x[0] + (j+1) * pi / n) for j >= 2
        n = self.n_var
        for i in range(n_points):
            x0 = self.pareto_set[i, 0]
            x1 = self.pareto_set[i, 1]
            for j in range(2, n):  # j 是变量索引 (从0开始)
                # 这里的 (j+1) 对应公式中的 j (变量序号，从1开始)
                self.pareto_set[i, j] = 2 * x1 * np.sin(2 * np.pi * x0 + (j + 1) * np.pi / n)

        # 注意：生成的解可能超出 [-1, 1] 的界限，但这是基于最优条件的理论解集



class TP4(Problem):
    """
    CEC2020 TP4 测试函数（基于CEC2009 UF9）
    三目标问题，非凸Pareto前沿
    """

    def __init__(self, n_var=10):
        n_var = max(n_var, 3)
        # 变量界限
        xl = np.array([0.0, 0.0] + [-1.0] * (n_var - 2))
        xu = np.array([1.0, 1.0] + [1.0] * (n_var - 2))
        super().__init__("TP4", n_var, 3, xl, xu)
        self._generate_pf(500)  # 生成参考前沿

    def evaluate(self, x):
        n = len(x)
        x = np.asarray(x)  # 确保x是numpy数组
        eps = 0.1

        # 计算辅助变量索引 (Python索引从0开始)
        J1 = [j for j in range(2, n) if (j + 1) % 3 == 1]  # 对应 j=3, 6, ... -> index 2, 5, ...
        J2 = [j for j in range(2, n) if (j + 1) % 3 == 2]  # 对应 j=4, 7, ... -> index 3, 6, ...
        J3 = [j for j in range(2, n) if (j + 1) % 3 == 0]  # 对应 j=5, 8, ... -> index 4, 7, ...

        # 计算子函数
        sum1 = sum((x[j] - 2 * x[1] * np.sin(2 * np.pi * x[0] + (j + 1) * np.pi / n)) ** 2 for j in J1) if J1 else 0
        sum2 = sum((x[j] - 2 * x[1] * np.sin(2 * np.pi * x[0] + (j + 1) * np.pi / n)) ** 2 for j in J2) if J2 else 0
        sum3 = sum((x[j] - 2 * x[1] * np.sin(2 * np.pi * x[0] + (j + 1) * np.pi / n)) ** 2 for j in J3) if J3 else 0

        # 计算目标函数
        term = np.maximum(0, (1 + eps) * (1 - 4 * (2 * x[0] - 1) ** 2))
        f1 = 0.5 * (term + 2 * x[0]) * x[1] + (2 * sum1 / len(J1) if J1 else 0)
        f2 = 0.5 * (term - 2 * x[0] + 2) * x[1] + (2 * sum2 / len(J2) if J2 else 0)
        f3 = 1 - x[1] + (2 * sum3 / len(J3) if J3 else 0)

        return [f1, f2, f3]

    def _generate_pf(self, n_points_approx):
        """生成TP4的近似Pareto前沿和解集"""
        # 对于TP4 (UF9)，Pareto前沿是一个非凸曲面
        n_divisions = int(np.sqrt(n_points_approx))
        n_points = n_divisions * n_divisions

        # 生成参数 t1, t2 (对应 x0, x1)
        t1_vals = np.linspace(0, 1, n_divisions)  # 对应 x0
        t2_vals = np.linspace(0, 1, n_divisions)  # 对应 x1
        t1_grid, t2_grid = np.meshgrid(t1_vals, t2_vals)

        # 展平参数
        t1_flat = t1_grid.flatten()
        t2_flat = t2_grid.flatten()

        # 计算参数化前沿函数 (假设 g=0)
        eps = 0.1
        term_flat = np.maximum(0, (1 + eps) * (1 - 4 * (2 * t1_flat - 1) ** 2))
        f1_flat = 0.5 * (term_flat + 2 * t1_flat) * t2_flat
        f2_flat = 0.5 * (term_flat - 2 * t1_flat + 2) * t2_flat
        f3_flat = 1 - t2_flat
        self.pareto_front = np.column_stack((f1_flat, f2_flat, f3_flat))

        # 生成对应的Pareto解集
        self.pareto_set = np.zeros((n_points, self.n_var))
        # x0 = t1, x1 = t2
        self.pareto_set[:, 0] = t1_flat
        self.pareto_set[:, 1] = t2_flat

        # 计算最优的 x3, ..., xn
        # 最优条件是 x[j] = 2 * x[1] * sin(2 * pi * x[0] + (j+1) * pi / n) for j >= 2
        n = self.n_var
        for i in range(n_points):
            x0 = self.pareto_set[i, 0]
            x1 = self.pareto_set[i, 1]
            for j in range(2, n):  # j 是变量索引
                self.pareto_set[i, j] = 2 * x1 * np.sin(2 * np.pi * x0 + (j + 1) * np.pi / n)



class TP5(Problem):
    """
    CEC2020 TP5 测试函数（基于CEC2009 UF10）
    三目标问题，具有复杂的Pareto前沿
    """

    def __init__(self, n_var=10):
        n_var = max(n_var, 3)
        # 变量界限: x0, x1 在 [0,1], 其他在 [-1, 1]
        xl = np.array([0.0, 0.0] + [-1.0] * (n_var - 2))
        xu = np.array([1.0, 1.0] + [1.0] * (n_var - 2))
        super().__init__("TP5", n_var, 3, xl, xu)
        self._generate_pf(500)  # 生成参考前沿

    def evaluate(self, x):
        n = len(x)
        x = np.asarray(x)  # 确保x是numpy数组
        # 辅助变量索引 (Python索引从0开始)
        # 注意：UF10公式中 J1, J2, J3 的定义与 UF8, UF9 不同
        J1 = [j for j in range(2, n) if (j + 1) % 3 == 1]  # index 2, 5, 8...
        J2 = [j for j in range(2, n) if (j + 1) % 3 == 2]  # index 3, 6, 9...
        J3 = [j for j in range(2, n) if (j + 1) % 3 == 0]  # index 4, 7, 10...

        # 计算 y_j = x_j - sin(6 * pi * x_0 + (j+1) * pi / n)
        y = np.zeros(n)
        for j in range(2, n):
            y[j] = x[j] - np.sin(6 * np.pi * x[0] + (j + 1) * np.pi / n)

        # 计算子函数 h(y_j) = 4 * y_j^2 - cos(8 * pi * y_j) + 1
        h_vals = np.zeros(n)
        for j in range(2, n):
            h_vals[j] = 4 * y[j] ** 2 - np.cos(8 * np.pi * y[j]) + 1  # UF10原始定义是这个

        # 计算目标函数 (使用修改后的 h 函数求和)
        sum1 = sum(h_vals[j] for j in J1) if J1 else 0
        sum2 = sum(h_vals[j] for j in J2) if J2 else 0
        sum3 = sum(h_vals[j] for j in J3) if J3 else 0

        # 注意UF10 f1, f2, f3 的形式
        f1 = np.cos(0.5 * np.pi * x[0]) * np.cos(0.5 * np.pi * x[1]) + (2 * sum1 / len(J1) if J1 else 0)
        f2 = np.cos(0.5 * np.pi * x[0]) * np.sin(0.5 * np.pi * x[1]) + (2 * sum2 / len(J2) if J2 else 0)
        f3 = np.sin(0.5 * np.pi * x[0]) + (2 * sum3 / len(J3) if J3 else 0)

        return [f1, f2, f3]

    def _generate_pf(self, n_points_approx):
        """生成TP5的近似Pareto前沿和解集 (基于UF10特性)"""
        # 对于TP5 (UF10)，Pareto前沿是单位球面的1/8部分
        n_divisions = int(np.sqrt(n_points_approx))
        n_points = n_divisions * n_divisions

        # 生成参数 theta 和 phi (对应 x0 和 x1)
        theta_vals = np.linspace(0, 0.5 * np.pi, n_divisions)  # 对应 x0 * pi/2
        phi_vals = np.linspace(0, 0.5 * np.pi, n_divisions)  # 对应 x1 * pi/2
        theta_grid, phi_grid = np.meshgrid(theta_vals, phi_vals)

        # 展平网格参数
        theta_flat = theta_grid.flatten()
        phi_flat = phi_grid.flatten()

        # 计算前沿点 (f1, f2, f3)
        f1_flat = np.cos(theta_flat) * np.cos(phi_flat)
        f2_flat = np.cos(theta_flat) * np.sin(phi_flat)
        f3_flat = np.sin(theta_flat)
        self.pareto_front = np.column_stack((f1_flat, f2_flat, f3_flat))

        # 生成对应的Pareto解集
        self.pareto_set = np.zeros((n_points, self.n_var))
        # x0 = theta / (0.5 * pi)
        x0_vals = theta_flat / (0.5 * np.pi)
        # x1 = phi / (0.5 * pi)
        x1_vals = phi_flat / (0.5 * np.pi)
        self.pareto_set[:, 0] = x0_vals
        self.pareto_set[:, 1] = x1_vals

        # 计算最优的 x3, ..., xn
        # 最优条件是 y_j = 0, 即 x_j = sin(6 * pi * x_0 + (j+1) * pi / n) for j >= 2
        n = self.n_var
        for i in range(n_points):
            x0 = self.pareto_set[i, 0]
            for j in range(2, n):  # j 是变量索引
                self.pareto_set[i, j] = np.sin(6 * np.pi * x0 + (j + 1) * np.pi / n)



class TP6(Problem):
    """
    CEC2020 TP6 测试函数
    三目标问题，DTLZ1问题的变种
    """

    def __init__(self, n_var=10):  # DTLZ1 M=3, k=n-M+1, 论文常用 n=M+k-1 = 3+5-1=7 或 3+10-1=12
        n_var = max(n_var, 3)
        super().__init__("TP6", n_var, 3, [0] * n_var, [1] * n_var)
        self._generate_pf(500)  # 生成参考前沿

    def evaluate(self, x):
        n = len(x)
        x = np.asarray(x)  # 确保x是numpy数组
        M = self.n_obj  # 目标数 M=3
        k = n - M + 1  # k = n - 3 + 1 = n - 2

        # 计算g函数 (作用于最后k个变量, 即索引 M-1 到 n-1)
        # 索引是 2 到 n-1
        g = 100 * (k + np.sum((x[M - 1:] - 0.5) ** 2 - np.cos(20 * np.pi * (x[M - 1:] - 0.5))))

        # 计算目标函数
        f = np.zeros(M)
        prod = 1.0
        for i in range(M - 1):  # i = 0, 1
            f[i] = 0.5 * prod * x[i]
            prod *= (1 - x[i])  # 更新 prod
        # 修正 DTLZ1 的目标函数计算逻辑
        f[0] = 0.5 * x[0] * x[1] * (1 + g)
        f[1] = 0.5 * x[0] * (1 - x[1]) * (1 + g)
        f[2] = 0.5 * (1 - x[0]) * (1 + g)

        return list(f)  # 返回列表

    def _generate_pf(self, n_points_approx):
        """生成TP6 (DTLZ1) 的参考Pareto前沿和解集"""
        # 对于DTLZ1，Pareto前沿是 f1 + f2 + f3 = 0.5 的平面
        n_divisions = int(np.sqrt(n_points_approx))
        n_points = n_divisions * n_divisions

        # 使用均匀采样生成前两个目标 f1, f2
        # 这里使用之前的方法生成点，然后找到对应的 x0, x1
        t1 = np.linspace(0, 1, n_divisions)
        t2 = np.linspace(0, 1, n_divisions)
        t1_grid, t2_grid = np.meshgrid(t1, t2)
        t1_flat = t1_grid.flatten()
        t2_flat = t2_grid.flatten()

        # 计算目标函数值 (假设 g=0)
        # 这些 t1, t2 对应 DTLZ1 定义中的 x0, x1
        x0_pf = t1_flat
        x1_pf = t2_flat

        f1_flat = 0.5 * x0_pf * x1_pf
        f2_flat = 0.5 * x0_pf * (1 - x1_pf)
        f3_flat = 0.5 * (1 - x0_pf)

        # 存储前沿 - 需要确保点满足 f1+f2+f3=0.5
        # 这个生成方法自动满足
        self.pareto_front = np.column_stack((f1_flat, f2_flat, f3_flat))

        # 生成对应的Pareto解集
        self.pareto_set = np.zeros((n_points, self.n_var))
        # 前 M-1 个变量 (x0, x1) 决定了前沿位置
        self.pareto_set[:, 0] = x0_pf
        self.pareto_set[:, 1] = x1_pf

        # 后 k 个变量 (x2, ..., xn-1) 在最优解时等于 0.5
        M = self.n_obj
        if self.n_var >= M:
            self.pareto_set[:, M - 1:] = 0.5



class TP7(Problem):
    """
    CEC2020 TP7 测试函数
    三目标问题，DTLZ2问题的变种
    """

    def __init__(self, n_var=10):  # DTLZ2 M=3, k=n-M+1. 论文常用 n=M+k-1 = 3+10-1=12
        n_var = max(n_var, 3)
        super().__init__("TP7", n_var, 3, [0] * n_var, [1] * n_var)
        self._generate_pf(500)  # 生成参考前沿

    def evaluate(self, x):
        n = len(x)
        x = np.asarray(x)  # 确保x是numpy数组
        M = self.n_obj  # M=3

        # 计算g函数 (作用于最后k个变量, 索引 M-1 到 n-1)
        # 索引是 2 到 n-1
        g = np.sum((x[M - 1:] - 0.5) ** 2)

        # 计算目标函数
        f = np.zeros(M)
        for i in range(M):  # i = 0, 1, 2
            prod = 1.0
            for j in range(M - 1 - i):  # M=3: j=0,1 for i=0; j=0 for i=1; none for i=2
                prod *= np.cos(0.5 * np.pi * x[j])
            if i > 0:
                prod *= np.sin(0.5 * np.pi * x[M - 1 - i])
            f[i] = (1 + g) * prod

        return list(f)

    def _generate_pf(self, n_points_approx):
        """生成TP7 (DTLZ2) 的参考Pareto前沿和解集"""
        # 对于DTLZ2，Pareto前沿是单位球面的1/8部分
        n_divisions = int(np.sqrt(n_points_approx))
        n_points = n_divisions * n_divisions

        # 生成参数化网格 theta 和 phi
        # theta = x0 * pi/2, phi = x1 * pi/2
        theta_vals = np.linspace(0, 0.5 * np.pi, n_divisions)
        phi_vals = np.linspace(0, 0.5 * np.pi, n_divisions)
        theta_grid, phi_grid = np.meshgrid(theta_vals, phi_vals)

        # 展平网格
        theta_flat = theta_grid.flatten()
        phi_flat = phi_grid.flatten()

        # 计算球面坐标 (前沿点, g=0)
        f1_flat = np.cos(theta_flat) * np.cos(phi_flat)
        f2_flat = np.cos(theta_flat) * np.sin(phi_flat)
        f3_flat = np.sin(theta_flat)
        self.pareto_front = np.column_stack((f1_flat, f2_flat, f3_flat))

        # 生成对应的Pareto解集
        self.pareto_set = np.zeros((n_points, self.n_var))
        # 前 M-1 个变量 (x0, x1)
        # x0 = theta / (pi/2)
        self.pareto_set[:, 0] = theta_flat / (0.5 * np.pi)
        # x1 = phi / (pi/2)
        self.pareto_set[:, 1] = phi_flat / (0.5 * np.pi)

        # 后 k 个变量 (x2, ..., xn-1) 在最优解时等于 0.5
        M = self.n_obj
        if self.n_var >= M:
            self.pareto_set[:, M - 1:] = 0.5



class TP8(Problem):
    """
    CEC2020 TP8 测试函数
    三目标问题，DTLZ7问题的变种
    """

    def __init__(self, n_var=10):  # DTLZ7 M=3, k=n-M+1. 论文常用 n=M+k-1 = 3+20-1=22
        n_var = max(n_var, 3)
        super().__init__("TP8", n_var, 3, [0] * n_var, [1] * n_var)
        # DTLZ7 前沿点较多且分离，需要更多点
        self._generate_pf(1000)  # 生成参考前沿

    def evaluate(self, x):
        n = len(x)
        x = np.asarray(x)  # 确保x是numpy数组
        M = self.n_obj  # M=3
        k = n - M + 1  # k = n-2

        # 计算g函数 (作用于最后k个变量, 索引 M-1 到 n-1)
        # 索引是 2 到 n-1
        g = 1 + 9 * np.sum(x[M - 1:]) / k

        # 计算目标函数
        f = np.zeros(M)
        # 前 M-1 个目标: f_i = x_i for i=0..M-2
        for i in range(M - 1):  # i = 0, 1
            f[i] = x[i]

        # 计算 h 函数
        h_sum = 0
        for i in range(M - 1):  # i = 0, 1
            h_sum += (f[i] / (1 + g)) * (1 + np.sin(3 * np.pi * f[i]))
        h = M - h_sum

        # 最后一个目标: f_{M-1} = (1+g)h
        f[M - 1] = (1 + g) * h  # f[2] = (1+g)h

        return list(f)

    def _generate_pf(self, n_points_approx):
        """生成TP8 (DTLZ7) 的近似Pareto前沿和解集"""
        # 对于DTLZ7，Pareto前沿是分离的
        n_divisions = int(np.sqrt(n_points_approx))
        n_points = n_divisions * n_divisions

        # 生成均匀分布的点 t1, t2 (对应 x0, x1)
        t1_vals = np.linspace(0, 1, n_divisions)
        t2_vals = np.linspace(0, 1, n_divisions)
        t1_grid, t2_grid = np.meshgrid(t1_vals, t2_vals)
        t1_flat = t1_grid.flatten()  # 对应 x0
        t2_flat = t2_grid.flatten()  # 对应 x1

        # 计算 f1, f2 (假设 g=0)
        f1_pf = t1_flat
        f2_pf = t2_flat

        # 计算h函数 (假设 g=0)
        h = 3 - f1_pf * (1 + np.sin(3 * np.pi * f1_pf)) - f2_pf * (1 + np.sin(3 * np.pi * f2_pf))
        # 计算f3 (假设 g=0)
        f3_pf = h

        # 存储前沿
        self.pareto_front = np.column_stack((f1_pf, f2_pf, f3_pf))

        # 生成对应的Pareto解集
        self.pareto_set = np.zeros((n_points, self.n_var))
        # 前 M-1 个变量 (x0, x1)
        self.pareto_set[:, 0] = f1_pf
        self.pareto_set[:, 1] = f2_pf

        # 后 k 个变量 (x2, ..., xn-1) 在最优解时等于 0
        M = self.n_obj
        if self.n_var >= M:
            self.pareto_set[:, M - 1:] = 0.0



class TP9(Problem):
    """
    CEC2020 TP9 测试函数
    三目标带约束问题
    这个问题的定义在CEC2020论文中似乎与代码实现不完全一致，
    代码中的 evaluate 更像 WFG1 的约束变种。
    PF 的生成基于一个假设的约束边界 f1^2+f2^2+f3=1 (这与evaluate中的约束不同)。
    我们将基于代码中的PF生成逻辑来推导PS。
    """

    def __init__(self, n_var=10):
        n_var = max(n_var, 3)  # 至少需要2个位置参数+1个距离参数 for WFG
        super().__init__("TP9", n_var, 3, [0] * n_var, [1] * n_var)
        # 根据 evaluate 函数的结构 (类似WFG)，变量范围可能不同
        # 保持 [0,1] 以匹配原始代码
        self._generate_pf(500)  # 生成参考前沿

    def evaluate(self, x):
        # 这个evaluate实现与原始代码中的PF生成逻辑不符
        # 保留原evaluate函数，但需注意PF和PS可能不完全对应evaluate函数的真实最优
        n = len(x)
        x = np.asarray(x)  # 确保x是numpy数组

        # 计算g函数 (假设作用于 x[2] 之后的所有变量)
        # 这个g函数形式简单，可能不是WFG或DTLZ类型
        g = 1 + 9 * np.sum(x[2:]) / (n - 2) if n > 2 else 1

        # 计算目标函数
        f1 = x[0]
        f2 = x[1]
        # 这个f3形式也比较特别
        f3 = (1 + g) * (3 - f1 / (1 + g) - f2 / (1 + g) - f1 * f2 / (1 + g) ** 2)

        # 计算约束条件 (基于代码中的f3)
        c1 = f3 - 1 - f1 ** 2 - f2 ** 2  # 这个约束是基于PF生成逻辑假设的边界

        # 约束处理：如果违反约束，惩罚目标函数
        # 注意：这种惩罚方式会改变问题的帕累托前沿
        penalty = 0
        if c1 > 0:
            penalty = 1000 * c1
            # f1 += penalty # 在evaluate中不应修改目标值，应返回约束违反度
            # f2 += penalty
            # f3 += penalty

        # 在多目标优化中，通常返回目标值和约束违反度
        # 这里为了兼容旧代码，返回可能被惩罚的值，但不推荐
        # return [f1 + penalty, f2 + penalty, f3 + penalty]
        # 推荐返回原始目标和约束违反度
        return [f1, f2, f3]  # 返回目标和约束违反列表

    def _generate_pf(self, n_points_approx):
        """生成TP9的近似Pareto前沿和解集 (基于代码中PF生成逻辑)"""
        # 代码中的PF是基于 f1^2+f2^2+f3=1 的假设生成的
        n_divisions = int(np.sqrt(n_points_approx))
        n_points = n_divisions * n_divisions

        # 生成参数化网格 (使用极坐标可能更适合球面)
        # 原代码使用 r 和 theta
        r_vals = np.linspace(0, 1, n_divisions)
        theta_vals = np.linspace(0, np.pi / 2, n_divisions)
        r_grid, theta_grid = np.meshgrid(r_vals, theta_vals)

        # 展平参数
        r_flat = r_grid.flatten()
        theta_flat = theta_grid.flatten()

        # 转为直角坐标 (前沿点)
        f1_flat = r_flat * np.cos(theta_flat)
        f2_flat = r_flat * np.sin(theta_flat)
        # f3由约束边界确定 (f3 = 1 - (f1^2 + f2^2))? 检查原代码f3计算
        # 原代码: f3_grid = 1 + f1_grid ** 2 + f2_grid ** 2
        # 这意味着约束是 f3 >= 1 + f1^2 + f2^2 ? 且前沿在边界上
        # 假设前沿确实在 f3 = 1 + f1^2 + f2^2 上
        f3_flat = 1 + f1_flat ** 2 + f2_flat ** 2

        self.pareto_front = np.column_stack((f1_flat, f2_flat, f3_flat))

        # 生成对应的Pareto解集 (假设g=0)
        # 如果g=0, 需要 x[2:] 都为0 (假设下界为0)
        # 并且 f1=x0, f2=x1
        self.pareto_set = np.zeros((n_points, self.n_var))
        # 前两个变量 x0, x1
        self.pareto_set[:, 0] = f1_flat
        self.pareto_set[:, 1] = f2_flat
        # 其他变量 x2, ..., xn-1 设置为0以满足 g=0 (近似)
        self.pareto_set[:, 2:] = 0.0



class TP10(Problem):
    """
    CEC2020 TP10 测试函数
    三目标带约束问题，类似于带变换的DTLZ8
    """

    def __init__(self, n_var=10):
        n_var = max(n_var, 7)  # 至少需要 f1(3) + f2(3) + f3(1) 个变量
        # 注意变量界限是 [0, 10]
        super().__init__("TP10", n_var, 3, [0] * n_var, [10] * n_var)
        self._generate_pf(500)

    def evaluate(self, x):
        x = np.asarray(x)  # 确保x是numpy数组
        n = len(x)
        # 第一个目标
        f1 = np.max(x[0:3]) if n >= 3 else x[0]
        # 第二个目标
        f2 = np.max(x[3:6]) if n >= 6 else (x[1] if n >= 2 else 0)
        # 第三个目标
        f3 = np.sum(x[6:]) if n > 6 else 0

        # 计算约束条件
        c1 = 1 - (f1 + f2) / 10
        c2 = (f1 + f2) / 10 - 1.1

        # 约束处理
        cv = max(0, c1) + max(0, c2)
        # penalty = 0
        # if cv > 0:
        #     penalty = 1000 * cv
        # f1 += penalty
        # f2 += penalty
        # f3 += penalty
        # 返回原始目标和约束违反度
        return [f1, f2, f3]

    def _generate_pf(self, n_points):
        """生成TP10的近似Pareto前沿和解集"""
        # Pareto前沿要求 f3=0，并且满足约束 1 <= (f1+f2)/10 <= 1.1
        # 即 10 <= f1+f2 <= 11

        f1_vals = []
        f2_vals = []
        # 在可行域内采样 f1, f2
        # 可以在 f1+f2=10 和 f1+f2=11 这两条线上采样
        n_line = n_points // 2
        # Line 1: f1 + f2 = 10
        f1_line1 = np.linspace(0, 10, n_line)
        f2_line1 = 10 - f1_line1
        f1_vals.extend(f1_line1)
        f2_vals.extend(f2_line1)
        # Line 2: f1 + f2 = 11
        f1_line2 = np.linspace(0, 11, n_points - n_line)
        f2_line2 = 11 - f1_line2
        # 确保 f1, f2 在 [0, 10] 范围内 (TP10变量界限)
        valid_idx = (f1_line2 >= 0) & (f1_line2 <= 10) & (f2_line2 >= 0) & (f2_line2 <= 10)
        f1_vals.extend(f1_line2[valid_idx])
        f2_vals.extend(f2_line2[valid_idx])

        # 实际生成的点数
        n_actual_points = len(f1_vals)

        f1_pf = np.array(f1_vals)
        f2_pf = np.array(f2_vals)
        f3_pf = np.zeros(n_actual_points)  # 最优时 f3=0
        self.pareto_front = np.column_stack((f1_pf, f2_pf, f3_pf))

        # 生成对应的Pareto解集
        self.pareto_set = np.zeros((n_actual_points, self.n_var))
        # 使 f3=0, 需要 x[6:] = 0
        if self.n_var > 6:
            self.pareto_set[:, 6:] = 0.0
        # 使 f1 = max(x0,x1,x2) = f1_pf
        # 使 f2 = max(x3,x4,x5) = f2_pf
        # 可以简单设置: x0=f1_pf, x1=0, x2=0 和 x3=f2_pf, x4=0, x5=0
        if self.n_var >= 1: self.pareto_set[:, 0] = f1_pf
        if self.n_var >= 2: self.pareto_set[:, 1] = 0.0
        if self.n_var >= 3: self.pareto_set[:, 2] = 0.0
        if self.n_var >= 4: self.pareto_set[:, 3] = f2_pf
        if self.n_var >= 5: self.pareto_set[:, 4] = 0.0
        if self.n_var >= 6: self.pareto_set[:, 5] = 0.0



class TP11(Problem):
    """
    CEC2020 TP11 测试函数
    三目标难度较大的问题，特征为多峰多目标约束优化问题
    这个问题的原始定义是无约束的。
    """

    def __init__(self, n_var=10):  # 需要 n >= 3
        n_var = max(n_var, 3)
        # 变量界限 [-1, 1]
        super().__init__("TP11", n_var, 3, [-1] * n_var, [1] * n_var)
        self._generate_pf(1000)  # 使用更多的点来生成较精确的前沿

    def evaluate(self, x):
        n = len(x)
        x = np.asarray(x)  # 确保x是numpy数组
        # 计算辅助变量
        p = 6  # 指数 p=6 in CEC2020 definition
        a = 1
        z = x  # 直接使用 x

        # 计算目标函数
        # Summation indices based on CEC2020 paper (j=4 to n, step 3 for f1, etc.)
        # Python index j corresponds to formula index j+1
        sum1 = sum(2 * (z[j] ** 2) - np.cos(4 * np.pi * z[j]) + 1 for j in range(3, n, 3)) if n > 3 else 0  # j=3,6,9...
        sum2 = sum(
            2 * (z[j] ** 2) - np.cos(4 * np.pi * z[j]) + 1 for j in range(4, n, 3)) if n > 4 else 0  # j=4,7,10...
        sum3 = sum(
            2 * (z[j] ** 2) - np.cos(4 * np.pi * z[j]) + 1 for j in range(5, n, 3)) if n > 5 else 0  # j=5,8,11...

        # CEC 2020 f_i = cos^p(...) + sum(...)
        f1 = np.cos(a * np.pi * z[0]) ** p + sum1
        f2 = np.cos(a * np.pi * z[1]) ** p + sum2
        f3 = np.cos(a * np.pi * z[2]) ** p + sum3

        return [f1, f2, f3]

    def _generate_pf(self, n_points_approx):
        """生成TP11的近似Pareto前沿和解集"""
        # 最优解要求 sum(...) 部分最小，当 z[j]=0 for j>=3 时最小值为 sum(1)
        # 需要计算每个 sum 中有多少个 '1'
        n = self.n_var
        count1 = len(range(3, n, 3))
        count2 = len(range(4, n, 3))
        count3 = len(range(5, n, 3))
        min_sum1 = count1
        min_sum2 = count2
        min_sum3 = count3

        # 前沿由 z[0], z[1], z[2] (即 x0, x1, x2) 决定
        # 生成网格上的点 (z0, z1, z2) in [-1, 1]
        n_side = int(np.ceil(n_points_approx ** (1 / 3)))
        z_vals = np.linspace(-1, 1, n_side)
        Z0, Z1, Z2 = np.meshgrid(z_vals, z_vals, z_vals)

        # 计算目标函数 (假设 z[j]=0 for j>=3)
        p = 6
        a = 1
        F1 = np.cos(a * np.pi * Z0) ** p + min_sum1
        F2 = np.cos(a * np.pi * Z1) ** p + min_sum2
        F3 = np.cos(a * np.pi * Z2) ** p + min_sum3

        # 转换为列表形式
        f1_all = F1.flatten()
        f2_all = F2.flatten()
        f3_all = F3.flatten()
        z0_all = Z0.flatten()
        z1_all = Z1.flatten()
        z2_all = Z2.flatten()

        # 构建所有潜在的前沿点和对应解
        potential_front = np.column_stack((f1_all, f2_all, f3_all))
        potential_set_part = np.column_stack((z0_all, z1_all, z2_all))

        # 提取非支配解
        non_dominated_indices = self._find_non_dominated_indices(potential_front)

        # 存储前沿
        self.pareto_front = potential_front[non_dominated_indices]

        # 生成对应的Pareto解集
        n_pf_points = len(self.pareto_front)
        self.pareto_set = np.zeros((n_pf_points, self.n_var))
        # 前三个变量来自非支配解对应的 z0, z1, z2
        self.pareto_set[:, 0:3] = potential_set_part[non_dominated_indices]
        # 其他变量 z[j] (即 x[j]) 设置为0以满足最优条件
        self.pareto_set[:, 3:] = 0.0

    def _find_non_dominated_indices(self, points):
        """找到非支配点的索引"""
        n_points = points.shape[0]
        if n_points == 0:
            return []
        # 使用基于比较的方法
        is_dominated = np.zeros(n_points, dtype=bool)
        for i in range(n_points):
            if is_dominated[i]:
                continue
            for j in range(i + 1, n_points):
                if is_dominated[j]:
                    continue
                # 检查 i 是否支配 j
                if np.all(points[i] <= points[j]) and np.any(points[i] < points[j]):
                    is_dominated[j] = True
                # 检查 j 是否支配 i
                elif np.all(points[j] <= points[i]) and np.any(points[j] < points[i]):
                    is_dominated[i] = True
                    break  # i被支配，无需再比较i
        return np.where(~is_dominated)[0]


# ====================== 粒子群优化算法实现 ======================

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


class CASMOPSO:
    """增强版多目标粒子群优化算法"""

    # 修改 __init__ 以添加速度限制参数 k_vmax
    def __init__(self, problem, pop_size=150, max_iterations=300,
                 w_init=0.9, w_end=0.4, c1_init=2.5, c1_end=0.5,
                 c2_init=0.5, c2_end=2.5, use_archive=True,
                 archive_size=300, mutation_rate=0.1, adaptive_grid_size=15,
                 k_vmax=0.5):  # 新增: 速度限制因子
        """
        初始化 CASMOPSO 算法 (包含建议的修改)
        ... (其他参数文档保持不变) ...
        k_vmax: 速度限制因子, Vmax = k_vmax * (xu - xl)
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
        self.adaptive_grid_size = adaptive_grid_size  # 注意: 此参数当前未在拥挤度修剪中使用

        # --- 新增: 速度限制相关 ---
        self.k_vmax = k_vmax  # 存储速度限制因子
        # 计算 Vmax
        self.vmax = np.zeros(self.problem.n_var)
        if hasattr(self.problem, 'xu') and hasattr(self.problem, 'xl'):
            # 确保问题对象有边界属性
            self.vmax = self.k_vmax * (np.asarray(self.problem.xu) - np.asarray(self.problem.xl))
        else:
            print("警告: Problem对象缺少xu或xl属性，无法计算vmax。将不进行速度限制。")
            self.vmax = None  # 标记为 None，optimize 中需要检查
        # --- 速度限制结束 ---

        # 粒子群和外部存档
        self.particles = []
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
        """执行优化过程 (包含建议的修改)"""
        # 初始化粒子群
        self._initialize_particles()

        # 初始化存档
        self.archive = []

        # 初始评估
        for particle in self.particles:
            # 处理 TP9/TP10 可能返回元组的情况
            evaluation_result = self.problem.evaluate(particle.position)
            if isinstance(evaluation_result, tuple) and len(evaluation_result) == 2:
                objectives = evaluation_result[0]
            else:
                objectives = evaluation_result
            particle.fitness = np.array(objectives)  # 使用 numpy 数组
            particle.best_position = particle.position.copy()  # 初始化 pbest 位置
            particle.best_fitness = particle.fitness.copy()  # 初始化 pbest 适应度

        # 初始化外部存档 (使用初始 pbest)
        if self.use_archive:
            self._update_archive()

        # 优化迭代
        # 使用 tqdm 添加进度条
        pbar = tqdm(range(self.max_iterations), desc=f"Optimizing {self.problem.name} with {self.__class__.__name__}",
                    disable=not verbose)
        # for iteration in range(self.max_iterations): # 旧循环
        for iteration in pbar:  # 新循环带进度条
            # if verbose and iteration % 10 == 0: # 不再需要手动打印进度
            #     print(f"迭代 {iteration}/{self.max_iterations}，当前存档大小: {len(self.archive)}")

            # 更新参数
            progress = iteration / self.max_iterations
            w = self.w_init - (self.w_init - self.w_end) * progress
            c1 = self.c1_init - (self.c1_init - self.c1_end) * progress
            c2 = self.c2_init + (self.c2_end - self.c2_init) * progress

            # 对每个粒子
            for particle in self.particles:
                # 选择领导者
                if self.archive and self.use_archive:
                    leader = self._select_leader(particle)  # _select_leader 内部调用 _crowding_distance_leader
                else:
                    leader = self._select_leader_from_swarm(particle)

                # 如果没有领导者可选 (例如初始时存档为空且无法从种群选)
                if leader is None:
                    leader = particle  # 让粒子飞向自己的历史最优 pbest

                # 更新速度
                particle.update_velocity(leader.best_position, w, c1, c2)

                # --- 应用速度限制 ---
                if self.vmax is not None:  # 检查 vmax 是否已计算
                    particle.velocity = np.clip(particle.velocity, -self.vmax, self.vmax)
                # --- 速度限制结束 ---

                # 更新位置
                particle.update_position()

                # 应用变异
                self._apply_mutation(particle, progress)  # 变异发生在更新位置之后

                # 评估新位置
                evaluation_result = self.problem.evaluate(particle.position)
                if isinstance(evaluation_result, tuple) and len(evaluation_result) == 2:
                    objectives = evaluation_result[0]
                else:
                    objectives = evaluation_result
                particle.fitness = np.array(objectives)  # 更新 fitness

                # --- 修改 pbest 更新逻辑 ---
                # 更新个体最优 (pbest)
                new_fitness = particle.fitness  # 已经是 numpy 数组
                current_pbest_fitness = particle.best_fitness  # 确保 pbest fitness 也是数组

                # 修改: 只要新解不被旧 pbest 支配，就更新 pbest
                if not self._dominates(current_pbest_fitness, new_fitness):
                    particle.best_position = particle.position.copy()
                    particle.best_fitness = new_fitness.copy()
                # --- pbest 更新逻辑修改结束 ---

            # 更新外部存档 (使用更新后的 pbest)
            if self.use_archive:
                self._update_archive()

            # 跟踪性能指标 (每隔一定迭代次数或最后一次记录)
            # if tracking and (iteration % 10 == 0 or iteration == self.max_iterations - 1): # 按频率记录
            if tracking and iteration % 10 == 0:  # 跟踪的频率可以调整
                self._track_performance(iteration)

            # 更新进度条描述信息 (可选)
            if verbose and iteration % 10 == 0:
                pbar.set_postfix({"ArchiveSize": len(self.archive)})

        # 确保记录最后一次迭代的性能
        if tracking:
            self._track_performance(self.max_iterations - 1)

        # 关闭进度条
        pbar.close()

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

            # 特殊初始化前20%的粒子
            if i < self.pop_size // 5:
                # 均匀分布粒子位置以提高多样性
                for j in range(self.problem.n_var):
                    alpha = i / (self.pop_size // 5)
                    particle.position[j] = self.problem.xl[j] + alpha * (self.problem.xu[j] - self.problem.xl[j])

            self.particles.append(particle)

    def _select_leader(self, particle):
        """选择领导者"""
        if not self.archive:
            return None

        # 如果存档太小，随机选择
        if len(self.archive) <= 2:
            return random.choice(self.archive)

        # 使用拥挤度选择
        return self._crowding_distance_leader(particle)

    def _crowding_distance_leader(self, particle):
        """基于拥挤度的领导者选择 (增大锦标赛规模)"""
        if not self.archive: return None  # 增加存档为空的处理
        if len(self.archive) <= 1:
            return self.archive[0]

        # --- 修改: 增大锦标赛选择的候选数量 ---
        # tournament_size = min(3, len(self.archive)) # 原来的设置
        tournament_size = min(7, len(self.archive))  # 尝试选择 7 个候选，或不超过存档大小
        # --- 修改结束 ---

        # 确保不会因 tournament_size <= 0 出错（虽然理论上不会）
        if tournament_size <= 0: return random.choice(self.archive)

        candidates_idx = np.random.choice(len(self.archive), tournament_size, replace=False)
        candidates = [self.archive[i] for i in candidates_idx]

        # 计算拥挤度
        fitnesses = [c.best_fitness for c in candidates]  # 确保使用 best_fitness
        # 检查 fitnesses 是否为空或包含无效值
        if not fitnesses: return random.choice(self.archive)  # 如果候选者没有 fitness，随机选

        # 确保 fitnesses 是有效的 numpy 数组
        try:
            fitnesses_array = np.array(fitnesses)
            if fitnesses_array.ndim != 2 or fitnesses_array.shape[1] != self.problem.n_obj:
                # print(f"警告: 候选者 fitness 形状不正确: {fitnesses_array.shape}")
                # 这里可以选择随机返回一个候选者，避免计算拥挤度出错
                return random.choice(candidates)
        except ValueError as e:
            # print(f"警告: 无法将候选者 fitness 转换为数组: {e}")
            return random.choice(candidates)

        crowding_distances = self._calculate_crowding_distance(fitnesses_array)  # 传递数组

        # 选择拥挤度最大的
        if len(crowding_distances) > 0:
            max_idx_in_candidates = np.argmax(crowding_distances)
            return candidates[max_idx_in_candidates]
        else:
            # 如果拥挤度计算返回空列表，随机选择
            return random.choice(candidates)

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
            if np.random.random() < current_rate:
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
            if not is_dominated and not any(
                    np.array_equal(particle.best_position, a.best_position) for a in self.archive):
                # 深拷贝粒子
                archive_particle = Particle(particle.dimensions, particle.bounds)
                archive_particle.position = particle.best_position.copy()
                archive_particle.best_position = particle.best_position.copy()
                archive_particle.fitness = particle.best_fitness.copy()
                archive_particle.best_fitness = particle.best_fitness.copy()

                self.archive.append(archive_particle)

        # 如果存档超过大小限制，使用拥挤度排序保留多样性
        if len(self.archive) > self.archive_size:
            self._prune_archive()

    def _prune_archive(self):
        """使用拥挤度排序修剪存档"""
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
        """判断fitness1是否支配fitness2（最小化问题）"""
        # 至少一个目标更好，其他不差
        better = False
        for i in range(len(fitness1)):
            if fitness1[i] > fitness2[i]:  # 假设最小化
                return False
            if fitness1[i] < fitness2[i]:
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

    def _get_pareto_set(self):
        """获取算法生成的Pareto解集"""
        if self.use_archive and self.archive:
            return np.array([p.best_position for p in self.archive])
        else:
            # 从粒子群中提取非支配解
            non_dominated = []
            for p in self.particles:
                if not any(self._dominates(other.best_fitness, p.best_fitness) for other in self.particles):
                    non_dominated.append(p.best_position)
            return np.array(non_dominated)

    def _track_performance(self, iteration):
        """跟踪性能指标 - 扩展版本"""
        # 获取当前Pareto前沿和解集
        front = self._get_pareto_front()
        solution_set = self._get_pareto_set() if hasattr(self, '_get_pareto_set') else None

        # 保存迭代次数和前沿
        self.tracking['iterations'].append(iteration)
        self.tracking['fronts'].append(front)

        # 获取真实前沿和解集
        true_front = self.problem.get_pareto_front()
        true_set = self.problem.get_pareto_set()

        # 计算SP指标 (均匀性)
        if len(front) > 1:
            sp = PerformanceIndicators.spacing(front)
            self.tracking['metrics']['sp'].append(sp)
        else:
            self.tracking['metrics']['sp'].append(float('nan'))

        # 有真实前沿时计算IGDF指标
        if true_front is not None and len(front) > 0:
            igdf = PerformanceIndicators.igdf(front, true_front)
            self.tracking['metrics']['igdf'].append(igdf)
        else:
            self.tracking['metrics']['igdf'].append(float('nan'))

        # 有真实解集时计算IGDX指标
        if true_set is not None and solution_set is not None and len(solution_set) > 0:
            igdx = PerformanceIndicators.igdx(solution_set, true_set)
            self.tracking['metrics']['igdx'].append(igdx)
        else:
            self.tracking['metrics']['igdx'].append(float('nan'))

        # 计算RPSP指标
        if true_front is not None and len(front) > 0:
            rpsp = PerformanceIndicators.rpsp(front, true_front)
            self.tracking['metrics']['rpsp'].append(rpsp)
        else:
            self.tracking['metrics']['rpsp'].append(float('nan'))

        # 计算HV指标
        if self.problem.n_obj == 3 and len(front) > 0:
            # 设置参考点
            if true_front is not None:
                ref_point = np.max(true_front, axis=0) * 1.1
            else:
                ref_point = np.max(front, axis=0) * 1.1

            try:
                hv = PerformanceIndicators.hypervolume(front, ref_point)
                self.tracking['metrics']['hv'].append(hv)
            except Exception as e:
                print(f"HV计算错误: {e}")
                self.tracking['metrics']['hv'].append(float('nan'))
        else:
            self.tracking['metrics']['hv'].append(float('nan'))


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

    # --- _track_performance 方法保持不变 ---
    def _track_performance(self, iteration):
        """跟踪性能指标 - 扩展版本"""
        front = self._get_pareto_front()
        solution_set = self._get_pareto_set() if hasattr(self, '_get_pareto_set') else None
        self.tracking['iterations'].append(iteration)
        self.tracking['fronts'].append(front)
        true_front = self.problem.get_pareto_front()
        true_set = self.problem.get_pareto_set()

        # SP
        if len(front) > 1:
            self.tracking['metrics']['sp'].append(PerformanceIndicators.spacing(front))
        else:
            self.tracking['metrics']['sp'].append(float('nan'))
        # IGDF
        if true_front is not None and len(front) > 0:
            self.tracking['metrics']['igdf'].append(PerformanceIndicators.igdf(front, true_front))
        else:
            self.tracking['metrics']['igdf'].append(float('nan'))
        # IGDX
        if true_set is not None and solution_set is not None and len(solution_set) > 0:
            self.tracking['metrics']['igdx'].append(PerformanceIndicators.igdx(solution_set, true_set))
        else:
            self.tracking['metrics']['igdx'].append(float('nan'))
        # RPSP
        if true_front is not None and len(front) > 0:
            self.tracking['metrics']['rpsp'].append(PerformanceIndicators.rpsp(front, true_front))
        else:
            self.tracking['metrics']['rpsp'].append(float('nan'))
        # HV
        if self.problem.n_obj == 3 and len(front) > 0:
            ref_point = np.max(true_front, axis=0) * 1.1 if true_front is not None else np.max(front, axis=0) * 1.1
            try:
                self.tracking['metrics']['hv'].append(PerformanceIndicators.hypervolume(front, ref_point))
            except Exception as e:
                print(f"HV计算错误: {e}"); self.tracking['metrics']['hv'].append(float('nan'))
        else:
            self.tracking['metrics']['hv'].append(float('nan'))


# ====================== 多目标遗传算法======================

class MOEAD:
    """基于分解的多目标进化算法(MOEA/D)"""

    def __init__(self, problem, pop_size=300, max_generations=300, T=20, delta=0.9, nr=2):
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
                'igdf': [],
                'igdx': [],
                'rpsp': [],
                'hv': [],
                'sp': []
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
        for gen in range(self.max_generations):
            if verbose and gen % 10 == 0:
                print(f"迭代 {gen}/{self.max_generations}")

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
                child_obj = np.array(self.problem.evaluate(child))

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
        """初始化权重向量，确保生成足够数量的向量"""
        if self.problem.n_obj == 3:
            # 三目标问题使用改进的方法生成权重
            self.weights = self._generate_uniform_weights(self.problem.n_obj, self.pop_size)
        else:
            # 其他维度使用随机权重
            self.weights = np.random.random((self.pop_size, self.problem.n_obj))
            # 归一化
            self.weights = self.weights / np.sum(self.weights, axis=1)[:, np.newaxis]

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
        """跟踪性能指标 - 扩展版本"""
        # 获取当前Pareto前沿和解集
        front = self._get_pareto_front()
        solution_set = self._get_pareto_set() if hasattr(self, '_get_pareto_set') else None

        # 保存迭代次数和前沿
        self.tracking['iterations'].append(iteration)
        self.tracking['fronts'].append(front)

        # 获取真实前沿和解集
        true_front = self.problem.get_pareto_front()
        true_set = self.problem.get_pareto_set()

        # 计算SP指标 (均匀性)
        if len(front) > 1:
            sp = PerformanceIndicators.spacing(front)
            self.tracking['metrics']['sp'].append(sp)
        else:
            self.tracking['metrics']['sp'].append(float('nan'))

        # 有真实前沿时计算IGDF指标
        if true_front is not None and len(front) > 0:
            igdf = PerformanceIndicators.igdf(front, true_front)
            self.tracking['metrics']['igdf'].append(igdf)
        else:
            self.tracking['metrics']['igdf'].append(float('nan'))

        # 有真实解集时计算IGDX指标
        if true_set is not None and solution_set is not None and len(solution_set) > 0:
            igdx = PerformanceIndicators.igdx(solution_set, true_set)
            self.tracking['metrics']['igdx'].append(igdx)
        else:
            self.tracking['metrics']['igdx'].append(float('nan'))

        # 计算RPSP指标
        if true_front is not None and len(front) > 0:
            rpsp = PerformanceIndicators.rpsp(front, true_front)
            self.tracking['metrics']['rpsp'].append(rpsp)
        else:
            self.tracking['metrics']['rpsp'].append(float('nan'))

        # 计算HV指标
        if self.problem.n_obj == 3 and len(front) > 0:
            # 设置参考点
            if true_front is not None:
                ref_point = np.max(true_front, axis=0) * 1.1
            else:
                ref_point = np.max(front, axis=0) * 1.1

            try:
                hv = PerformanceIndicators.hypervolume(front, ref_point)
                self.tracking['metrics']['hv'].append(hv)
            except Exception as e:
                print(f"HV计算错误: {e}")
                self.tracking['metrics']['hv'].append(float('nan'))
        else:
            self.tracking['metrics']['hv'].append(float('nan'))


class NSGAII:
    """NSGA-II算法实现"""

    def __init__(self, problem, pop_size=100, max_generations=100,
                 pc=0.9,          # 交叉概率 (Crossover probability)
                 eta_c=20,        # SBX 交叉分布指数 (Distribution index for SBX)
                 pm_ratio=1.0,    # 变异概率因子 (pm = pm_ratio / n_var)
                 eta_m=20):       # 多项式变异分布指数 (Distribution index for polynomial mutation)
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
        if not front: # 检查 front 是否为空
            return

        n = len(front)
        for p in front:
            p['crowding_distance'] = 0.0 # 确保初始化为浮点数

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
                epsilon = 1e-9 # 一个很小的值
                norm = (f_max - f_min) + epsilon # 加上 epsilon 避免严格为0
                # --- 修改结束 ---

                for i in range(1, n - 1):
                    # 使用原始 front 列表中的索引来更新距离
                    prev_idx = sorted_indices[i-1]
                    next_idx = sorted_indices[i+1]
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
        random.shuffle(parent_indices) # 打乱父代顺序

        for i in range(0, self.pop_size, 2):
            # 选择父代索引，处理最后一个父代可能落单的情况
            idx1 = parent_indices[i]
            idx2 = parent_indices[i + 1] if (i + 1) < len(parents) else parent_indices[0] # 落单则与第一个配对

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
                    if random.random() < 0.5: # 对每个变量 50% 概率交叉
                        y1, y2 = p1['x'][j], p2['x'][j]
                        if abs(y1 - y2) > 1e-10:
                            if y1 > y2: y1, y2 = y2, y1 # 确保 y1 <= y2

                            rand = random.random()
                            beta = 1.0 + (2.0 * (y1 - xl[j]) / (y2 - y1)) if (y2-y1)>1e-10 else 1.0
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
                    if random.random() < self.pm: # 使用 self.pm
                        y = child['x'][j]
                        delta1 = (y - xl[j]) / (xu[j] - xl[j]) if (xu[j]-xl[j])>1e-10 else 0.5
                        delta2 = (xu[j] - y) / (xu[j] - xl[j]) if (xu[j]-xl[j])>1e-10 else 0.5
                        delta1 = np.clip(delta1, 0, 1) # 确保在[0,1]
                        delta2 = np.clip(delta2, 0, 1) # 确保在[0,1]

                        rand = random.random()
                        mut_pow = 1.0 / (self.eta_m + 1.0) # 使用 self.eta_m

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
                        child['x'][j] = np.clip(y, xl[j], xu[j]) # 边界处理

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

        return offspring[:self.pop_size] # 返回精确 pop_size 个子代

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

    def _track_performance(self, iteration):
        """跟踪性能指标 - 扩展版本"""
        # 获取当前Pareto前沿和解集
        front = self._get_pareto_front()
        solution_set = self._get_pareto_set() if hasattr(self, '_get_pareto_set') else None

        # 保存迭代次数和前沿
        self.tracking['iterations'].append(iteration)
        self.tracking['fronts'].append(front)

        # 获取真实前沿和解集
        true_front = self.problem.get_pareto_front()
        true_set = self.problem.get_pareto_set()

        # 计算SP指标 (均匀性)
        if len(front) > 1:
            sp = PerformanceIndicators.spacing(front)
            self.tracking['metrics']['sp'].append(sp)
        else:
            self.tracking['metrics']['sp'].append(float('nan'))

        # 有真实前沿时计算IGDF指标
        if true_front is not None and len(front) > 0:
            igdf = PerformanceIndicators.igdf(front, true_front)
            self.tracking['metrics']['igdf'].append(igdf)
        else:
            self.tracking['metrics']['igdf'].append(float('nan'))

        # 有真实解集时计算IGDX指标
        if true_set is not None and solution_set is not None and len(solution_set) > 0:
            igdx = PerformanceIndicators.igdx(solution_set, true_set)
            self.tracking['metrics']['igdx'].append(igdx)
        else:
            self.tracking['metrics']['igdx'].append(float('nan'))

        # 计算RPSP指标
        if true_front is not None and len(front) > 0:
            rpsp = PerformanceIndicators.rpsp(front, true_front)
            self.tracking['metrics']['rpsp'].append(rpsp)
        else:
            self.tracking['metrics']['rpsp'].append(float('nan'))

        # 计算HV指标
        if self.problem.n_obj == 3 and len(front) > 0:
            # 设置参考点
            if true_front is not None:
                ref_point = np.max(true_front, axis=0) * 1.1
            else:
                ref_point = np.max(front, axis=0) * 1.1

            try:
                hv = PerformanceIndicators.hypervolume(front, ref_point)
                self.tracking['metrics']['hv'].append(hv)
            except Exception as e:
                print(f"HV计算错误: {e}")
                self.tracking['metrics']['hv'].append(float('nan'))
        else:
            self.tracking['metrics']['hv'].append(float('nan'))


class SPEA2:
    """强度Pareto进化算法2"""

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
                'igdf': [],
                'igdx': [],
                'rpsp': [],
                'hv': [],
                'sp': []
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
            p_obj = np.array(p['objectives'])  # 确保是numpy数组

            for j, q in enumerate(combined_pop):
                if i != j:
                    q_obj = np.array(q['objectives'])  # 确保是numpy数组
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
                child_obj = np.array(self.problem.evaluate(child_x))  # 确保是numpy数组

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
                objectives = np.array(self.problem.evaluate(x))  # 确保是numpy数组
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
        """跟踪性能指标 - 扩展版本"""
        # 获取当前Pareto前沿和解集
        front = self._get_pareto_front()
        solution_set = self._get_pareto_set() if hasattr(self, '_get_pareto_set') else None

        # 保存迭代次数和前沿
        self.tracking['iterations'].append(iteration)
        self.tracking['fronts'].append(front)

        # 获取真实前沿和解集
        true_front = self.problem.get_pareto_front()
        true_set = self.problem.get_pareto_set()

        # 计算SP指标 (均匀性)
        if len(front) > 1:
            sp = PerformanceIndicators.spacing(front)
            self.tracking['metrics']['sp'].append(sp)
        else:
            self.tracking['metrics']['sp'].append(float('nan'))

        # 有真实前沿时计算IGDF指标
        if true_front is not None and len(front) > 0:
            igdf = PerformanceIndicators.igdf(front, true_front)
            self.tracking['metrics']['igdf'].append(igdf)
        else:
            self.tracking['metrics']['igdf'].append(float('nan'))

        # 有真实解集时计算IGDX指标
        if true_set is not None and solution_set is not None and len(solution_set) > 0:
            igdx = PerformanceIndicators.igdx(solution_set, true_set)
            self.tracking['metrics']['igdx'].append(igdx)
        else:
            self.tracking['metrics']['igdx'].append(float('nan'))

        # 计算RPSP指标
        if true_front is not None and len(front) > 0:
            rpsp = PerformanceIndicators.rpsp(front, true_front)
            self.tracking['metrics']['rpsp'].append(rpsp)
        else:
            self.tracking['metrics']['rpsp'].append(float('nan'))

        # 计算HV指标
        if self.problem.n_obj == 3 and len(front) > 0:
            # 设置参考点
            if true_front is not None:
                ref_point = np.max(true_front, axis=0) * 1.1
            else:
                ref_point = np.max(front, axis=0) * 1.1

            try:
                hv = PerformanceIndicators.hypervolume(front, ref_point)
                self.tracking['metrics']['hv'].append(hv)
            except Exception as e:
                print(f"HV计算错误: {e}")
                self.tracking['metrics']['hv'].append(float('nan'))
        else:
            self.tracking['metrics']['hv'].append(float('nan'))


# ====================== 性能评估指标 ======================

class PerformanceIndicators:
    """性能评估指标类，包含各种常用指标的计算方法"""

    @staticmethod
    def spacing(front):
        """计算Pareto前沿的均匀性指标SP"""
        if len(front) < 2:
            return float('nan')  # 改为返回NaN而不是0

        try:
            # 计算每对解之间的欧几里得距离
            distances = []
            for i in range(len(front)):
                min_dist = float('inf')
                for j in range(len(front)):
                    if i != j:
                        # 确保正确的数组转换
                        dist = np.sqrt(np.sum((front[i] - front[j]) ** 2))
                        min_dist = min(min_dist, dist)
                distances.append(min_dist)

            # 计算平均距离
            d_mean = np.mean(distances)

            # 计算标准差
            sp = np.sqrt(np.sum((distances - d_mean) ** 2) / len(distances))

            return sp
        except Exception as e:
            print(f"SP计算错误: {e}")
            return float('nan')

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
        这是一个简化版本，仅适用于三目标问题
        """
        # 对于高维问题，应使用更高效的算法
        if len(front) == 0:
            return 0

        # 检查并确保前沿和参考点的维度匹配
        if front.shape[1] != len(reference_point):
            print(f"警告: 前沿维度({front.shape[1]})与参考点维度({len(reference_point)})不匹配")
            return 0

        try:
            from pymoo.indicators.hv import HV
            return HV(ref_point=reference_point).do(front)
        except ImportError:
            # 简化方法 - 适用于三维问题的蒙特卡洛方法
            # 生成随机点并检查是否被前沿支配
            n_samples = 10000
            dominated_count = 0

            for _ in range(n_samples):
                # 生成参考点和前沿之间的随机点
                point = np.array([np.random.uniform(min_val, reference_point[i])
                                  for i, min_val in enumerate(np.min(front, axis=0))])

                # 检查是否被任何前沿点支配
                dominated = False
                for sol in front:
                    if np.all(sol <= point):
                        dominated = True
                        break

                if dominated:
                    dominated_count += 1

            # 计算超体积
            volume = np.prod(reference_point - np.min(front, axis=0))
            return (dominated_count / n_samples) * volume

    @staticmethod
    def igdf(approximation_front, true_front):
        """
        计算前沿空间中的IGD (IGDF)
        从真实Pareto前沿到近似前沿的平均距离
        """
        if len(approximation_front) == 0 or len(true_front) == 0:
            return float('nan')

        try:
            # 计算每个真实前沿点到近似前沿的最小距离
            distances = cdist(true_front, approximation_front, 'euclidean')
            min_distances = np.min(distances, axis=1)

            # 返回平均距离
            return np.mean(min_distances)
        except Exception as e:
            print(f"IGDF计算错误: {e}")
            return float('nan')

    @staticmethod
    def igdx(approximation_set, true_set):
        """
        计算决策变量空间中的IGD (IGDX)
        从真实Pareto解集到近似解集的平均距离
        """
        if approximation_set is None or true_set is None:
            return float('nan')
        if len(approximation_set) == 0 or len(true_set) == 0:
            return float('nan')

        try:
            # 计算每个真实解集点到近似解集的最小距离
            distances = cdist(true_set, approximation_set, 'euclidean')
            min_distances = np.min(distances, axis=1)

            # 返回平均距离
            return np.mean(min_distances)
        except Exception as e:
            print(f"IGDX计算错误: {e}")
            return float('nan')

    @staticmethod
    def rpsp(front, reference_front, r=0.1):
        """
        计算r-PSP (Radial-based Pareto Set Proximity)
        r: 径向扩展参数(默认0.1)
        """
        if len(front) < 2 or len(reference_front) < 2:
            return float('nan')

        try:
            # 1. 标准化前沿
            front_min = np.min(front, axis=0)
            front_max = np.max(front, axis=0)
            front_range = front_max - front_min

            # 避免除零
            front_range[front_range < 1e-10] = 1

            norm_front = (front - front_min) / front_range
            norm_ref = (reference_front - front_min) / front_range

            # 2. 计算径向距离
            n_obj = front.shape[1]
            rpsp_sum = 0
            count = 0

            for ref_point in norm_ref:
                # 找最近的近似前沿点
                min_dist = float('inf')
                for approx_point in norm_front:
                    # 径向距离计算
                    diff_vector = approx_point - ref_point
                    angle_penalty = np.linalg.norm(diff_vector) * r
                    dist = np.linalg.norm(diff_vector) + angle_penalty
                    min_dist = min(min_dist, dist)

                rpsp_sum += min_dist
                count += 1

            return rpsp_sum / count if count > 0 else float('nan')
        except Exception as e:
            print(f"RPSP计算错误: {e}")
            return float('nan')


# ====================== 可视化功能 ======================

class Visualizer:
    """可视化工具类，用于绘制Pareto前沿、解集和性能指标"""

    @staticmethod
    def plot_pareto_front_comparison(problem, algorithms_results, save_path=None, plot_true_front=True):
        """
        比较不同算法的Pareto前沿，改进的视角设置

        problem: 测试问题实例
        algorithms_results: 字典，键为算法名称，值为算法结果
        save_path: 保存图像的路径
        plot_true_front: 是否绘制真实Pareto前沿
        """
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        # 绘制算法结果
        markers = ['o', 's', '^', 'D', 'p', '*']
        colors = plt.cm.viridis(np.linspace(0, 1, len(algorithms_results)))

        # 记录所有点的坐标范围，用于设置轴限制
        all_points = []

        for (algo_name, result), marker, color in zip(algorithms_results.items(), markers, colors):
            pareto_front = result["pareto_front"]
            ax.scatter(pareto_front[:, 0], pareto_front[:, 1], pareto_front[:, 2],
                       marker=marker, s=30, color=color, label=f"{algo_name}")
            all_points.append(pareto_front)

        # 绘制真实Pareto前沿
        if plot_true_front and problem.pareto_front is not None:
            ax.scatter(problem.pareto_front[:, 0], problem.pareto_front[:, 1], problem.pareto_front[:, 2],
                       marker='+', s=10, color='red', alpha=0.5, label='True PF')
            all_points.append(problem.pareto_front)

        # 设置图例和标签
        ax.set_xlabel('$f_1$', labelpad=10)
        ax.set_ylabel('$f_2$', labelpad=10)
        ax.set_zlabel('$f_3$', labelpad=10)
        ax.set_title(f'Pareto front for {problem.name}')

        # 计算所有点的坐标范围
        all_points = np.vstack(all_points)
        min_vals = np.min(all_points, axis=0)
        max_vals = np.max(all_points, axis=0)

        # 添加一些边距
        padding = (max_vals - min_vals) * 0.1

        # 设置轴限制，确保完整显示
        ax.set_xlim(min_vals[0] - padding[0], max_vals[0] + padding[0])
        ax.set_ylim(min_vals[1] - padding[1], max_vals[1] + padding[1])
        ax.set_zlim(min_vals[2] - padding[2], max_vals[2] + padding[2])

        # 调整图例位置和大小
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=3)

        # 设置更好的视角
        ax.view_init(elev=30, azim=45)

        # 设置网格线
        ax.grid(True, linestyle='--', alpha=0.3)

        # 保存或显示图像
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig, ax

    @staticmethod
    def plot_pareto_set_comparison(problem, algorithms_results, save_path=None, plot_true_set=True):
        """
        比较不同算法的Pareto解集

        problem: 测试问题实例
        algorithms_results: 字典，键为算法名称，值为算法结果
        save_path: 保存图像的路径
        plot_true_set: 是否绘制真实Pareto解集
        """
        # 检查问题是否为三维变量
        if problem.n_var < 3:
            print("警告: 问题变量维度小于3，无法绘制3D解集")
            return None, None

        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        # 绘制算法结果
        markers = ['o', 's', '^', 'D', 'p', '*']
        colors = plt.cm.viridis(np.linspace(0, 1, len(algorithms_results)))

        # 记录所有点的坐标范围
        all_points = []

        for (algo_name, result), marker, color in zip(algorithms_results.items(), markers, colors):
            if "pareto_set" in result:
                pareto_set = result["pareto_set"]
                # 只绘制前三个维度
                ax.scatter(pareto_set[:, 0], pareto_set[:, 1], pareto_set[:, 2],
                           marker=marker, s=30, color=color, label=f"{algo_name} (PS)")
                all_points.append(pareto_set[:, :3])

        # 绘制真实Pareto解集
        if plot_true_set and problem.pareto_set is not None:
            ax.scatter(problem.pareto_set[:, 0], problem.pareto_set[:, 1], problem.pareto_set[:, 2],
                       marker='+', s=10, color='red', alpha=0.5, label='True PS')
            all_points.append(problem.pareto_set[:, :3])

        # 设置图例和标签
        ax.set_xlabel('$x_1$', labelpad=10)
        ax.set_ylabel('$x_2$', labelpad=10)
        ax.set_zlabel('$x_3$', labelpad=10)
        ax.set_title(f'Pareto set for {problem.name}')

        # 确保完整显示坐标轴
        if all_points:
            all_points = np.vstack(all_points)
            min_vals = np.min(all_points, axis=0)
            max_vals = np.max(all_points, axis=0)

            # 添加边距
            padding = (max_vals - min_vals) * 0.1

            # 设置轴限制
            ax.set_xlim(min_vals[0] - padding[0], max_vals[0] + padding[0])
            ax.set_ylim(min_vals[1] - padding[1], max_vals[1] + padding[1])
            ax.set_zlim(min_vals[2] - padding[2], max_vals[2] + padding[2])

        # 调整图例位置和大小
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=3)

        # 设置视角
        ax.view_init(elev=30, azim=45)

        # 设置网格线
        ax.grid(True, linestyle='--', alpha=0.3)

        # 保存或显示图像
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig, ax

    @staticmethod
    def plot_algorithm_performance_boxplots(algorithms_results, problem_name,
                                            metrics=["igdf", "igdx", "rpsp", "hv", "sp"], save_path=None):
        """改进的箱线图函数，支持新指标，改为纵向排布"""
        n_metrics = len(metrics)
        # --- 修改这里 ---
        # 改为 n_metrics 行, 1 列
        fig, axes = plt.subplots(n_metrics, 1, figsize=(8, 6 * n_metrics))
        # --- 修改结束 ---

        if n_metrics == 1:
            # 如果只有一个指标，确保 axes 是一个列表或可迭代对象
            axes = [axes]

        metric_labels = {
            "igdf": "IGDF",
            "igdx": "IGDX",
            "rpsp": "RPSP",
            "hv": "HV",
            "sp": "SP",
            "igd": "IGD"
        }

        # 调试输出
        print(f"Creating performance boxplots for {problem_name}...")
        for metric in metrics:
            print(f"  Checking {metric} data:")
            for algo_name, result in algorithms_results.items():
                # 检查指标是否在tracking结构中
                if "tracking" in result and "metrics" in result["tracking"] and metric in result["tracking"]["metrics"]:
                    values = [v for v in result["tracking"]["metrics"][metric] if not np.isnan(v)]
                    print(f"    {algo_name}: {len(values)} valid data points")
                else:
                    print(f"    {algo_name}: No data for {metric}")

        for i, metric in enumerate(metrics):
            # 收集所有算法的指标值
            data = []
            labels = []

            for algo_name, result in algorithms_results.items():
                # 从tracking结构中获取数据（实际数据存储位置）
                if "tracking" in result and "metrics" in result["tracking"] and metric in result["tracking"]["metrics"]:
                    metric_values = result["tracking"]["metrics"][metric]
                    values = [v for v in metric_values if not np.isnan(v)]

                    if values:
                        data.append(values)
                        labels.append(algo_name)

            if data:
                # 检查数据是否合理，特别是SP值往往很小
                if metric == "sp":
                    print(f"  SP data ranges: {[min(d) for d in data]} to {[max(d) for d in data]}")
                    # 对于异常小的值，可以进行数据转换以便更好地可视化
                    if any(min(d) < 0.001 for d in data):
                        print("  Warning: Very small SP values detected, consider log transformation")

                try:
                    # 创建小提琴图和箱线图
                    if len(data) > 0:
                        # --- 确保使用正确的 axes 对象 ---
                        ax = axes[i]  # 获取当前子图
                        # --- 修改结束 ---

                        violin_parts = ax.violinplot(data, showmeans=False, showmedians=True)

                        # 设置颜色
                        colors = plt.cm.viridis(np.linspace(0, 1, len(data)))
                        for j, pc in enumerate(violin_parts['bodies']):
                            pc.set_facecolor(colors[j])
                            pc.set_edgecolor('black')
                            pc.set_alpha(0.7)

                        # 添加boxplot
                        bp = ax.boxplot(data, positions=range(1, len(data) + 1),
                                        widths=0.15, patch_artist=True,
                                        showfliers=True, showmeans=True, meanline=True)

                        # 自定义boxplot颜色
                        for j, box in enumerate(bp['boxes']):
                            box.set(facecolor='white', alpha=0.5)

                        # 设置标签和标题
                        ax.set_xticks(range(1, len(labels) + 1))
                        ax.set_xticklabels(labels, rotation=45, ha='right')
                        ax.set_ylabel(metric_labels.get(metric, metric.upper()))
                        ax.set_title(f'{metric_labels.get(metric, metric.upper())} Performance')
                        ax.grid(True, linestyle='--', alpha=0.3, axis='y')

                        # 设置更好的y轴范围
                        if metric in ["igdf", "igdx", "rpsp", "sp", "igd"]:  # 较小值更好
                            if min([min(d) for d in data]) < 0.1:
                                ax.set_ylim(bottom=0)  # 从0开始

                        # 优化布局
                        ax.spines['top'].set_visible(False)
                        ax.spines['right'].set_visible(False)
                    else:
                        # --- 确保使用正确的 axes 对象 ---
                        ax = axes[i]
                        # --- 修改结束 ---
                        ax.text(0.5, 0.5, f"No valid {metric.upper()} data",
                                ha='center', va='center', fontsize=12)
                        ax.set_axis_off()
                except Exception as e:
                    # --- 确保使用正确的 axes 对象 ---
                    ax = axes[i]
                    # --- 修改结束 ---
                    print(f"Error plotting {metric}: {e}")
                    ax.text(0.5, 0.5, f"Error plotting {metric} data",
                            ha='center', va='center', fontsize=12)
                    ax.set_axis_off()
            else:
                # --- 确保使用正确的 axes 对象 ---
                ax = axes[i]
                # --- 修改结束 ---
                ax.text(0.5, 0.5, f"No valid {metric.upper()} data",
                        ha='center', va='center', fontsize=12)
                ax.set_axis_off()

        # 设置总标题
        plt.suptitle(f'Algorithm Performance on {problem_name}', fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])  # 调整布局以适应总标题和X轴标签

        # 保存图像
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    @staticmethod
    def plot_convergence(algorithms_results, metric_name="igdf", problem_name="", save_path=None):
        """
        绘制收敛曲线，支持多种性能指标

        algorithms_results: 字典，键为算法名称，值为算法结果
        metric_name: 指标名称（igdf, igdx, rpsp, hv, sp）
        problem_name: 问题名称
        save_path: 保存图像的路径
        """
        plt.figure(figsize=(10, 6))

        # 调试信息
        print(f"绘制{problem_name}问题的{metric_name.upper()}收敛曲线...")
        data_plotted = False

        for algo_name, result in algorithms_results.items():
            if "tracking" in result and "metrics" in result["tracking"]:
                iterations = result["tracking"]["iterations"]
                metric_values = result["tracking"]["metrics"].get(metric_name, [])

                # 检查数据有效性
                valid_values = [v for v in metric_values if not np.isnan(v)]
                print(f"  {algo_name}: 共{len(metric_values)}个值，其中{len(valid_values)}个有效值")

                if iterations and valid_values:
                    # 处理NaN值 - 用前一个有效值填充
                    clean_values = []
                    last_valid = None
                    for v in metric_values:
                        if not np.isnan(v):
                            clean_values.append(v)
                            last_valid = v
                        elif last_valid is not None:
                            clean_values.append(last_valid)
                        else:
                            clean_values.append(0)  # 初始无效值填充为0

                    plt.plot(iterations, clean_values, '-o', label=algo_name, markersize=4)
                    data_plotted = True
                else:
                    print(f"  {algo_name}: 无足够有效数据，跳过绘图")

        # 设置标题和标签
        metric_labels = {
            "igdf": "IGDF (Inverted Generational Distance in F-space)",
            "igdx": "IGDX (Inverted Generational Distance in X-space)",
            "rpsp": "RPSP (r-Pareto Set Proximity)",
            "hv": "HV (Hypervolume)",
            "sp": "SP (Spacing)"
        }

        metric_label = {
            "igdf": "IGDF",
            "igdx": "IGDX",
            "rpsp": "RPSP",
            "hv": "HV",
            "sp": "SP"
        }

        plt.xlabel('Iterations', fontsize=12)
        plt.ylabel(metric_labels.get(metric_name, metric_name.upper()), fontsize=12)
        plt.title(f'{metric_label.get(metric_name, metric_name.upper())} For {problem_name}', fontsize=14)

        # 设置网格和图例
        plt.grid(True, linestyle='--', alpha=0.7)

        # Y轴范围调整 - 对于不同指标采用不同策略
        if data_plotted:
            if metric_name in ["igdf", "igdx", "rpsp", "sp", "igd"]:
                # 这些指标越小越好，从0开始显示
                ymin, ymax = plt.ylim()
                plt.ylim(0, ymax)

            plt.legend(loc='best', fontsize=10)

            # 保存图像
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"  图像已保存到: {save_path}")
        else:
            # 没有有效数据时显示提示信息
            plt.text(0.5, 0.5, f"NO {metric_name.upper()} Value",
                     ha='center', va='center', fontsize=14, transform=plt.gca().transAxes)
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"  空图像已保存到: {save_path}")

        return plt.gcf()

    @staticmethod
    def plot_radar_chart(algorithms_results, problem_name, metrics=["igdf", "igdx", "rpsp", "hv", "sp"],
                         save_path=None):
        """
        绘制雷达图(蜘蛛网图)展示算法在多个指标上的性能对比

        algorithms_results: 字典，键为算法名称，值为算法结果
        problem_name: 问题名称
        metrics: 要比较的指标列表
        save_path: 保存图像的路径
        """
        # 计算每个算法在每个指标上的标准化得分
        algo_names = list(algorithms_results.keys())
        n_metrics = len(metrics)

        # 提取各指标数据
        data = {}
        for metric in metrics:
            metric_values = {}
            for algo_name in algo_names:
                if f"avg_{metric}" in algorithms_results[algo_name].get(problem_name, {}).get("metrics", {}):
                    value = algorithms_results[algo_name][problem_name]["metrics"][f"avg_{metric}"]
                    if not np.isnan(value):
                        metric_values[algo_name] = value

            if metric_values:
                data[metric] = metric_values

        # 标准化分数 - 针对不同指标使用不同规则
        scores = {algo: [0] * n_metrics for algo in algo_names}

        for i, metric in enumerate(metrics):
            if metric not in data or not data[metric]:
                continue

            values = data[metric]
            # 对于IGD*和SP，较小值更好；对于HV，较大值更好
            if metric in ["igdf", "igdx", "rpsp", "sp", "igd"]:
                best = min(values.values())
                worst = max(values.values())
                # 标准化公式: (worst - value) / (worst - best)
                diff = worst - best
                if diff > 0:
                    for algo in values:
                        scores[algo][i] = (worst - values[algo]) / diff
            else:  # hv
                best = max(values.values())
                worst = min(values.values())
                # 标准化公式: (value - worst) / (best - worst)
                diff = best - worst
                if diff > 0:
                    for algo in values:
                        scores[algo][i] = (values[algo] - worst) / diff

        # 绘制雷达图
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, polar=True)

        # 设置角度
        angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
        angles += angles[:1]  # 闭合图形

        # 设置标签位置
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([metric.upper() for metric in metrics])

        # 绘制每个算法的得分
        colors = plt.cm.tab10(np.linspace(0, 1, len(algo_names)))

        for i, algo in enumerate(algo_names):
            if algo not in scores:
                continue

            values = scores[algo]
            values += values[:1]  # 闭合图形
            ax.plot(angles, values, 'o-', linewidth=2, color=colors[i], label=algo)
            ax.fill(angles, values, alpha=0.1, color=colors[i])

        # 美化图表
        ax.set_ylim(0, 1.05)
        ax.set_title(f'Performance Comparison on {problem_name}', fontsize=14, pad=20)
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))

        # 保存图像
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    @staticmethod
    def plot_metrics_summary(results, problems, metrics=["igdf", "igdx", "rpsp", "hv", "sp"], save_path=None):
        """
        绘制各算法在所有问题上的指标汇总热图

        results: 结果字典
        problems: 问题列表
        metrics: 要比较的指标列表
        save_path: 保存图像的路径
        """
        algo_names = list(results.keys())
        problem_names = [p.name for p in problems]
        n_algos = len(algo_names)
        n_problems = len(problem_names)

        # 绘制每个指标的热图
        for metric in metrics:
            fig, ax = plt.subplots(figsize=(max(10, n_problems * 0.8), max(8, n_algos * 0.6)))

            # 准备数据
            data = np.zeros((n_algos, n_problems))
            data.fill(np.nan)  # 初始化为NaN

            # 提取数据
            for i, algo in enumerate(algo_names):
                for j, prob in enumerate(problem_names):
                    if prob in results[algo] and f"avg_{metric}" in results[algo][prob]["metrics"]:
                        data[i, j] = results[algo][prob]["metrics"][f"avg_{metric}"]

            # 处理NaN值
            mask = np.isnan(data)

            # 根据指标类型调整颜色映射
            if metric in ["igdf", "igdx", "rpsp", "sp", "igd"]:
                cmap = "YlOrRd_r"  # 逆序：较低的值(更好)显示为较浅的颜色
            else:  # hv
                cmap = "YlOrRd"  # 正序：较高的值(更好)显示为较深的颜色

            # 绘制热图
            im = ax.imshow(data, cmap=cmap, aspect='auto')

            # 添加数值标签
            for i in range(n_algos):
                for j in range(n_problems):
                    if not mask[i, j]:
                        # 根据值的大小调整颜色
                        val = data[i, j]
                        if metric in ["igdf", "igdx", "rpsp", "sp", "igd"]:
                            is_best_in_col = val == np.nanmin(data[:, j])
                        else:  # hv
                            is_best_in_col = val == np.nanmax(data[:, j])

                        # 最优值用粗体标记
                        if is_best_in_col:
                            ax.text(j, i, f"{val:.2e}", ha="center", va="center",
                                    color="black", fontweight='bold')
                        else:
                            ax.text(j, i, f"{val:.2e}", ha="center", va="center",
                                    color="black")

            # 设置坐标轴
            ax.set_xticks(np.arange(n_problems))
            ax.set_yticks(np.arange(n_algos))
            ax.set_xticklabels(problem_names)
            ax.set_yticklabels(algo_names)

            # 旋转x轴标签以防重叠
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

            # 添加标题和颜色条
            metric_titles = {
                "igdf": "IGDF (Convergence in objective space)",
                "igdx": "IGDX (Convergence in decision space)",
                "rpsp": "RPSP (r-Radial set proximity)",
                "hv": "HV (Hypervolume)",
                "sp": "SP (Distribution uniformity)"
            }

            plt.colorbar(im, ax=ax, label=f"{metric.upper()} Value")
            ax.set_title(f"{metric_titles.get(metric, metric.upper())} Comparison")

            fig.tight_layout()

            # 保存图像
            if save_path:
                metric_save_path = save_path.replace(".png", f"_{metric}.png")
                plt.savefig(metric_save_path, dpi=300, bbox_inches='tight')

        return fig

    @staticmethod
    def plot_performance_comparison(algorithms_results, problems, metrics=["igdf", "igdx", "rpsp", "hv", "sp"],
                                    save_path=None):
        """
        比较不同算法在多个问题上的性能

        algorithms_results: 字典，键为算法名称，值为算法结果
        problems: 问题列表或名称列表
        metrics: 要比较的指标列表
        save_path: 保存图像的路径
        """
        n_metrics = len(metrics)
        n_problems = len(problems)

        fig, axes = plt.subplots(n_metrics, 1, figsize=(12, 5 * n_metrics))
        if n_metrics == 1:
            axes = [axes]

        colors = plt.cm.viridis(np.linspace(0, 1, len(algorithms_results)))

        for i, metric in enumerate(metrics):
            # 提取每个算法在每个问题上的性能数据
            problem_names = [p.name if hasattr(p, 'name') else str(p) for p in problems]
            data = {algo_name: [] for algo_name in algorithms_results.keys()}

            for j, problem in enumerate(problem_names):
                for algo_name in algorithms_results.keys():
                    if problem in algorithms_results[algo_name]:
                        result = algorithms_results[algo_name][problem]
                        # 获取最后一个度量值作为最终性能
                        if "tracking" in result and "metrics" in result["tracking"]:
                            metric_values = result["tracking"]["metrics"].get(metric, [])
                            if metric_values:
                                data[algo_name].append(metric_values[-1])
                            else:
                                data[algo_name].append(float('nan'))
                        else:
                            data[algo_name].append(float('nan'))
                    else:
                        data[algo_name].append(float('nan'))

            # 绘制条形图
            bar_width = 0.8 / len(algorithms_results)
            for k, (algo_name, values) in enumerate(data.items()):
                x = np.arange(n_problems) + k * bar_width
                axes[i].bar(x, values, width=bar_width, label=algo_name, color=colors[k])

            # 设置标签和标题
            axes[i].set_xlabel('问题')
            axes[i].set_ylabel(metric.upper())

            metric_titles = {
                "igdf": "IGDF (目标空间收敛性)",
                "igdx": "IGDX (决策空间收敛性)",
                "rpsp": "RPSP (径向集逼近)",
                "hv": "HV (超体积)",
                "sp": "SP (均匀性)",
                "igd": "IGD (倒代距离)"
            }

            axes[i].set_title(f'{metric_titles.get(metric, metric.upper())} 性能比较')
            axes[i].set_xticks(np.arange(n_problems) + (len(algorithms_results) - 1) * bar_width / 2)
            axes[i].set_xticklabels(problem_names, rotation=45)
            axes[i].grid(True, linestyle='--', alpha=0.3, axis='y')

            if i == 0:
                axes[i].legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=len(algorithms_results))

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig


# ====================== 实验框架 ======================

class ExperimentFramework:
    """实验框架类，用于运行和比较不同算法"""

    def __init__(self, save_dir="results"):
        """
        初始化实验框架

        save_dir: 保存结果的目录
        """
        self.save_dir = save_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # 配置日志
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            filename=os.path.join(save_dir, 'experiment.log')
        )
        self.logger = logging.getLogger("ExperimentFramework")

    def run_experiment(self, problems, algorithms, algorithm_params, n_runs=10, verbose=True):
        """
        运行实验，并生成全面的性能指标可视化

        参数:
        problems: 测试问题列表
        algorithms: 算法类列表
        algorithm_params: 算法参数字典
        n_runs: 每个算法和问题的运行次数
        verbose: 是否输出进度信息

        返回: 结果字典
        """
        # 创建结果字典
        results = {algo.__name__: {} for algo in algorithms}

        for problem in problems:
            problem_name = problem.name

            if verbose:
                print(f"Running experiments on problem: {problem_name}")

            self.logger.info(f"Starting experiments on problem: {problem_name}")

            for algorithm_class in algorithms:
                algo_name = algorithm_class.__name__

                if verbose:
                    print(f"  Algorithm: {algo_name}")

                self.logger.info(f"Running {algo_name} on {problem_name}")

                # 获取算法参数
                params = algorithm_params.get(algo_name, {})

                # 初始化结果
                results[algo_name][problem_name] = {
                    "pareto_fronts": [],
                    "pareto_sets": [],
                    "metrics": {
                        "igdf": [],
                        "igdx": [],
                        "rpsp": [],
                        "hv": [],
                        "sp": []
                    },
                    "runtimes": []
                }

                # 运行多次实验
                for run in range(n_runs):
                    if verbose:
                        print(f"    Run {run + 1}/{n_runs}")

                    # 创建算法实例
                    algorithm = algorithm_class(problem, **params)

                    # 运行算法
                    start_time = time.time()
                    pareto_front = algorithm.optimize(tracking=True, verbose=False)
                    end_time = time.time()

                    # 收集Pareto解集（如果有）
                    if hasattr(algorithm, '_get_pareto_set'):
                        pareto_set = algorithm._get_pareto_set()
                    else:
                        pareto_set = None

                    # 记录结果
                    results[algo_name][problem_name]["pareto_fronts"].append(pareto_front)
                    if pareto_set is not None:
                        results[algo_name][problem_name]["pareto_sets"].append(pareto_set)
                    results[algo_name][problem_name]["runtimes"].append(end_time - start_time)

                    # 收集跟踪数据
                    if hasattr(algorithm, 'tracking'):
                        if run == 0:  # 只保存第一次运行的跟踪数据
                            results[algo_name][problem_name]["tracking"] = algorithm.tracking

                    # 收集指标
                    if hasattr(algorithm, 'tracking') and "metrics" in algorithm.tracking:
                        for metric_name in ["igdf", "igdx", "rpsp", "hv", "sp"]:
                            values = algorithm.tracking["metrics"].get(metric_name, [])
                            if values:
                                final_value = values[-1]
                                results[algo_name][problem_name]["metrics"][metric_name].append(final_value)

                # 计算最佳Pareto前沿
                best_idx = 0
                # 优先使用IGDF指标来确定最佳解，如果没有则尝试其他指标
                for metric_name in ["igdf", "igdx", "rpsp", "hv", "sp"]:
                    metric_values = results[algo_name][problem_name]["metrics"].get(metric_name, [])
                    if metric_values:
                        if metric_name in ["igdf", "igdx", "rpsp", "sp"]:  # 越小越好
                            best_idx = np.argmin(metric_values)
                        else:  # hv，越大越好
                            best_idx = np.argmax(metric_values)
                        break

                # 保存最佳解
                if results[algo_name][problem_name]["pareto_fronts"]:
                    results[algo_name][problem_name]["pareto_front"] = \
                        results[algo_name][problem_name]["pareto_fronts"][best_idx]
                if "pareto_sets" in results[algo_name][problem_name] and results[algo_name][problem_name][
                    "pareto_sets"]:
                    results[algo_name][problem_name]["pareto_set"] = results[algo_name][problem_name]["pareto_sets"][
                        best_idx]

                # 计算平均指标
                for metric_name in ["igdf", "igdx", "rpsp", "hv", "sp"]:
                    values = results[algo_name][problem_name]["metrics"][metric_name]
                    if values:
                        # 过滤NaN值
                        valid_values = [v for v in values if not np.isnan(v)]
                        if valid_values:
                            results[algo_name][problem_name]["metrics"][f"avg_{metric_name}"] = np.mean(valid_values)
                            results[algo_name][problem_name]["metrics"][f"std_{metric_name}"] = np.std(valid_values)
                        else:
                            results[algo_name][problem_name]["metrics"][f"avg_{metric_name}"] = float('nan')
                            results[algo_name][problem_name]["metrics"][f"std_{metric_name}"] = float('nan')
                    else:
                        results[algo_name][problem_name]["metrics"][f"avg_{metric_name}"] = float('nan')
                        results[algo_name][problem_name]["metrics"][f"std_{metric_name}"] = float('nan')

                # 输出结果汇总
                self.logger.info(f"Completed {algo_name} on {problem_name}:")
                for metric_name in ["igdf", "igdx", "rpsp", "hv", "sp"]:
                    avg_key = f"avg_{metric_name}"
                    if avg_key in results[algo_name][problem_name]["metrics"]:
                        value = results[algo_name][problem_name]["metrics"][avg_key]
                        if not np.isnan(value):
                            self.logger.info(f"  {avg_key}: {value:.6f}")

                self.logger.info(
                    f"  Average runtime: {np.mean(results[algo_name][problem_name]['runtimes']):.2f} seconds")

            # 保存问题的比较图
            algorithms_results = {algo_name: results[algo_name][problem_name] for algo_name in results.keys()}

            # 1. 绘制Pareto前沿比较图
            Visualizer.plot_pareto_front_comparison(
                problem,
                algorithms_results,
                save_path=os.path.join(self.save_dir, f"{problem_name}_pareto_front.png")
            )

            # 2. 绘制Pareto解集比较图（如果可用）
            has_pareto_sets = any("pareto_set" in result for result in algorithms_results.values())
            if has_pareto_sets:
                Visualizer.plot_pareto_set_comparison(
                    problem,
                    algorithms_results,
                    save_path=os.path.join(self.save_dir, f"{problem_name}_pareto_set.png")
                )

            # 3. 绘制各指标收敛曲线
            metrics = ["igdf", "igdx", "rpsp", "hv", "sp"]
            for metric_name in metrics:
                Visualizer.plot_convergence(
                    algorithms_results,
                    metric_name=metric_name,
                    problem_name=problem_name,
                    save_path=os.path.join(self.save_dir, f"{problem_name}_{metric_name}_convergence.png")
                )

            # 4. 绘制性能指标小提琴图/箱线图
            Visualizer.plot_algorithm_performance_boxplots(
                algorithms_results,
                problem_name,
                metrics=metrics,
                save_path=os.path.join(self.save_dir, f"{problem_name}_performance_boxplots.png")
            )
            '''
            # 5. 绘制雷达图
            Visualizer.plot_radar_chart(
                algorithms_results,
                problem_name=problem_name,
                metrics=metrics,
                save_path=os.path.join(self.save_dir, f"{problem_name}_radar_chart.png")
            )
            '''
        # 6. 绘制总体性能热图
        '''
        Visualizer.plot_metrics_summary(
            results,
            problems,
            metrics=metrics,
            save_path=os.path.join(self.save_dir, "metrics_heatmap.png")
        )
        '''
        # 7. 保存结果汇总
        self._save_summary(results, problems)

        if verbose:
            print(f"\n实验完成! 结果已保存到 {self.save_dir} 目录")
            print(f"生成了以下图像类型:")
            print(f"  - Pareto前沿比较图 (*_pareto_front.png)")
            print(f"  - Pareto解集比较图 (*_pareto_set.png)")
            print(f"  - 性能指标收敛曲线 (*_igdf/igdx/rpsp/hv/sp_convergence.png)")
            print(f"  - 性能指标小提琴图 (*_performance_boxplots.png)")
            # print(f"  - 多指标雷达图 (*_radar_chart.png)")
            # print(f"  - 性能指标热图 (metrics_heatmap_*.png)")
            print(f"  - 详细指标汇总表 (summary_*.txt)")

        return results

    def _save_summary(self, results, problems):
        """保存结果汇总 - 增强版支持IGDF、IGDX、RPSP、HV、SP指标"""
        summary_path = os.path.join(self.save_dir, "summary.txt")
        latex_path = os.path.join(self.save_dir, "summary_latex.tex")
        metrics_summary_path = os.path.join(self.save_dir, "metrics_summary.txt")

        # 首先创建简单的文本摘要
        with open(summary_path, "w") as f:
            f.write("====== 多目标优化实验结果汇总 ======\n\n")

            # 按问题组织结果
            for problem in problems:
                problem_name = problem.name
                f.write(f"问题: {problem_name}\n")
                f.write("-" * 60 + "\n")

                # 指标表格头
                f.write("\n性能指标:\n")
                algo_names = list(results.keys())
                header = "指标".ljust(15)
                for algo_name in algo_names:
                    header += (algo_name.ljust(20))
                f.write(header + "\n")
                f.write("-" * (15 + 20 * len(algo_names)) + "\n")

                # 填充指标值
                metrics = ["avg_igdf", "avg_igdx", "avg_rpsp", "avg_hv", "avg_sp"]
                metric_display = {
                    "avg_igdf": "IGDF (均值)",
                    "avg_igdx": "IGDX (均值)",
                    "avg_rpsp": "RPSP (均值)",
                    "avg_hv": "HV (均值)",
                    "avg_sp": "SP (均值)"
                }

                for metric in metrics:
                    line = metric_display[metric].ljust(15)
                    for algo_name in algo_names:
                        if problem_name in results[algo_name]:
                            value = results[algo_name][problem_name]["metrics"].get(metric, float('nan'))
                            if np.isnan(value):
                                line += "N/A".ljust(20)
                            else:
                                line += f"{value:.6f}".ljust(20)
                        else:
                            line += "N/A".ljust(20)
                    f.write(line + "\n")

                # 添加标准差信息
                std_metrics = ["std_igdf", "std_igdx", "std_rpsp", "std_hv", "std_sp"]
                std_display = {
                    "std_igdf": "IGDF (标准差)",
                    "std_igdx": "IGDX (标准差)",
                    "std_rpsp": "RPSP (标准差)",
                    "std_hv": "HV (标准差)",
                    "std_sp": "SP (标准差)"
                }

                for metric in std_metrics:
                    line = std_display[metric].ljust(15)
                    for algo_name in algo_names:
                        if problem_name in results[algo_name]:
                            value = results[algo_name][problem_name]["metrics"].get(metric, float('nan'))
                            if np.isnan(value):
                                line += "N/A".ljust(20)
                            else:
                                line += f"{value:.6f}".ljust(20)
                        else:
                            line += "N/A".ljust(20)
                    f.write(line + "\n")

                # 运行时间
                line = "运行时间".ljust(15)
                for algo_name in algo_names:
                    if problem_name in results[algo_name]:
                        value = np.mean(results[algo_name][problem_name]["runtimes"])
                        line += f"{value:.2f}s".ljust(20)
                    else:
                        line += "N/A".ljust(20)
                f.write(line + "\n\n")

                # Pareto前沿大小
                line = "前沿大小".ljust(15)
                for algo_name in algo_names:
                    if problem_name in results[algo_name] and "pareto_front" in results[algo_name][problem_name]:
                        value = len(results[algo_name][problem_name]["pareto_front"])
                        line += f"{value}".ljust(20)
                    else:
                        line += "N/A".ljust(20)
                f.write(line + "\n\n")

                f.write("=" * 60 + "\n\n")

            # 总体性能排名
            f.write("\n总体性能排名:\n")
            f.write("-" * 50 + "\n")

            # 计算每个算法在每个指标上的平均排名
            rankings = {algo_name: {"igdf": [], "igdx": [], "rpsp": [], "hv": [], "sp": []} for algo_name in algo_names}

            for problem in problems:
                problem_name = problem.name

                # 计算IGDF排名
                igdf_values = {}
                for algo_name in algo_names:
                    if problem_name in results[algo_name] and "avg_igdf" in results[algo_name][problem_name]["metrics"]:
                        igdf_values[algo_name] = results[algo_name][problem_name]["metrics"]["avg_igdf"]

                if igdf_values:
                    # 对IGDF值进行排序（较小的值排名靠前）
                    sorted_algos = sorted(igdf_values.keys(),
                                          key=lambda x: igdf_values[x] if not np.isnan(igdf_values[x]) else float(
                                              'inf'))
                    for rank, algo_name in enumerate(sorted_algos, 1):
                        if not np.isnan(igdf_values[algo_name]):
                            rankings[algo_name]["igdf"].append(rank)

                # 计算IGDX排名
                igdx_values = {}
                for algo_name in algo_names:
                    if problem_name in results[algo_name] and "avg_igdx" in results[algo_name][problem_name]["metrics"]:
                        igdx_values[algo_name] = results[algo_name][problem_name]["metrics"]["avg_igdx"]

                if igdx_values:
                    # 对IGDX值进行排序（较小的值排名靠前）
                    sorted_algos = sorted(igdx_values.keys(),
                                          key=lambda x: igdx_values[x] if not np.isnan(igdx_values[x]) else float(
                                              'inf'))
                    for rank, algo_name in enumerate(sorted_algos, 1):
                        if not np.isnan(igdx_values[algo_name]):
                            rankings[algo_name]["igdx"].append(rank)

                # 计算RPSP排名
                rpsp_values = {}
                for algo_name in algo_names:
                    if problem_name in results[algo_name] and "avg_rpsp" in results[algo_name][problem_name]["metrics"]:
                        rpsp_values[algo_name] = results[algo_name][problem_name]["metrics"]["avg_rpsp"]

                if rpsp_values:
                    # 对RPSP值进行排序（较小的值排名靠前）
                    sorted_algos = sorted(rpsp_values.keys(),
                                          key=lambda x: rpsp_values[x] if not np.isnan(rpsp_values[x]) else float(
                                              'inf'))
                    for rank, algo_name in enumerate(sorted_algos, 1):
                        if not np.isnan(rpsp_values[algo_name]):
                            rankings[algo_name]["rpsp"].append(rank)

                # 计算HV排名
                hv_values = {}
                for algo_name in algo_names:
                    if problem_name in results[algo_name] and "avg_hv" in results[algo_name][problem_name]["metrics"]:
                        hv_values[algo_name] = results[algo_name][problem_name]["metrics"]["avg_hv"]

                if hv_values:
                    # 对HV值进行排序（较大的值排名靠前）
                    sorted_algos = sorted(hv_values.keys(),
                                          key=lambda x: -hv_values[x] if not np.isnan(hv_values[x]) else float('inf'))
                    for rank, algo_name in enumerate(sorted_algos, 1):
                        if not np.isnan(hv_values[algo_name]):
                            rankings[algo_name]["hv"].append(rank)

                # 计算SP排名
                sp_values = {}
                for algo_name in algo_names:
                    if problem_name in results[algo_name] and "avg_sp" in results[algo_name][problem_name]["metrics"]:
                        sp_values[algo_name] = results[algo_name][problem_name]["metrics"]["avg_sp"]

                if sp_values:
                    # 对SP值进行排序（较小的值排名靠前）
                    sorted_algos = sorted(sp_values.keys(),
                                          key=lambda x: sp_values[x] if not np.isnan(sp_values[x]) else float('inf'))
                    for rank, algo_name in enumerate(sorted_algos, 1):
                        if not np.isnan(sp_values[algo_name]):
                            rankings[algo_name]["sp"].append(rank)

            # 计算平均排名
            avg_rankings = {}
            for algo_name in algo_names:
                avg_igdf = np.mean(rankings[algo_name]["igdf"]) if rankings[algo_name]["igdf"] else float('nan')
                avg_igdx = np.mean(rankings[algo_name]["igdx"]) if rankings[algo_name]["igdx"] else float('nan')
                avg_rpsp = np.mean(rankings[algo_name]["rpsp"]) if rankings[algo_name]["rpsp"] else float('nan')
                avg_hv = np.mean(rankings[algo_name]["hv"]) if rankings[algo_name]["hv"] else float('nan')
                avg_sp = np.mean(rankings[algo_name]["sp"]) if rankings[algo_name]["sp"] else float('nan')

                # 计算综合排名（五个指标的平均值）
                valid_rankings = [r for r in [avg_igdf, avg_igdx, avg_rpsp, avg_hv, avg_sp] if not np.isnan(r)]
                avg_overall = np.mean(valid_rankings) if valid_rankings else float('nan')

                avg_rankings[algo_name] = {
                    "igdf": avg_igdf,
                    "igdx": avg_igdx,
                    "rpsp": avg_rpsp,
                    "hv": avg_hv,
                    "sp": avg_sp,
                    "overall": avg_overall
                }

            # 输出排名表格
            header = "算法".ljust(20) + "IGDF排名".ljust(15) + "IGDX排名".ljust(15) + "RPSP排名".ljust(
                15) + "HV排名".ljust(15) + "SP排名".ljust(15) + "综合排名".ljust(15)
            f.write(header + "\n")
            f.write("-" * (20 + 15 * 6) + "\n")

            # 按总体排名排序算法
            sorted_algos = sorted(avg_rankings.keys(), key=lambda x: avg_rankings[x]["overall"] if not np.isnan(
                avg_rankings[x]["overall"]) else float('inf'))

            for algo_name in sorted_algos:
                line = algo_name.ljust(20)
                line += f"{avg_rankings[algo_name]['igdf']:.2f}".ljust(15)
                line += f"{avg_rankings[algo_name]['igdx']:.2f}".ljust(15)
                line += f"{avg_rankings[algo_name]['rpsp']:.2f}".ljust(15)
                line += f"{avg_rankings[algo_name]['hv']:.2f}".ljust(15)
                line += f"{avg_rankings[algo_name]['sp']:.2f}".ljust(15)
                line += f"{avg_rankings[algo_name]['overall']:.2f}".ljust(15)
                f.write(line + "\n")

            f.write("\n注意: 排名值越低越好。IGDF、IGDX、RPSP、SP指标值越小越好，HV指标值越大越好。\n")

        # 创建LaTeX格式的表格 - 类似于图片中的格式
        with open(latex_path, "w") as f:
            f.write("% LaTeX表格格式 - 可直接复制到LaTeX文档中使用\n")
            f.write("\\begin{table}[htbp]\n")
            f.write("\\centering\n")
            f.write("\\caption{各算法在不同测试问题上的性能指标。显著值以粗体标记。}\n")
            f.write("\\begin{tabular}{|l|l|" + "c|" * len(algo_names) + "}\n")
            f.write("\\hline\n")
            f.write("ID & & " + " & ".join(algo_names) + " \\\\ \\hline\n")

            # 按问题和指标填充表格
            for problem in problems:
                problem_name = problem.name

                # 对每个指标创建两行 (Mean和Std)
                for metric_base in ["igdf", "igdx", "rpsp", "hv", "sp"]:
                    metric_mean = f"avg_{metric_base}"
                    metric_std = f"std_{metric_base}"

                    # 确定最优值 (IGDF,IGDX,RPSP,SP是越小越好，HV是越大越好)
                    best_values = {}
                    for algo_name in algo_names:
                        if problem_name in results[algo_name] and metric_mean in results[algo_name][problem_name][
                            "metrics"]:
                            value = results[algo_name][problem_name]["metrics"][metric_mean]
                            if not np.isnan(value):
                                best_values[algo_name] = value

                    if best_values:
                        if metric_base in ["igdf", "igdx", "rpsp", "sp"]:
                            best_algo = min(best_values, key=best_values.get)
                            best_value = best_values[best_algo]
                        else:  # hv
                            best_algo = max(best_values, key=best_values.get)
                            best_value = best_values[best_algo]

                    # 均值行
                    f.write(f"{problem_name} & Mean & ")
                    mean_values = []
                    for algo_name in algo_names:
                        if problem_name in results[algo_name] and metric_mean in results[algo_name][problem_name][
                            "metrics"]:
                            value = results[algo_name][problem_name]["metrics"][metric_mean]
                            if np.isnan(value):
                                mean_values.append("---")
                            else:
                                # 检查是否是最优值
                                is_best = False
                                if best_values:
                                    if metric_base in ["igdf", "igdx", "rpsp", "sp"]:
                                        is_best = np.abs(value - best_value) < 1e-6
                                    else:  # hv
                                        is_best = np.abs(value - best_value) < 1e-6

                                if is_best:
                                    mean_values.append(f"\\textbf{{{value:.4e}}}")
                                else:
                                    mean_values.append(f"{value:.4e}")
                        else:
                            mean_values.append("---")

                    f.write(" & ".join(mean_values) + " \\\\ \n")

                    # 标准差行
                    f.write(f" & Std & ")
                    std_values = []
                    for algo_name in algo_names:
                        if problem_name in results[algo_name] and metric_std in results[algo_name][problem_name][
                            "metrics"]:
                            value = results[algo_name][problem_name]["metrics"][metric_std]
                            if np.isnan(value):
                                std_values.append("---")
                            else:
                                # 标准差不需要标记最优
                                std_values.append(f"{value:.4e}")
                        else:
                            std_values.append("---")

                    f.write(" & ".join(std_values) + " \\\\ \\hline\n")

            f.write("\\end{tabular}\n")
            f.write("\\label{tab:performance_metrics}\n")
            f.write("\\end{table}\n")

        # 创建每个指标的详细汇总
        with open(metrics_summary_path, "w") as f:
            f.write("====== 多目标优化实验详细指标汇总 ======\n\n")

            # 为每个指标创建独立表格
            for metric_base in ["igdf", "igdx", "rpsp", "hv", "sp"]:
                metric_name = metric_base.upper()
                if metric_base == "igdf":
                    f.write(f"\n{metric_name} - 目标空间收敛性指标 (越小越好)\n")
                elif metric_base == "igdx":
                    f.write(f"\n{metric_name} - 决策空间收敛性指标 (越小越好)\n")
                elif metric_base == "rpsp":
                    f.write(f"\n{metric_name} - r-径向集逼近指标 (越小越好)\n")
                elif metric_base == "hv":
                    f.write(f"\n{metric_name} - 超体积指标 (越大越好)\n")
                elif metric_base == "sp":
                    f.write(f"\n{metric_name} - 解分布均匀性指标 (越小越好)\n")

                f.write("-" * 80 + "\n")

                # 表头
                header = "问题".ljust(15)
                for algo_name in algo_names:
                    header += (algo_name.ljust(20))
                f.write(header + "\n")
                f.write("-" * (15 + 20 * len(algo_names)) + "\n")

                # 填充每个问题的值
                for problem in problems:
                    problem_name = problem.name
                    metric_mean = f"avg_{metric_base}"

                    # 找出最优值
                    best_values = {}
                    for algo_name in algo_names:
                        if problem_name in results[algo_name] and metric_mean in results[algo_name][problem_name][
                            "metrics"]:
                            value = results[algo_name][problem_name]["metrics"][metric_mean]
                            if not np.isnan(value):
                                best_values[algo_name] = value

                    if best_values:
                        if metric_base in ["igdf", "igdx", "rpsp", "sp"]:
                            best_algo = min(best_values, key=best_values.get)
                            best_value = best_values[best_algo]
                        else:  # hv
                            best_algo = max(best_values, key=best_values.get)
                            best_value = best_values[best_algo]

                    # 填充值
                    line = problem_name.ljust(15)
                    for algo_name in algo_names:
                        if problem_name in results[algo_name] and metric_mean in results[algo_name][problem_name][
                            "metrics"]:
                            value = results[algo_name][problem_name]["metrics"][metric_mean]
                            if np.isnan(value):
                                line += "N/A".ljust(20)
                            else:
                                # 标记最优值
                                is_best = False
                                if best_values:
                                    if metric_base in ["igdf", "igdx", "rpsp", "sp"]:
                                        is_best = np.abs(value - best_value) < 1e-6
                                    else:  # hv
                                        is_best = np.abs(value - best_value) < 1e-6

                                if is_best:
                                    line += f"*{value:.6f}*".ljust(20)
                                else:
                                    line += f"{value:.6f}".ljust(20)
                        else:
                            line += "N/A".ljust(20)
                    f.write(line + "\n")

                f.write("\n")

            # 添加综合排名
            f.write("\n总体排名汇总 (数值越小越好)\n")
            f.write("-" * 80 + "\n")

            header = "算法".ljust(15) + "IGDF排名".ljust(12) + "IGDX排名".ljust(12) + "RPSP排名".ljust(
                12) + "HV排名".ljust(12) + "SP排名".ljust(12) + "综合排名".ljust(12)
            f.write(header + "\n")
            f.write("-" * 80 + "\n")

            # 按综合排名排序
            for algo_name in sorted_algos:
                line = algo_name.ljust(15)
                line += f"{avg_rankings[algo_name]['igdf']:.2f}".ljust(12)
                line += f"{avg_rankings[algo_name]['igdx']:.2f}".ljust(12)
                line += f"{avg_rankings[algo_name]['rpsp']:.2f}".ljust(12)
                line += f"{avg_rankings[algo_name]['hv']:.2f}".ljust(12)
                line += f"{avg_rankings[algo_name]['sp']:.2f}".ljust(12)
                line += f"{avg_rankings[algo_name]['overall']:.2f}".ljust(12)
                f.write(line + "\n")


# ====================== 主函数 ======================

def main():
    """主函数，运行实验"""
    # 设置随机种子
    np.random.seed(42)
    random.seed(42)

    # 创建结果目录
    results_dir = "cec2020_results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # 设置问题
    problems = [
        TP1(n_var=30),
        TP2(n_var=30),
        TP3(n_var=30),
        TP4(n_var=30),
        TP5(n_var=30),
        TP6(n_var=30),
        TP7(n_var=30),
        TP8(n_var=30),
        TP9(n_var=30),
        TP10(n_var=30),
        TP11(n_var=30)
    ]

    # 设置算法 - 使用所有五种算法
    algorithms = [
        CASMOPSO,
        MOPSO,
        NSGAII,
        MOEAD,
        SPEA2
    ]

    # 算法参数
    algorithm_params = {
        "CASMOPSO": {
            "pop_size": 200,  # 种群大小
            "max_iterations": 200,  # 迭代次数
            "w_init": 0.9,  # 惯性权重初始值
            "w_end": 0.4,  # 惯性权重终值
            "c1_init": 2.5,  # 个体认知
            "c1_end": 0.5,  # 个体认知终值
            "c2_init": 0.5,  # 社会认知初值
            "c2_end": 2.5,  # 社会认知终值
            "use_archive": True,  # 存档大小
            "archive_size": 300,
            "mutation_rate": 0.1,  # 变异率
            "adaptive_grid_size": 15,  # 网格大小
            "k_vmax": 0.5 #速度限制因子
        },
        "MOPSO": {  # 使用新的动态参数接口
            "pop_size": 150,
            "max_iterations": 200,
            "w_init": 0.9, "w_end": 0.4,  # 动态惯性权重
            "c1_init": 1.5, "c1_end": 1.5,  # (等效于 c1=1.5)
            "c2_init": 1.5, "c2_end": 1.5,  # (等效于 c2=1.5)
            "use_archive": True,
            "archive_size": 150  # 标准存档大小
        },
        "NSGAII": {
            "pop_size": 150,
            "max_generations": 200
        },
        "MOEAD": {
            "pop_size": 150,
            "max_generations": 200,
            "T": 20,
            "delta": 0.9,
            "nr": 2
        },
        "SPEA2": {
            "pop_size": 100,
            "archive_size": 100,
            "max_generations": 200
        }
    }

    # 创建实验框架
    experiment = ExperimentFramework(save_dir=results_dir)

    # 运行实验
    results = experiment.run_experiment(
        problems=problems,
        algorithms=algorithms,
        algorithm_params=algorithm_params,
        n_runs=10,  # 减少运行次数以节省时间
        verbose=True
    )

    print(f"\n实验完成! 结果已保存到 {results_dir} 目录")


if __name__ == "__main__":
    main()
