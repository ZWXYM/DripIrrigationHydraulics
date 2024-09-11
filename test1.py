import random
import math
import numpy as np
from deap import base, creator, tools, algorithms
from typing import List, Tuple

dripper_distance = 0.8
# 常量定义
WATER_DENSITY = 1000  # kg/m^3
WATER_KINEMATIC_VISCOSITY = 1.004e-6  # m^2/s at 20°C
GRAVITY = 9.81  # m/s^2


def water_speed(diameter: float, flow: float) -> float:
    """计算流速 (m/s)"""
    d = diameter / 1000  # 转换为米
    area = math.pi * (d / 2) ** 2
    return flow / area


def reynolds_number(diameter: float, velocity: float) -> float:
    """计算雷诺数"""
    d = diameter / 1000  # 转换为米
    return (velocity * d) / WATER_KINEMATIC_VISCOSITY


def friction_factor(diameter: float, flow_rate: float) -> Tuple[float, float]:
    """计算摩擦系数和雷诺数"""
    d = diameter / 1000  # 转换为米
    velocity = water_speed(diameter, flow_rate)
    Re = reynolds_number(diameter, velocity)

    # Colebrook-White 方程的简化形式
    epsilon = 1.5e-6  # 管道粗糙度，假设为光滑管道
    relative_roughness = epsilon / d

    if Re < 2300:  # 层流
        f = 64 / Re
    elif Re > 4000:  # 湍流
        # 使用 Swamee-Jain 方程近似求解 Colebrook-White 方程
        f = 0.25 / (math.log10(relative_roughness / 3.7 + 5.74 / Re ** 0.9)) ** 2
    else:  # 过渡区
        # 在过渡区线性插值
        f_lam = 64 / 2300
        f_turb = 0.25 / (math.log10(relative_roughness / 3.7 + 5.74 / 4000 ** 0.9)) ** 2
        f = f_lam + (f_turb - f_lam) * (Re - 2300) / (4000 - 2300)

    return f, Re


def pressure_loss(diameter: float, length: float, flow_rate: float) -> float:
    """计算水头损失 (m)"""
    d = diameter / 1000  # 转换为米
    v = water_speed(diameter, flow_rate)
    f, _ = friction_factor(diameter, flow_rate)
    return f * (length / d) * (v ** 2 / (2 * GRAVITY))


# ... (其他函数保持不变)
def shuili(dripper_length, dripper_spacing, dripper_min_flow, fuzhu_sub_length, field_length, field_wide,
           Soil_bulk_density,
           field_z,
           field_p, field_max, field_min, sr, st, ib, nn, work_time, lgz0, lgz1, lgz2, Full_field_long,
           Full_field_wide):
    num_rows, plants_per_row, total_plants = calculate_plant_count(field_length, fuzhu_sub_length, sr, st)
    num_dripper = calculate_dripper_count(dripper_length, dripper_spacing, fuzhu_sub_length)
    block_number = Full_field_long / field_length * Full_field_wide / field_wide
    fuzhu_number = int(field_wide / fuzhu_sub_length)
    plant = num_dripper / total_plants  # 每个植物附件的滴头数
    mmax = Soil_bulk_density * field_z * field_p * (field_max - field_min)  # 最大净灌水定额(mm)
    TMAX = mmax / ib  # 设计灌水周期d、mmax单位
    m1 = mmax / nn  # 设计毛灌水定额(mm)
    if plant <= 1:
        t = (m1 * dripper_spacing * st) / dripper_min_flow  # 一次灌水延续时间h
    else:
        t = (m1 * sr * st) / plant * dripper_min_flow  # 一次灌水延续时间h
    T = t * (fuzhu_number / lgz0) * (block_number / lgz1) * math.ceil(22 / lgz2) / work_time  # 完成所有灌水所需时间Day
    print(f"设计灌水周期: {TMAX:.3f}天")
    print(f"一次灌水延续时间：{t:.3f}h")
    print(f"完成所有灌水所需时间：{T:.3f}天")
    return TMAX, t


def calculate_flow_rates(dripper_min_flow, dripper_length, dripper_spacing, fuzhu_sub_length, lgz0, lgz1):
    dripper_flow = dripper_min_flow / 3600000  # L/h转换为m³/s，单条滴灌带流量
    num_drippers = math.floor(dripper_length / dripper_spacing)  # 一条滴灌带上的滴灌头数量
    lateral_flow = dripper_flow * math.floor(fuzhu_sub_length / dripper_distance) * 2 * lgz0 * num_drippers
    # 单条辅助农管上滴灌带单侧数量*双侧2*每个轮灌组开启的辅助农管条数1
    sub_flow = lateral_flow * lgz1  # 工作状态下干管在一个轮灌组内的流量
    return lateral_flow, sub_flow, dripper_flow


# 计算地块内支管所需最小水头及水头损失
def calculate_head(sub_diameter, lateral_diameter, length_x, length_y, dripper_min_flow, dripper_length,
                   dripper_spacing, fuzhu_sub_length, lgz0, lgz1, lgz2):
    lateral_flow, sub_flow, dripper_flow = calculate_flow_rates(dripper_min_flow, dripper_length,
                                                                dripper_spacing, fuzhu_sub_length, lgz0, lgz1)

    # 计算农管水头损失
    lateral_loss = pressure_loss(lateral_diameter, length_x, lateral_flow)

    # 计算斗管水头损失，使用50到length_y，每次变化100，将所得结果求和后取平均值
    sub_losses = [pressure_loss(sub_diameter, y, sub_flow) for y in range(50, length_y + 1, 100)]
    sub_loss = sum(sub_losses) / len(sub_losses)
    # sub_loss = pressure_loss(sub_diameter, length_y, sub_flow)
    # 农管上一条滴灌带所需水头
    dripper_loss = 10

    # 计算一条支管所需水头
    required_head = lateral_loss + sub_loss + dripper_loss

    # 计算一条支管所需压力
    pressure = required_head * 1e3 * 9.81

    f_lateral, Re_lateral = friction_factor(lateral_diameter, lateral_flow)  # 沿程阻力系数

    f_sub, Re_sub = friction_factor(sub_diameter, sub_flow)  # 沿程阻力系数
    lateral_speed = water_speed(lateral_diameter, lateral_flow)
    sub_speed = water_speed(sub_diameter, sub_flow)
    main_speed = water_speed(600, sub_flow * lgz2)
    print(f'最远端滴灌带所需水头: {required_head:.3f} m')
    print(f'最远端滴灌带所需压力: {pressure / 1000000:.3f} MPa')
    print(f"滴灌带水头: {dripper_loss:.3f} m")
    print(f"单条农管入口所需水头: {lateral_loss + dripper_loss:.3f} m")
    print(f"农管雷诺数: {Re_lateral:.3f}")
    print(f"农管沿程阻力系数: {f_lateral:.3f}")
    print(f"农管水头损失: {lateral_loss:.3f} m")
    print(f"农管轮灌最大流量: {lateral_flow:.3f} m³/s")
    print(f"农管轮灌最大流速: {lateral_speed:.3f} m/s")
    print(f"斗管雷诺数: {Re_sub:.3f}")
    print(f"斗管沿程阻力系数: {f_sub:.3f}")
    print(f"斗管水头损失: {sub_loss:.3f} m")
    print(f"斗管轮灌最大流量: {sub_flow:.3f} m³/s")
    print(f"斗管轮灌最大流速: {sub_speed:.3f} m/s")
    print(f"支管轮灌最大流速: {main_speed:.3f} m/s")  # 开启多少条斗管
    return required_head


# 计算地块内植物数量
def calculate_plant_count(field_length, fuzhu_sub_length, sr, st):
    # 计算行数
    num_rows = math.floor(fuzhu_sub_length / sr)
    # 计算每行植物数
    plants_per_row = math.floor(field_length / st)  # 将cm转换为m
    # 计算总植物数
    total_plants = math.floor(num_rows * plants_per_row)
    return num_rows, plants_per_row, total_plants


# 计算辅助农管控制范围内滴灌头数量
def calculate_dripper_count(dripper_length, dripper_spacing, fuzhu_sub_length):
    num_drippers = math.floor(dripper_length / dripper_spacing)
    num_dripper = math.floor(fuzhu_sub_length / dripper_distance) * num_drippers
    return num_dripper


class Fitness:
    """适应度类，替代 DEAP 的 creator.FitnessMin"""
    def __init__(self):
        self.values = ()
        self.valid = False

    def __lt__(self, other):
        return self.values < other.values

class Individual(list):
    """个体类，替代 DEAP 的 creator.Individual"""
    def __init__(self, *args):
        super().__init__(*args)
        self.fitness = Fitness()


# 修改评估函数以确保 lgz1 为偶数
def evaluate(individual: Individual) -> Tuple[float]:
    lgz1, lgz2 = individual
    lgz1 = lgz1 * 2  # Ensure lgz1 is even

    # Calculate required head and irrigation time
    required_head, dripper_loss = calculate_head(160, 90, 180, 750, 2.1, 50, 0.3, 40, 1, lgz1, lgz2)
    TMAX, T = shuili(50, 0.3, 2.1, 40, 100, 200, 1.46, 600, 0.7, 0.9, 0.8, 0.8, 0.1, 8, 0.9, 15, 1, lgz1, lgz2, 800,
                     800)

    # Calculate the objective function value (head ratio)
    head_ratio = required_head / dripper_loss

    # Check constraints
    if T > TMAX:
        return (float('inf'),)  # Return a very large value as a penalty

    return (head_ratio,)


# 设置遗传算法参数
toolbox = base.Toolbox()
toolbox.register("attr_lgz1", random.randint, 1, 5)  # lgz1 范围为 1-5，实际值会是 2-10 的偶数
toolbox.register("attr_lgz2", random.randint, 1, 20)
toolbox.register("individual", tools.initCycle, Individual,
                 (toolbox.attr_lgz1, toolbox.attr_lgz2), n=1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutUniformInt, low=[1, 1], up=[5, 20], indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)


def main():
    random.seed(42)
    pop = toolbox.population(n=50)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)

    # 自定义简单遗传算法
    for gen in range(50):
        # 评估种群
        invalid_ind = [ind for ind in pop if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
            ind.fitness.valid = True

        # 选择下一代个体
        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))

        # 对选中的个体进行交叉和变异
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < 0.7:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < 0.2:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # 更新种群
        pop[:] = offspring

        # 输出统计信息
        fits = [ind.fitness.values[0] for ind in pop]
        print(f"Generation {gen}: Min {min(fits)}, Max {max(fits)}, Avg {sum(fits) / len(fits)}")

        # 更新名人堂
        hof.update(pop)

    best = hof[0]
    lgz1 = best[0] * 2  # 确保 lgz1 为偶数
    lgz2 = best[1]
    print(f"\nBest solution: lgz1 = {lgz1}, lgz2 = {lgz2}")
    print(f"Best fitness: {best.fitness.values[0]}")

    # 计算并打印最佳解的详细信息
    required_head, dripper_loss = calculate_head(160, 90, 180, 750, 2.1, 50, 0.3, 40, 1, lgz1, lgz2)
    TMAX, T = shuili(50, 0.3, 2.1, 40, 100, 200, 1.46, 600, 0.7, 0.9, 0.8, 0.8, 0.1, 8, 0.9, 15, 1, lgz1, lgz2, 800,
                     800)

    print(f"\nDetailed results:")
    print(f"Required head: {required_head:.3f} m")
    print(f"Dripper loss: {dripper_loss:.3f} m")
    print(f"Head ratio: {required_head / dripper_loss:.3f}")
    print(f"TMAX: {TMAX:.3f} days")
    print(f"T: {T:.3f} days")


if __name__ == "__main__":
    main()
