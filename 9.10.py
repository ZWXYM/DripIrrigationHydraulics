import random
import math
import numpy as np
from deap import base, creator, tools, algorithms
from typing import Any

dripper_distance = 0.8


# 从原始代码中复制必要的函数
def water_speed(diameter, flow):
    d = diameter / 1000
    speed = flow / ((d / 2) ** 2 * math.pi)
    return speed


def friction_factor(diameter, flow_rate, pipe_roughness):
    d = diameter / 1000  # 转换为米
    v = water_speed(diameter, flow_rate)
    Re = 1000 * v * d / 1.004e-3  # 动力粘度可以作为参数传入
    relative_roughness = pipe_roughness / d

    if Re < 2300:
        # 层流
        return 64 / Re, Re
    elif Re > 4000:
        # 湍流，使用Colebrook-White方程的显式近似
        A = (relative_roughness / 3.7)**1.11 + (5.74 / Re) ** 0.9
        f = 0.25 / (math.log10(A) ** 2)
        return f, Re
    else:
        # 过渡区，线性插值
        f_2300 = 64 / 2300
        A_4000 = relative_roughness / 3.7 + 5.74 / 4000 ** 0.9
        f_4000 = 0.25 / (math.log10(A_4000) ** 2)
        f = f_2300 + (f_4000 - f_2300) * (Re - 2300) / (4000 - 2300)
        return f, Re


def pressure_loss(diameter, length, flow_rate):
    f, Re = friction_factor(diameter, flow_rate, 1.5e-6)
    d = diameter / 1000
    v = water_speed(diameter, flow_rate)
    h_f = f * (length / d) * (v ** 2 / (2 * 9.81))
    return h_f


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


def calculate_flow_rates(dripper_min_flow, dripper_length, dripper_spacing, fuzhu_sub_length, lgz0, lgz1):
    dripper_flow = dripper_min_flow / 3600000
    num_drippers = math.floor(dripper_length / dripper_spacing)
    lateral_flow = dripper_flow * math.floor(fuzhu_sub_length / dripper_distance) * 2 * lgz0 * num_drippers
    sub_flow = lateral_flow * lgz1
    return lateral_flow, sub_flow, dripper_flow


def calculate_head(sub_diameter, lateral_diameter, length_x, length_y, dripper_min_flow, dripper_length,
                   dripper_spacing, fuzhu_sub_length, lgz0, lgz1, lgz2):
    lateral_flow, sub_flow, dripper_flow = calculate_flow_rates(dripper_min_flow, dripper_length,
                                                                dripper_spacing, fuzhu_sub_length, lgz0, lgz1)
    lateral_loss = pressure_loss(lateral_diameter, length_x, lateral_flow)
    sub_losses = [pressure_loss(sub_diameter, y, sub_flow) for y in range(50, length_y + 1, 100)]
    sub_loss = sum(sub_losses) / len(sub_losses)
    dripper_loss = 10
    required_head = lateral_loss + sub_loss + dripper_loss
    main_speed = water_speed(600, sub_flow * lgz2)
    return required_head, dripper_loss, main_speed


def shuili(dripper_length, dripper_spacing, dripper_min_flow, fuzhu_sub_length, field_length, field_wide,
           Soil_bulk_density, field_z, field_p, field_max, field_min, sr, st, ib, nn, work_time, lgz0, lgz1, lgz2,
           Full_field_long, Full_field_wide):
    num_rows = math.floor(fuzhu_sub_length / sr)
    plants_per_row = math.floor(field_length / st)
    total_plants = math.floor(num_rows * plants_per_row)
    num_dripper = math.floor(fuzhu_sub_length / 0.8) * math.floor(dripper_length / dripper_spacing)
    block_number = Full_field_long / field_length * Full_field_wide / field_wide
    fuzhu_number = int(field_wide / fuzhu_sub_length)
    plant = num_dripper / total_plants
    mmax = Soil_bulk_density * field_z * field_p * (field_max - field_min)
    TMAX = mmax / ib
    m1 = mmax / nn
    if plant <= 1:
        t = (m1 * dripper_spacing * st) / dripper_min_flow
    else:
        t = (m1 * sr * st) / plant * dripper_min_flow
    T = t * (fuzhu_number / lgz0) * (block_number / lgz1) * math.ceil(22 / lgz2) / work_time
    return TMAX, T


# 定义评估函数
def evaluate(individual):
    lgz1, lgz2 = individual

    # 计算所需水头和灌水时间
    required_head, dripper_loss, main_speed = calculate_head(160, 90, 180, 750, 2.1, 50, 0.3, 40, 1, lgz1, lgz2)
    TMAX, T = shuili(50, 0.3, 2.1, 40, 100, 200, 1.46, 600, 0.7, 0.9, 0.8, 0.8, 0.1, 8, 0.9, 15, 1, lgz1, lgz2, 800,
                     800)

    # 计算目标函数值（水头比）
    head_ratio = required_head / dripper_loss

    # 检查约束条件
    if T > TMAX:
        return (float('inf'),)  # 返回一个非常大的值作为惩罚
    else:
        if lgz1 % 2 == 0:
            if main_speed <= 2:
                return (head_ratio,)
            else:
                return (float('inf'),)
        else:
            return (float('inf'),)


# 设置遗传算法参数
toolbox: Any = base.Toolbox()
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)
toolbox.register("attr_lgz1", random.randint, 1, 10)
toolbox.register("attr_lgz2", random.randint, 1, 20)
toolbox.register("individual", tools.initCycle, creator.Individual,
                 (toolbox.attr_lgz1, toolbox.attr_lgz2), n=1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutUniformInt, low=[1, 1], up=[10, 20], indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)


# 主函数
def main():
    random.seed(42)
    pop = toolbox.population(n=50)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)

    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.7, mutpb=0.2, ngen=100,
                                   stats=stats, halloffame=hof, verbose=True)
    best_individual = tools.selBest(pop, 1)[0]
    print(f"Best individual: {best_individual}")
    print(f"Best fitness: {best_individual.fitness.values[0]}")
    best = hof[0]
    print(f"\nBest solution: lgz1 = {best[0]}, lgz2 = {best[1]}")
    print(f"Best fitness: {best.fitness.values[0]}")

    # 计算并打印最佳解的详细信息
    required_head, dripper_loss, main_speed = calculate_head(160, 90, 180, 750, 2.1, 50, 0.3, 40, 1, best[0], best[1])

    print(f"\nDetailed results:")
    print(f"Head ratio: {required_head / dripper_loss:.3f}")
    final_printa(160, 90, 180, 750, 2.1, 50, 0.3, 40, 1, best[0], best[1])
    final_printb(50, 0.3, 2.1, 40, 100, 200, 1.46, 600, 0.7, 0.9, 0.8, 0.8, 0.1, 8, 0.9, 15, 1, best[0], best[1],
                 800, 800)


def final_printa(sub_diameter, lateral_diameter, length_x, length_y, dripper_min_flow, dripper_length,
                 dripper_spacing, fuzhu_sub_length, lgz0, lgz1, lgz2):
    lateral_flow, sub_flow, dripper_flow = calculate_flow_rates(dripper_min_flow, dripper_length,
                                                                dripper_spacing, fuzhu_sub_length, lgz0, lgz1)
    lateral_loss = pressure_loss(lateral_diameter, length_x, lateral_flow)
    sub_losses = [pressure_loss(sub_diameter, y, sub_flow) for y in range(50, length_y + 1, 100)]
    sub_loss = sum(sub_losses) / len(sub_losses)
    dripper_loss = 10
    required_head = lateral_loss + sub_loss + dripper_loss
    pressure = required_head * 1e3 * 9.81
    f_lateral, Re_lateral = friction_factor(lateral_diameter, lateral_flow, 1.5e-6)  # 沿程阻力系数
    f_sub, Re_sub = friction_factor(sub_diameter, sub_flow, 1.5e-6)  # 沿程阻力系数
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


def final_printb(dripper_length, dripper_spacing, dripper_min_flow, fuzhu_sub_length, field_length, field_wide,
                 Soil_bulk_density, field_z, field_p, field_max, field_min, sr, st, ib, nn, work_time, lgz0, lgz1, lgz2,
                 Full_field_long, Full_field_wide):
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


if __name__ == "__main__":
    main()
