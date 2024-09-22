import random
import math
import numpy as np
from deap import base, creator, tools, algorithms
from typing import Any

dripper_spacing = 0.3


# 流速计算函数，输入管径mm、流量m3/s
def water_speed(diameter, flow_rate):
    d = diameter / 1000
    speed = flow_rate / ((d / 2) ** 2 * math.pi)
    return speed


# 雷诺数及沿程阻力系数计算函数
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
        A = (relative_roughness / 3.7) ** 1.11 + (5.74 / Re) ** 0.9
        f = 0.25 / (math.log10(A) ** 2)
        return f, Re
    else:
        # 过渡区，线性插值
        f_2300 = 64 / 2300
        A_4000 = relative_roughness / 3.7 + 5.74 / 4000 ** 0.9
        f_4000 = 0.25 / (math.log10(A_4000) ** 2)
        f = f_2300 + (f_4000 - f_2300) * (Re - 2300) / (4000 - 2300)
        return f, Re


# 水头损失值计算输入管径mm、长度m、流量m3/s
def pressure_loss(diameter, length, flow_rate):
    f, Re = friction_factor(diameter, flow_rate, 1.5e-6)
    d = diameter / 1000
    v = water_speed(diameter, flow_rate)
    h_f = f * (length / d) * (v ** 2 / (2 * 9.81))
    return h_f


# 计算地块内植物数量，输入地块长度、辅助农管长度、
def calculate_plant_count(fuzhu_field_wide, fuzhu_sub_length, sr, st):
    # 计算行数
    num_rows = math.floor(fuzhu_sub_length / sr)
    # 计算每行植物数
    plants_per_row = math.floor(fuzhu_field_wide / st)  # 将cm转换为m
    # 计算总植物数
    total_plants = math.floor(num_rows * plants_per_row)
    return num_rows, plants_per_row, total_plants


# 计算辅助农管控制范围内滴灌头数量
def calculate_dripper_count(dripper_length, fuzhu_sub_length, dripper_distance):
    num_drippers = math.floor(dripper_length / dripper_spacing)
    num_dripper = math.floor(fuzhu_sub_length / dripper_distance) * num_drippers
    return num_dripper


# 流量计算函数
def calculate_flow_rates(dripper_min_flow, dripper_length, fuzhu_sub_length, dripper_distance, lgz0, lgz1, lgz2):
    dripper_flow = dripper_min_flow / 3600000
    num_drippers = math.floor(dripper_length / dripper_spacing)
    lateral_flow = dripper_flow * math.floor(fuzhu_sub_length / dripper_distance) * 2 * lgz0 * num_drippers
    sub_flow = lateral_flow * lgz1
    main_flow = sub_flow * lgz2
    return lateral_flow, sub_flow, dripper_flow, main_flow


# 水头损失计算总函数
def calculate_head(sub_diameter, lateral_diameter, main_diameter, length_x, length_y, dripper_min_flow, dripper_length,
                   fuzhu_sub_length, dripper_distance, lgz0, lgz1, lgz2):
    lateral_flow, sub_flow, dripper_flow, main_flow = calculate_flow_rates(dripper_min_flow, dripper_length,
                                                                           fuzhu_sub_length,
                                                                           dripper_distance, lgz0, lgz1, lgz2)
    lateral_loss = pressure_loss(lateral_diameter, length_x, lateral_flow)
    sub_losses = [pressure_loss(sub_diameter, y, sub_flow) for y in range(50, length_y + 1, 100)]
    sub_loss = sum(sub_losses) / len(sub_losses)
    dripper_loss = 10
    required_head = lateral_loss + sub_loss + dripper_loss
    main_losses = [pressure_loss(main_diameter, y, main_flow) for y in range(lgz2 * 400, 21 * 400 + 1, 100)]
    main_loss = sum(main_losses) / len(main_losses)
    f_lateral, Re_lateral = friction_factor(lateral_diameter, lateral_flow, 1.5e-6)  # 沿程阻力系数
    f_sub, Re_sub = friction_factor(sub_diameter, sub_flow, 1.5e-6)  # 沿程阻力系数
    f_main, Re_main = friction_factor(main_diameter, main_flow, 1.5e-6)
    main_speed = water_speed(main_diameter, main_flow)
    sub_speed = water_speed(sub_diameter, sub_flow)
    lateral_speed = water_speed(lateral_diameter, lateral_flow)
    pressure = (required_head * 1e3 * 9.81) / 1000000
    PRINTA = [required_head, pressure, dripper_loss, lateral_loss, Re_lateral, lateral_loss, lateral_flow,
              lateral_speed, Re_sub, f_sub, sub_loss, sub_flow, sub_speed, Re_main, f_main, main_loss, main_flow,
              main_speed]
    return PRINTA


def shuili(dripper_length, dripper_min_flow, fuzhu_sub_length, field_length, field_wide,
           Soil_bulk_density, field_z, field_p, field_p_old, field_max, field_min, sr, st, ib, nn, work_time, lgz0,
           lgz1, lgz2,
           Full_field_long, Full_field_wide, dripper_distance):
    num_rows = math.floor(fuzhu_sub_length / sr)
    plants_per_row = math.floor(field_length / st)
    total_plants = math.floor(num_rows * plants_per_row)
    num_dripper = math.floor(fuzhu_sub_length / 0.8) * math.floor(dripper_length / dripper_spacing)
    block_number = Full_field_long / field_length * Full_field_wide / field_wide
    fuzhu_number = int(field_wide / fuzhu_sub_length)
    plant = num_dripper / total_plants
    mmax = (Soil_bulk_density * 1000) * field_z * field_p * (field_max - field_min) * field_p_old
    # TMAX = mmax / ib
    m1 = mmax / nn
    if plant <= 1:
        t = (m1 * dripper_spacing * dripper_distance) / dripper_min_flow
    else:
        t = (m1 * sr * st) / plant * dripper_min_flow
    T = t * (fuzhu_number / lgz0) * int(block_number / 2 / lgz1) * int(21 / lgz2) / work_time
    total_water_v = num_dripper * dripper_min_flow * t / 1000
    total_field_s = dripper_length * fuzhu_sub_length
    water_z = total_water_v / total_field_s * 1000
    TMAX = water_z / ib
    PRANTB = [mmax, TMAX, t, T]
    return PRANTB


# 定义评估函数
def evaluate(individual):
    lgz1, lgz2 = individual
    PRINTA = calculate_head(160, 90, 500, 180, 750, 2.1, 50, 40, 0.8, 1, lgz1, lgz2)
    required_head = PRINTA[0]
    dripper_loss = PRINTA[2]
    main_loss = PRINTA[15]
    # 计算所需水头和灌水时间
    PRANTB = shuili(50, 2.1, 40, 100, 200, 1.46, 0.6, 0.75, 0.28, 0.9, 0.7, 0.8, 0.1, 6, 0.95, 20, 1, lgz1, lgz2, 800,
                    800, 0.8)
    T = PRANTB[3]

    # 计算目标函数值（水头比）
    head_ratio = ((required_head + main_loss) / dripper_loss) + T
    # 检查约束条件
    if required_head <= 27:
        if lgz1 % 2 == 0:
            if main_loss <= 23:
                return (head_ratio,)
            else:
                return (float('inf'),)
        else:
            return (float('inf'),)
    else:
        return (float('inf'),)


'''
 if T > TMAX:
    return (float('inf'),)  # 返回一个非常大的值作为惩罚
else:
'''


# 主函数
def main():
    random.seed(10)
    # 设置遗传算法参数
    toolbox: Any = base.Toolbox()
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)
    toolbox.register("attr_lgz1", random.randint, 1, 10)
    toolbox.register("attr_lgz2", random.randint, 1, 20)
    toolbox.register("individual", tools.initCycle, creator.Individual, (toolbox.attr_lgz1, toolbox.attr_lgz2), n=1)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutUniformInt, low=[1, 1], up=[10, 20], indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)
    pop = toolbox.population(n=50)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)
    algorithms.eaSimple(pop, toolbox, cxpb=0.7, mutpb=0.5, ngen=200, stats=stats, halloffame=hof, verbose=True)
    best = hof[0]
    print(f"\nBest solution: lgz1 = {best[0]}, lgz2 = {best[1]}")
    print(f"Best fitness: {best.fitness.values[0]}")
    PRINTA = calculate_head(160, 90, 500, 180, 750, 2.1, 50, 40, 0.8, 1, best[0], best[1])
    PRINTB = shuili(50, 2.1, 40, 100, 200, 1.46, 0.6, 0.75, 0.28, 0.9, 0.7, 0.8, 0.1, 6, 0.95, 20, 1, best[0], best[1],
                    800,
                    800, 0.8)
    required_head = PRINTA[0]
    pressure = PRINTA[1]
    dripper_loss = PRINTA[2]
    lateral_loss = PRINTA[3]
    Re_lateral = PRINTA[4]
    f_lateral = PRINTA[5]
    lateral_flow = PRINTA[6]
    lateral_speed = PRINTA[7]
    Re_sub = PRINTA[8]
    f_sub = PRINTA[9]
    sub_loss = PRINTA[10]
    sub_flow = PRINTA[11]
    sub_speed = PRINTA[12]
    Re_main = PRINTA[13]
    f_main = PRINTA[14]
    main_loss = PRINTA[15]
    main_flow = PRINTA[16]
    main_speed = PRINTA[17]
    mmax = PRINTB[0]
    TMAX = PRINTB[1]
    t = PRINTB[2]
    T = PRINTB[3]
    print(f'最远端滴灌带所需水头: {required_head:.3f} m')
    print(f'最远端滴灌带所需压力: {pressure:.3f} MPa')
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
    print(f"支管雷诺数: {Re_main:.3f}")
    print(f"支管沿程阻力系数: {f_main:.3f}")
    print(f"支管水头损失: {main_loss:.3f} m")
    print(f"支管轮灌最大流量: {main_flow:.3f} m³/s")
    print(f"支管轮灌最大流速: {main_speed:.3f} m/s")  # 开启多少条斗管
    print(f"最大净灌水定额: {mmax:.3f}mm")
    print(f"设计灌水周期: {TMAX:.3f}天")
    print(f"一次灌水延续时间：{t:.3f}h")
    print(f"完成所有灌水所需时间：{T:.3f}天")


if __name__ == "__main__":
    main()
