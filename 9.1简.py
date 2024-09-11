import math
from pylab import mpl
import random

mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False
dripper_distance = 0.8


# 计算流速
def water_speed(diameter, flow):
    d = diameter / 1000
    speed = flow / ((d / 2) ** 2 * math.pi)
    return speed


# 摩擦损失系数计算
def friction_factor(diameter, flow_rate):
    d = diameter / 1000
    Re = 1000 * water_speed(diameter, flow_rate) * d / 1.004e-3
    # 雷诺数1000是密度，1.004e-3是粘滞系数pas
    epsilon = 1.5e-6  # 一般管道的相对粗糙度
    A = (epsilon / (3.7 * d)) ** 1.11 + (5.74 / Re) ** 0.9
    f = 0.25 / (math.log10(A) ** 2)  # 达西-魏斯巴赫公式计算摩擦损失系数
    return f, Re


# 管段水头损失计算
def pressure_loss(diameter, length, flow_rate):
    f, Re = friction_factor(diameter, flow_rate)
    d = diameter / 1000  # 转换为米
    v = water_speed(diameter, flow_rate)  # 流速(m/s)
    h_f = f * (length / d) * (v ** 2 / (2 * 9.81))  # 达西-魏斯巴赫公式计算压力损失(m)
    return h_f


# 计算流量
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


if __name__ == "__main__":
    calculate_head(160, 90, 180, 750, 2.1, 50, 0.3, 40, 1, 4, 12)
    shuili(50, 0.3, 2.1, 40, 100,
           200, 1.46, 600,
           0.7, 0.9, 0.8, 0.8, 0.1, 8, 0.9, 15, 1, 4, 12, 800, 800)
