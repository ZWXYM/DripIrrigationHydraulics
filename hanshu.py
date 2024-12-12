import math

dripper_spacing = 0.3
DATAA = []


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


def lgz_num_count(lgz1, lgz2, lgz3, num1, num2):
    lgz_num = math.ceil(4 / lgz3) * math.ceil(num1 / lgz1) * math.ceil(num2 / lgz2)
    t = 2
    worktime = 20
    TMAX = lgz_num * t
    T = TMAX / worktime
    return [t, worktime, TMAX, T]
