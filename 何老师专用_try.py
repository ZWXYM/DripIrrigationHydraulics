import random
import math
from deap import base, creator, tools, algorithms
import tkinter as tk
from tkinter import ttk, messagebox
import json

dripper_spacing = 0.3
DATAA = []


def create_deap_types():
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)


# 流速计算函数，输入管径mm、流量m3/s
def water_speed(diameter, flow_rate):
    """计算流速，处理流量为0的情况"""
    if flow_rate == 0:
        return 0
    d = diameter / 1000
    speed = flow_rate / ((d / 2) ** 2 * math.pi)
    return speed


# 雷诺数及沿程阻力系数计算函数
def friction_factor(diameter, flow_rate, pipe_roughness):
    """计算摩阻系数，处理流量为0的情况"""
    if flow_rate == 0:
        return 0, 0  # 流量为0时，摩阻系数和雷诺数均为0

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
    """计算水头损失，处理流量为0的情况"""
    if flow_rate == 0:
        return 0

    f, Re = friction_factor(diameter, flow_rate, 1.5e-6)
    d = diameter / 1000
    v = water_speed(diameter, flow_rate)
    h_f = f * (length / d) * (v ** 2 / (2 * 9.81))
    return h_f


# 流量计算函数
def calculate_flow_rates(lgz1, lgz2):
    dripper_min_flow = DATAA[0]
    dripper_length = DATAA[1]
    fuzhu_sub_length = DATAA[2]
    dripper_distance = DATAA[3]
    lgz0 = DATAA[15]
    dripper_flow = dripper_min_flow / 3600000
    num_drippers = math.floor(dripper_length / dripper_spacing)
    lateral_flow = dripper_flow * math.floor(fuzhu_sub_length / dripper_distance) * 2 * lgz0 * num_drippers
    sub_flow = lateral_flow * lgz1
    main_flow = sub_flow * lgz2
    flow = [lateral_flow, sub_flow, dripper_flow, main_flow]
    return flow


def calculate_sub_head(lgz1, lgz2):
    flow = calculate_flow_rates(lgz1, lgz2)
    lateral_flow, sub_flow, dripper_flow = flow[0:3]
    lateral_diameter = DATAA[5]
    lateral_length = DATAA[7]
    sub_diameter = DATAA[4]
    sub_length = DATAA[8]

    lateral_losss = [pressure_loss(lateral_diameter, y, lateral_flow) for y in range(0, int(lateral_length) + 1, 1)]
    lateral_loss = sum(lateral_losss) / len(lateral_losss)

    sub_diameter_b = DATAA[4] - 20
    sub_length_a = sub_length / 3
    sub_length_b = (sub_length * 2) / 3

    sub_loss_as = [pressure_loss(sub_diameter, y, sub_flow) for y in range(0, int(sub_length_a) + 1, 100)]
    sub_loss_a = sum(sub_loss_as) / len(sub_loss_as)
    sub_loss_bs = [pressure_loss(sub_diameter_b, y, sub_flow) for y in range(0, int(sub_length_b) + 1, 100)]
    sub_loss_b = sum(sub_loss_bs) / len(sub_loss_bs)

    sub_loss = sub_loss_a + sub_loss_b
    dripper_loss = 10
    required_head = lateral_loss + sub_loss + dripper_loss

    f_lateral, Re_lateral = friction_factor(lateral_diameter, lateral_flow, 1.5e-6)
    f_sub, Re_sub = friction_factor(sub_diameter, sub_flow, 1.5e-6)
    sub_speed_a = water_speed(sub_diameter, sub_flow)
    sub_speed_b = water_speed(sub_diameter_b, sub_flow)
    lateral_speed = water_speed(lateral_diameter, lateral_flow)
    pressure = (required_head * 1e3 * 9.81) / 1000000

    sub_PRINT = [required_head, pressure, dripper_loss, lateral_loss, Re_lateral, lateral_loss, lateral_flow,
                 lateral_speed, Re_sub, f_sub, sub_loss, sub_flow, sub_speed_a, sub_speed_b]
    return sub_PRINT


# 新增：计算轮灌组分配
def calculate_irrigation_groups(lgz2):
    """计算轮灌组分配，保证末端节点，优先使用前面的节点
    Args:
        lgz2: 每个轮灌组包含的管道数量（必须为偶数）
    Returns:
        list: 轮灌组信息列表
    """
    if lgz2 % 2 != 0:
        raise ValueError("轮灌组数量必须为偶数")

    groups = []
    nodes_per_group = lgz2 // 2  # 每组需要的节点数（因为每个节点连接上下两条管道）

    # 初始化节点池，分为前部和后部，包括16号节点
    front_nodes = list(range(1, 17))  # 1-16号节点
    back_nodes = list(range(32, 16, -1))  # 32-17号节点，从后往前
    group_id = 0

    # 持续分组直到节点用完
    while front_nodes or back_nodes:
        # 如果剩余节点不足一组
        remaining_nodes = len(front_nodes) + len(back_nodes)
        if remaining_nodes < nodes_per_group:
            # 将所有剩余节点放入最后一组
            group = {
                'group_id': group_id,
                'nodes': sorted(front_nodes + back_nodes),  # 确保节点编号有序
                'is_special': True,
                'sub_pipes_count': remaining_nodes * 2  # 每个节点连接2条支管
            }
            groups.append(group)
            break

        group_nodes = []
        # 从末端取一个节点
        if back_nodes:
            group_nodes.append(back_nodes.pop(0))  # 取最大的节点号

        # 剩余节点从前面取
        front_needed = nodes_per_group - len(group_nodes)
        while front_needed > 0 and front_nodes:
            group_nodes.append(front_nodes.pop(0))
            front_needed -= 1

        # 如果前面节点不够，从后面补充
        while len(group_nodes) < nodes_per_group and back_nodes:
            group_nodes.append(back_nodes.pop(0))

        group = {
            'group_id': group_id,
            'nodes': sorted(group_nodes),  # 确保节点编号有序
            'is_special': False,
            'sub_pipes_count': len(group_nodes) * 2  # 每个节点连接2条支管
        }
        groups.append(group)
        group_id += 1

    return groups


def calculate_group_flow_profile(group_nodes, total_nodes, sub_flow):
    """计算指定轮灌组的流量分布
    Args:
        group_nodes: 活跃节点列表
        total_nodes: 总节点数
        sub_flow: 单个支管的流量
    Returns:
        dict: 每个管段的流量
    """
    flow_profile = {}
    active_nodes = set(group_nodes)
    current_flow = sub_flow * len(group_nodes) * 2  # 初始流量

    # 从0到total_nodes-1计算每个管段的流量
    for i in range(total_nodes):
        flow_profile[i] = max(0, current_flow)  # 确保流量不为负
        # 只有当前节点是活跃节点时，才减少流量
        if i in active_nodes:
            current_flow = max(0, current_flow - sub_flow * 2)  # 确保流量不为负

    return flow_profile


def optimize_pipe_diameters(first_group_flow_profile, main_diameter, lateral_length):
    """基于第一轮灌组的流量优化管径
    Args:
        first_group_flow_profile: 第一轮灌组的流量分布
        main_diameter: 初始管径
        lateral_length: 农管长度
    Returns:
        dict: 每个管段的优化管径
    """
    segment_length = lateral_length * 2
    quarter_point = 8  # 32个管段的1/4点
    three_quarter_point = 24  # 32个管段的3/4点
    step = 50  # 管径优化步长
    min_diameter = 250  # 最小管径限制
    total_segments = len(first_group_flow_profile)

    print(f"\n开始管径优化...")
    print(f"初始管径: {main_diameter}mm")

    # 存储每个管段的优化管径
    optimized_diameters = {}

    # 前1/4段使用初始管径
    for i in range(quarter_point):
        optimized_diameters[i] = main_diameter

    # 优化中间段(1/4到3/4)的管径
    mid_section_diameters = []
    for i in range(quarter_point, three_quarter_point):
        current_flow = first_group_flow_profile[i]
        current_diameter = main_diameter

        # 对当前管段进行优化
        while current_diameter >= min_diameter:
            velocity = water_speed(current_diameter, current_flow)
            head_loss = pressure_loss(current_diameter, segment_length, current_flow)

            if (0.5 <= velocity <= 2.5 and  # 流速限制
                    head_loss <= DATAA[14] / (three_quarter_point - quarter_point)):  # 单段水头损失限制
                break

            current_diameter -= step

        # 确保不小于最小管径
        current_diameter = max(current_diameter, min_diameter)
        optimized_diameters[i] = current_diameter
        mid_section_diameters.append(current_diameter)

    # 找到中间段的最小管径
    min_mid_diameter = min(mid_section_diameters) if mid_section_diameters else min_diameter

    # 后1/4段使用中间段的最小管径
    for i in range(three_quarter_point, total_segments):
        optimized_diameters[i] = min_mid_diameter

    print("\n管径优化结果：")
    print(f"前1/4段 (0-{quarter_point}): {main_diameter}mm")
    print(f"中间段 ({quarter_point}-{three_quarter_point}): 动态优化，最小值 {min_mid_diameter}mm")
    print(f"后1/4段 ({three_quarter_point}-{total_segments}): {min_mid_diameter}mm")

    return optimized_diameters


def calculate_group_hydraulics(group, sub_flow, optimized_diameters, input_head, lateral_length):
    """计算单个轮灌组的水力特性，处理流量为0的情况"""
    segment_length = lateral_length * 2
    total_segments = 32

    # 计算流量分布
    active_nodes = set(group['nodes'])
    segments = []
    current_flow = sub_flow * len(group['nodes']) * 2  # 初始流量
    current_head = input_head

    print(f"\n计算轮灌组 {group['group_id'] + 1} 的水力特性:")
    print(f"初始流量: {current_flow:.6f} m³/s")

    for i in range(total_segments):
        # 使用优化后的管径
        diameter = optimized_diameters[i]
        current_flow = max(0, current_flow)  # 确保流量不为负

        # 计算当前管段的水力特性
        velocity = water_speed(diameter, current_flow)
        head_loss = pressure_loss(diameter, segment_length, current_flow)

        segment = {
            'segment_id': i,
            'distance': i * segment_length,
            'diameter': diameter,
            'flow': current_flow,
            'velocity': velocity,
            'head_loss': head_loss,
            'start_head': current_head,
            'end_head': current_head - head_loss,
            'is_active': i in active_nodes
        }
        segments.append(segment)

        # 打印调试信息
        if i % 8 == 0:  # 每8个管段打印一次
            print(f"管段 {i}: 流量={current_flow:.6f} m³/s, 流速={velocity:.2f} m/s, "
                  f"水头损失={head_loss:.2f} m")

        # 更新水头和流量
        current_head -= head_loss
        if i in active_nodes:
            new_flow = current_flow - sub_flow * 2
            current_flow = max(0, new_flow)  # 确保流量不为负

    return segments


def format_group_results(group, segments):
    """格式化单个轮灌组的结果输出"""
    result = f"\n{'=' * 20} 轮灌组 {group['group_id'] + 1} "
    result += "（特殊组）" if group['is_special'] else ""
    result += f" {'=' * 20}\n"
    result += f"活跃节点: {sorted(group['nodes'])}\n"
    result += f"支管数量: {group['sub_pipes_count']}\n\n"

    # 表头
    header = (f"{'管段编号':^8} {'距起点(m)':^12} {'管径(mm)':^10} {'起点水头(m)':^12} "
              f"{'终点水头(m)':^12} {'流量(m³/s)':^12} {'流速(m/s)':^10} "
              f"{'水头损失(m)':^12} {'节点状态':^10}\n")
    result += header
    result += "=" * 110 + "\n"

    # 管段数据
    for segment in segments:
        line = (f"{segment['segment_id']:^8d} {segment['distance']:^12.1f} "
                f"{segment['diameter']:^10.0f} {segment['start_head']:^12.2f} "
                f"{segment['end_head']:^12.2f} {segment['flow']:^12.6f} "
                f"{segment['velocity']:^10.2f} {segment['head_loss']:^12.2f} "
                f"{'活跃' if segment['is_active'] else '非活跃':^10s}\n")
        result += line

    return result


# 新增：考虑对称开启的节点水头计算
def calculate_node_heads_symmetric(lgz1, lgz2, main_diameter):
    """计算所有轮灌组的水力特性"""
    print("\n开始计算节点水头...")

    # 基本参数
    flow = calculate_flow_rates(lgz1, lgz2)
    main_flow = flow[3]
    input_head = DATAA[17]
    lateral_length = DATAA[7]
    sub_flow = main_flow / lgz2  # 每条支管的设计流量

    # 获取轮灌组信息
    irrigation_groups = calculate_irrigation_groups(lgz2)
    print(f"计算得到 {len(irrigation_groups)} 个轮灌组")

    # 为每个轮灌组计算水力特性
    all_groups_results = []
    for group in irrigation_groups:
        print(f"\n计算轮灌组 {group['group_id'] + 1} 的水力特性...")
        group_segments = calculate_group_hydraulics(
            group, sub_flow, main_diameter, input_head, lateral_length)
        all_groups_results.append((group, group_segments))

    print("\n所有轮灌组计算完成！")
    return all_groups_results


def optimize_segment_diameter(flow, length, required_head_diff, min_diameter=250, max_diameter=600):
    """优化单个管段的管径"""
    diameter = max_diameter
    step = 10  # 更小的步长以获得更精确的结果

    while diameter >= min_diameter:
        head_loss = pressure_loss(diameter, length, flow)
        if head_loss <= required_head_diff:
            # 尝试更小的管径
            diameter -= step
        else:
            # 找到最小可行管径，返回上一个有效值
            diameter += step
            break

    # 确保在范围内
    diameter = max(min_diameter, min(diameter, max_diameter))
    return diameter


def optimize_main_pipe_segments_symmetric(lgz1, lgz2):
    """优化考虑对称开启的管段，添加进度输出"""
    print(f"\n开始管段优化计算...")
    print(f"参数: lgz1={lgz1}, lgz2={lgz2}")

    nodes, segments, groups = calculate_node_heads_symmetric(lgz1, lgz2, DATAA[6])
    print(f"完成节点水头初始计算，共{len(segments)}个管段")

    optimized_segments = []
    current_head = DATAA[17]

    # 创建管段组以处理相同流量的管段
    flow_groups = {}
    for segment in segments:
        flow_key = f"{segment['flow']:.6f}"
        if flow_key not in flow_groups:
            flow_groups[flow_key] = []
        flow_groups[flow_key].append(segment)

    print(f"识别出{len(flow_groups)}个不同流量组")

    # 对每个流量组分别优化管径
    for i, (flow_key, group_segments) in enumerate(flow_groups.items()):
        print(f"\n处理流量组 {i + 1}/{len(flow_groups)}")
        print(f"流量: {float(flow_key):.6f} m³/s")
        print(f"该组包含管段数量: {len(group_segments)}")

        # 找出该组中最大的水头损失
        max_head_loss = max(seg['head_loss'] for seg in group_segments)
        print(f"该组最大水头损失: {max_head_loss:.3f} m")

        # 为该流量确定最优管径
        optimal_diameter = optimize_segment_diameter(
            flow=group_segments[0]['flow'],
            length=group_segments[0]['length'],
            required_head_diff=max_head_loss
        )
        print(f"优化后管径: {optimal_diameter} mm")

        # 使用相同的管径更新该组的所有管段
        for segment in group_segments:
            actual_head_loss = pressure_loss(optimal_diameter, segment['length'], segment['flow'])
            current_head -= actual_head_loss

            optimized_segment = {
                'segment_id': segment['segment_id'],
                'start_node': segment['start_node'],
                'end_node': segment['end_node'],
                'length': segment['length'],
                'flow': segment['flow'],
                'optimal_diameter': optimal_diameter,
                'head_loss': actual_head_loss,
                'start_head': current_head + actual_head_loss,
                'end_head': current_head,
                'is_active_start': nodes[segment['start_node']]['is_active'],
                'is_active_end': nodes[segment['end_node']]['is_active']
            }
            optimized_segments.append(optimized_segment)

    # 按segment_id排序
    optimized_segments.sort(key=lambda x: x['segment_id'])
    print("\n管段优化完成！")

    return optimized_segments, groups


def calculate_main_head(main_diameter, lgz1, lgz2):
    """原有的主管水头计算函数"""
    flow = calculate_flow_rates(lgz1, lgz2)
    main_flow = flow[3]
    input_head = DATAA[17]
    main_losses = [pressure_loss(main_diameter, y, main_flow) for y in range(lgz2 * 400, DATAA[16] * 400 + 1, 100)]
    main_loss = sum(main_losses) / len(main_losses)
    f_main, Re_main = friction_factor(main_diameter, main_flow, 1.5e-6)
    main_speed = water_speed(main_diameter, main_flow)
    final_end_head = input_head - main_loss
    main_PRINT = [Re_main, f_main, main_loss, main_flow, main_speed, final_end_head]
    return main_PRINT


def lgz_num_count(lgz1, lgz2):
    """计算轮灌组数量和时间"""
    fuzhu_num = DATAA[10] / DATAA[2]
    # 计算所需水头和灌水时间
    lgz_num = fuzhu_num * (6 / lgz1) * (DATAA[16] / lgz2)
    t = 2
    worktime = 20
    TMAX = lgz_num * t
    T = TMAX / worktime
    PRINTC = [t, worktime, TMAX, T]
    return PRINTC


def evaluate(individual):
    """评估函数，增加了数值检查"""
    lgz1, lgz2 = map(int, individual)

    # 确保lgz2为偶数
    if lgz2 % 2 != 0:
        return (float('inf'),)

    try:
        PRINTA = calculate_sub_head(lgz1, lgz2)
        PRINTB = calculate_main_head(DATAA[6], lgz1, lgz2)
        PRINTC = lgz_num_count(lgz1, lgz2)

        required_head = PRINTA[0]
        main_loss = PRINTB[2]
        T = PRINTC[3]

        # 检查计算结果是否有效
        if (any(map(math.isnan, [required_head, main_loss, T])) or
                any(map(math.isinf, [required_head, main_loss, T]))):
            return (float('inf'),)

        # 评估指标
        lateral_speed = PRINTA[7]  # 农管流速
        sub_speed = PRINTA[12]  # 斗管流速
        main_speed = PRINTB[4]  # 支管流速

        # 检查流速是否有效
        if (any(map(math.isnan, [lateral_speed, sub_speed, main_speed])) or
                any(map(math.isinf, [lateral_speed, sub_speed, main_speed]))):
            return (float('inf'),)

        # 流速惩罚项(理想流速范围0.5-2.5m/s)
        speed_penalty = 0
        for speed in [lateral_speed, sub_speed, main_speed]:
            if speed < 0.5:
                speed_penalty += (0.5 - speed) * 10
            elif speed > 2.5:
                speed_penalty += (speed - 2.5) * 10

        # 水头损失分布均匀性惩罚项
        loss_distribution = abs(main_loss / (required_head + 0.01) - 0.3)

        # 最终适应度函数
        fitness = (T * 0.5 +  # 灌溉时间权重
                   main_loss * 0.1 +  # 水头损失权重
                   speed_penalty * 0.3 +  # 流速惩罚权重
                   loss_distribution * 0.1)  # 水头损失分布权重

        # 约束条件检查
        if required_head <= DATAA[13]:
            if main_loss <= DATAA[14]:
                return (fitness,)

        return (float('inf'),)

    except (ValueError, ZeroDivisionError, TypeError):
        # 简单地返回无穷大，表示这是一个无效解
        return (float('inf'),)

    # 如果想要记录异常信息，可以使用这个版本：


"""
    try:
        # ... 同上 ...
    except (ValueError, ZeroDivisionError, TypeError) as e:
        print(f"评估函数出现错误: {str(e)}, individual: {individual}")
        return (float('inf'),)
"""


class IrrigationApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("灌溉系统优化")
        self.geometry("700x800")

        self.entries = {}
        self.toolbox = base.Toolbox()

        create_deap_types()
        self.create_widgets()
        self.setup_deap()

    def create_widgets(self):
        frame = ttk.Frame(self, padding="10")
        frame.grid(row=0, column=0, sticky="nsew")
        row = 0
        for label, default, var_type in [
            ('滴灌带滴孔流量(L/h)', 2.5, float),
            ('滴灌带长度(m)', 66, float),
            ('辅助农管长度(m)', 50, float),
            ('滴灌带间距(m)', 0.8, float),
            ('斗管初始管径(mm)', 160, int),
            ('农管管径(mm)', 90, int),
            ('支管初始管径(mm)', 500, int),
            ('农管长度(m)', 150, float),
            ('斗管长度(m)', 350, float),
            ('小地块y方向的长(m)', 0, float),
            ('小地块x方向的长(m)', 150, float),
            ('地块全长(m)', 0, float),
            ('地块全宽(m)', 0, float),
            ('最远端滴灌带所需最小水头(m)', 20, float),
            ('干管水头损失最大值(m)', 50, float),
            ('一条农管上开启的辅助农管条数', 1, int),
            ('支管上斗管数量', 64, int),
            ('支管入口水头', 51.36, float),
        ]:
            ttk.Label(frame, text=label).grid(row=row, column=0, sticky="w")
            entry = ttk.Entry(frame)
            entry.insert(0, str(default))
            entry.grid(row=row, column=1)
            self.entries[label] = (entry, var_type)
            row += 1

        ttk.Button(frame, text="运行优化", command=self.run_optimization).grid(
            row=row, column=0, columnspan=2, pady=10)
        ttk.Button(frame, text="加载预设", command=self.load_preset).grid(
            row=row + 1, column=0, columnspan=2, pady=10)
        ttk.Button(frame, text="保存预设", command=self.save_preset).grid(
            row=row + 2, column=0, columnspan=2, pady=10)

    def setup_deap(self):
        def even_randint():
            # 生成1到10范围内的随机整数
            return random.randint(1, 10)

        def even_randint2():
            # 生成1到10范围内的随机偶数
            return random.randint(1, 10) * 2

        self.toolbox.register("attr_lgz1", even_randint)
        self.toolbox.register("attr_lgz2", even_randint2)  # 使用自定义函数生成偶数

        self.toolbox.register("individual", tools.initCycle, creator.Individual,
                              (self.toolbox.attr_lgz1, self.toolbox.attr_lgz2), n=1)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("evaluate", evaluate)
        self.toolbox.register("mate", tools.cxTwoPoint)
        self.toolbox.register("mutate", tools.mutUniformInt, low=[1, 2], up=[10, 20], indpb=0.2)
        self.toolbox.register("select", tools.selTournament, tournsize=3)

    def run_optimization(self):
        global DATAA
        try:
            DATAA = [entry_type[1](entry_type[0].get()) for entry_type in self.entries.values()]
            results = self.optimize()
            self.show_results(results)
        except ValueError as e:
            messagebox.showerror("错误", f"输入值错误: {str(e)}")
        except Exception as e:
            messagebox.showerror("错误", f"发生错误: {str(e)}")
            import traceback
            traceback.print_exc()

    def optimize(self):
        """优化函数"""
        print("\n开始优化过程...")
        random.seed(42)
        pop = self.toolbox.population(n=200)
        hof = tools.HallOfFame(1)

        try:
            print("开始遗传算法迭代...")
            algorithms.eaSimple(pop, self.toolbox,
                                cxpb=0.9,
                                mutpb=0.2,
                                ngen=30,
                                stats=None,
                                halloffame=hof,
                                verbose=True)

            best = hof[0]
            print(f"\n找到最优解: lgz1={best[0]}, lgz2={best[1]}")

            # 获取轮灌组信息
            irrigation_groups = calculate_irrigation_groups(best[1])

            # 计算基本流量
            flow = calculate_flow_rates(best[0], best[1])
            sub_flow = flow[3] / best[1]  # 单个支管流量

            # 使用第一个轮灌组的流量分布优化管径
            first_group = irrigation_groups[0]
            first_group_flow = {i: sub_flow * len(first_group['nodes']) * 2 for i in range(32)}
            for node in first_group['nodes']:
                for i in range(node + 1, 32):
                    first_group_flow[i] -= sub_flow * 2

            # 获取优化后的管径分布
            optimized_diameters = optimize_pipe_diameters(
                first_group_flow, DATAA[6], DATAA[7])

            # 使用优化后的管径计算每个轮灌组的水力特性
            all_groups_results = []
            for group in irrigation_groups:
                print(f"\n计算轮灌组 {group['group_id'] + 1} {'(特殊组)' if group['is_special'] else ''}")
                print(f"活跃节点: {group['nodes']}")

                group_segments = calculate_group_hydraulics(
                    group, sub_flow, optimized_diameters, DATAA[17], DATAA[7])
                all_groups_results.append((group, group_segments))

            return best, all_groups_results

        except Exception as e:
            print(f"\n优化过程中出现错误: {str(e)}")
            messagebox.showerror("优化错误", f"优化过程中发生错误: {str(e)}")
            raise

    def show_results(self, results):
        """显示优化结果"""
        print("\n开始生成结果显示...")
        best, all_groups_results = results

        result_window = tk.Toplevel(self)
        result_window.title("优化结果")
        result_window.geometry("1200x800")

        # 创建带滚动条的文本框
        frame = ttk.Frame(result_window)
        frame.pack(fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        text = tk.Text(frame, wrap=tk.WORD, yscrollcommand=scrollbar.set)
        text.pack(fill=tk.BOTH, expand=True)
        scrollbar.config(command=text.yview)

        # 显示基本优化结果
        text.insert(tk.END, "=== 基本优化结果 ===\n\n")
        text.insert(tk.END, f"每斗管上开启的农管数量: {best[0]}\n")
        text.insert(tk.END, f"每支管上开启的斗管数量: {best[1]}\n\n")

        # 显示每个轮灌组的详细结果
        for group, segments in all_groups_results:
            group_results = format_group_results(group, segments)
            text.insert(tk.END, group_results)
            text.insert(tk.END, "\n")

        # 添加导出按钮
        def export_results():
            try:
                from datetime import datetime
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"irrigation_results_{timestamp}.txt"

                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(text.get("1.0", tk.END))

                messagebox.showinfo("导出成功", f"结果已导出到文件：\n{filename}")

            except Exception as e:
                messagebox.showerror("导出错误", f"导出数据时发生错误：\n{str(e)}")

        export_button = ttk.Button(frame, text="导出结果", command=export_results)
        export_button.pack(pady=10)

        print("结果显示完成！")

    def load_preset(self):
        try:
            with open("preset.json", "r") as f:
                preset = json.load(f)
            for label, value in preset.items():
                if label in self.entries:
                    self.entries[label][0].delete(0, tk.END)
                    self.entries[label][0].insert(0, str(value))
            messagebox.showinfo("成功", "预设加载成功")
        except FileNotFoundError:
            messagebox.showerror("错误", "未找到预设文件")
        except json.JSONDecodeError:
            messagebox.showerror("错误", "无效的预设文件格式")

    def save_preset(self):
        preset = {label: self.entries[label][1](entry.get())
                  for label, (entry, _) in self.entries.items()}
        with open("preset.json", "w") as f:
            json.dump(preset, f)
        messagebox.showinfo("成功", "预设保存成功")


def main():
    app = IrrigationApp()
    app.mainloop()


if __name__ == "__main__":
    main()
