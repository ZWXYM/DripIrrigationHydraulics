import random
import math
import numpy as np
from deap import base, creator, tools, algorithms
import tkinter as tk
from tkinter import ttk, messagebox
import json
import csv

dripper_spacing = 0.3
DATAA = []


def create_deap_types():
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)


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
    """计算轮灌组分配

    Args:
        lgz2: 每个轮灌组包含的管道数量（必须为偶数）

    Returns:
        list: 轮灌组信息列表
    """
    if lgz2 % 2 != 0:
        raise ValueError("轮灌组数量必须为偶数")

    groups = []
    nodes_per_group = lgz2 // 2  # 每组需要的节点数（因为每个节点连接上下两条管道）
    remaining_nodes = list(range(1, 33))  # 所有需要灌溉的节点
    group_id = 0

    while remaining_nodes:
        # 如果剩余节点数量小于一个完整组所需的节点数
        if len(remaining_nodes) < nodes_per_group:
            # 将所有剩余节点放入最后一组
            group = {
                'group_id': group_id,
                'nodes': remaining_nodes.copy(),
                'is_special': True,
                'sub_pipes_count': len(remaining_nodes) * 2  # 每个节点连接2条支管
            }
            groups.append(group)
            break

        # 正常分组（从两端取节点）
        front_nodes = remaining_nodes[:nodes_per_group // 2]  # 从前面取节点
        back_nodes = remaining_nodes[-nodes_per_group // 2:]  # 从后面取节点
        group_nodes = front_nodes + back_nodes

        group = {
            'group_id': group_id,
            'nodes': sorted(group_nodes),
            'is_special': False,
            'sub_pipes_count': len(group_nodes) * 2  # 每个节点连接2条支管
        }
        groups.append(group)

        # 从剩余节点列表中移除已分配的节点
        for node in group_nodes:
            remaining_nodes.remove(node)

        group_id += 1

    return groups


# 新增：考虑对称开启的节点水头计算
def calculate_node_heads_symmetric(lgz1, lgz2, main_diameter):
    """计算节点水头"""
    print("开始计算节点水头...")
    flow = calculate_flow_rates(lgz1, lgz2)
    main_flow = flow[3]
    input_head = DATAA[17]
    lateral_length = DATAA[7]
    segment_length = lateral_length * 2

    # 获取轮灌组信息
    irrigation_groups = calculate_irrigation_groups(lgz2)
    print(f"计算得到 {len(irrigation_groups)} 个轮灌组")

    # 创建节点和管段列表
    nodes = []
    segments = []

    # 计算每个sub管道分得的流量
    flow_per_sub = main_flow / lgz2
    print(f"每条支管分得流量: {flow_per_sub:.6f} m³/s")

    # 初始化所有节点信息
    for i in range(33):  # 0-32号节点
        node_info = {
            'node_id': i,
            'distance': i * segment_length,
            'head': None,
            'flow': None,
            'is_active': False
        }
        nodes.append(node_info)

    # 标记活跃节点
    active_nodes = set()
    for group in irrigation_groups:
        for node in group['nodes']:
            active_nodes.add(node)
            nodes[node]['is_active'] = True
    print(f"标记了 {len(active_nodes)} 个活跃节点")

    # 计算每段管道的流量和水头
    current_flow = main_flow
    current_head = input_head

    print("开始计算管段水力特性...")
    for i in range(32):  # 32个管段
        nodes[i]['flow'] = current_flow
        nodes[i]['head'] = current_head

        # 计算到下一个节点的水头损失
        segment_loss = pressure_loss(main_diameter, segment_length, current_flow)
        current_head -= segment_loss

        # 如果当前节点是活跃的，减少相应的流量（每个活跃节点分走两份流量，对应上下两条支管）
        if nodes[i]['is_active']:
            current_flow -= flow_per_sub * 2

        segment_info = {
            'segment_id': i,
            'start_node': i,
            'end_node': i + 1,
            'length': segment_length,
            'flow': current_flow,
            'head_loss': segment_loss,
            'diameter': main_diameter,
            'is_active_start': nodes[i]['is_active'],
            'is_active_end': nodes[i + 1]['is_active']
        }
        segments.append(segment_info)

        if i % 8 == 0:  # 每计算8个管段输出一次进度
            print(f"已完成管段 {i + 1}/32 的计算")

    # 设置最后一个节点的信息
    nodes[-1]['flow'] = current_flow
    nodes[-1]['head'] = current_head

    print("节点水头计算完成！")
    return nodes, segments, irrigation_groups


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


def guanjing(PRINTA, PRINTB, lgz1):
    """管径优化函数"""
    main_diameter = DATAA[6]
    length = DATAA[16] * 2 * DATAA[10]
    flow_rate = PRINTA[11] * lgz1

    diameter = main_diameter
    step = 50
    min_diameter = 250
    max_diameter = DATAA[6]
    last_diameter = diameter
    iteration_count = 0
    max_iterations = 100

    while min_diameter <= diameter <= max_diameter and iteration_count < max_iterations:
        head_loss = pressure_loss(diameter, length, flow_rate)
        end_head = DATAA[17] - head_loss

        if abs(end_head - PRINTA[0]) < 0.1:
            break

        if end_head > PRINTA[0]:
            new_diameter = max(diameter - step, min_diameter)
            if new_diameter == diameter:
                break
            diameter = new_diameter
        else:
            new_diameter = min(diameter + step, max_diameter)
            if new_diameter == diameter:
                break
            diameter = new_diameter

        if diameter == last_diameter:
            break

        last_diameter = diameter
        iteration_count += 1

    final_head_loss = pressure_loss(diameter, length, flow_rate)
    final_end_head = DATAA[17] - final_head_loss

    if final_end_head < DATAA[13]:
        diameter = max_diameter
        final_end_head = DATAA[17] - pressure_loss(diameter, length, flow_rate)

    new_head_loss = pressure_loss(diameter, length, flow_rate)
    f_new, Re_new = friction_factor(diameter, flow_rate, 1.5e-6)
    new_speed = water_speed(diameter, flow_rate)

    PRINTB[0] = Re_new
    PRINTB[1] = f_new
    PRINTB[2] = new_head_loss
    PRINTB[4] = new_speed
    PRINTB[5] = final_end_head
    return diameter


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
        """优化函数，增加了进度输出"""
        print("\n开始优化过程...")
        random.seed(42)
        pop = self.toolbox.population(n=200)
        hof = tools.HallOfFame(1)

        def safe_mean(x):
            return np.nanmean([val for val in x if val != float('inf') and val != float('-inf')])

        def safe_std(x):
            return np.nanstd([val for val in x if val != float('inf') and val != float('-inf')])

        def safe_min(x):
            valid_values = [val for val in x if val != float('inf') and val != float('-inf')]
            return np.nan if not valid_values else min(valid_values)

        def safe_max(x):
            valid_values = [val for val in x if val != float('inf') and val != float('-inf')]
            return np.nan if not valid_values else max(valid_values)

        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", safe_mean)
        stats.register("std", safe_std)
        stats.register("min", safe_min)
        stats.register("max", safe_max)

        try:
            print("开始遗传算法迭代...")
            algorithms.eaSimple(pop, self.toolbox,
                                cxpb=0.9,
                                mutpb=0.2,
                                ngen=30,
                                stats=stats,
                                halloffame=hof,
                                verbose=True)

            if not hof or len(hof) == 0:
                raise ValueError("优化未能找到有效解")

            best = hof[0]
            print(f"\n找到最优解: lgz1={best[0]}, lgz2={best[1]}")
            print(f"适应度值: {best.fitness.values[0]}")

            if best.fitness.values[0] == float('inf'):
                raise ValueError("未找到满足约束条件的解")

            print("\n计算最优解的详细结果...")
            PRINTA = calculate_sub_head(best[0], best[1])
            PRINTB = calculate_main_head(DATAA[6], best[0], best[1])
            PRINTC = lgz_num_count(best[0], best[1])

            print("\n开始优化主管道各段管径...")
            optimized_segments, groups = optimize_main_pipe_segments_symmetric(best[0], best[1])
            print("优化完成！")

            return best, PRINTA, PRINTB, PRINTC, optimized_segments, groups

        except Exception as e:
            print(f"\n优化过程中出现错误: {str(e)}")
            messagebox.showerror("优化错误", f"优化过程中发生错误: {str(e)}")
            raise

    @staticmethod
    def export_results(optimized_segments, irrigation_groups, lateral_length):
        """导出优化结果到CSV文件（静态方法）
        Args:
            optimized_segments: 优化后的管段信息列表
            irrigation_groups: 轮灌组信息列表
            lateral_length: 农管长度
        Returns:
            bool: 导出是否成功
        """
        try:
            # 导出管段信息
            segments_data = []
            for segment in optimized_segments:
                distance = segment['start_node'] * lateral_length * 2
                flow = segment['flow']
                velocity = water_speed(segment['optimal_diameter'], flow)

                segments_data.append({
                    '管段编号': segment['segment_id'],
                    '起点节点': segment['start_node'],
                    '终点节点': segment['end_node'],
                    '距起点(m)': distance,
                    '管径(mm)': segment['optimal_diameter'],
                    '流量(m³/s)': flow,
                    '流速(m/s)': velocity,
                    '水头损失(m)': segment['head_loss'],
                    '起点水头(m)': segment['start_head'],
                    '终点水头(m)': segment['end_head'],
                    '节点状态': "开启" if segment['is_active_start'] else "关闭"
                })

            # 创建导出文件的文件名
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"irrigation_results_{timestamp}.csv"

            # 导出为CSV
            with open(filename, 'w', newline='', encoding='utf-8-sig') as f:
                # 写入管段信息
                writer = csv.DictWriter(f, fieldnames=segments_data[0].keys())
                writer.writeheader()
                writer.writerows(segments_data)

                # 写入空行分隔
                f.write("\n\n")

                # 写入轮灌组信息
                f.write("轮灌组信息\n")
                for group in irrigation_groups:
                    f.write(f"轮灌组 {group['group_id'] + 1}\n")
                    f.write(f"北边开启节点,{','.join(map(str, sorted(group['north_nodes'])))}\n")
                    f.write(f"南边开启节点,{','.join(map(str, sorted(group['south_nodes'])))}\n")
                    if group['is_special']:
                        f.write("注: 该组包含额外节点以减少轮灌组数量\n")
                    f.write("\n")

            messagebox.showinfo("导出成功", f"数据已成功导出到文件：\n{filename}")
            print(f"数据已导出到：{filename}")
            return True

        except Exception as e:
            messagebox.showerror("导出错误", f"导出数据时发生错误：\n{str(e)}")
            print(f"导出数据时发生错误：{str(e)}")
            return False

    def show_results(self, results):
        """修改显示逻辑，更清晰地展示轮灌组信息"""
        print("\n开始生成结果显示...")
        best, PRINTA, PRINTB, PRINTC, optimized_segments, irrigation_groups = results

        result_window = tk.Toplevel(self)
        result_window.title("优化结果")
        result_window.geometry("1000x800")

        # 创建带滚动条的文本框
        frame = ttk.Frame(result_window)
        frame.pack(fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        text = tk.Text(frame, wrap=tk.WORD, yscrollcommand=scrollbar.set)
        text.pack(fill=tk.BOTH, expand=True)
        scrollbar.config(command=text.yview)

        # 基本信息显示
        print("显示基本优化结果...")
        text.insert(tk.END, "=== 基本优化结果 ===\n\n")
        text.insert(tk.END, f"每斗管上开启的农管数量: {best[0]}\n")
        text.insert(tk.END, f"每支管上开启的斗管数量: {best[1]}\n")
        text.insert(tk.END, f"完成灌溉预计时间: {PRINTC[3]:.2f} 天\n")
        text.insert(tk.END, f"最佳适应度: {best.fitness.values[0]:.4f}\n\n")

        # 轮灌组信息显示
        print("显示轮灌组信息...")
        text.insert(tk.END, "=== 轮灌组分配 ===\n\n")
        for group in irrigation_groups:
            text.insert(tk.END, f"轮灌组 {group['group_id'] + 1}:\n")
            text.insert(tk.END, f"包含节点: {sorted(group['nodes'])}\n")
            text.insert(tk.END, f"灌溉支管数量: {group['sub_pipes_count']}\n")
            if group['is_special']:
                text.insert(tk.END, "（该组为特殊组，包含剩余节点）\n")
            text.insert(tk.END, "\n")

        # Main管道分段优化结果
        print("显示管道分段优化结果...")
        text.insert(tk.END, "=== Main管道分段优化结果 ===\n\n")
        header = f"{'管段编号':^8} {'起点':^8} {'终点':^8} {'距起点(m)':^10} {'管径(mm)':^10} "
        header += f"{'流量(m³/s)':^12} {'流速(m/s)':^10} {'水头损失(m)':^12} {'起点水头(m)':^12} {'终点水头(m)':^12} {'节点状态':^10}\n"
        text.insert(tk.END, header)
        text.insert(tk.END, "=" * 130 + "\n")

        print("生成管段详细信息...")
        for segment in optimized_segments:
            distance = segment['start_node'] * DATAA[7] * 2
            flow = segment['flow']
            velocity = water_speed(segment['optimal_diameter'], flow)
            status = "开启" if segment['is_active_start'] else "关闭"

            line = f"{segment['segment_id']:^8d} {segment['start_node']:^8d} {segment['end_node']:^8d} "
            line += f"{distance:^10.1f} {segment['optimal_diameter']:^10.1f} {flow:^12.6f} "
            line += f"{velocity:^10.2f} {segment['head_loss']:^12.2f} {segment['start_head']:^12.2f} "
            line += f"{segment['end_head']:^12.2f} {status:^10s}\n"
            text.insert(tk.END, line)

        # 显示最后一个节点信息
        print("添加最后节点信息...")
        last_distance = len(optimized_segments) * DATAA[7] * 2
        last_status = "开启" if optimized_segments[-1]['is_active_end'] else "关闭"
        last_line = f"{len(optimized_segments):^8d} {optimized_segments[-1]['end_node']:^8d} {'-':^8s} "
        last_line += f"{last_distance:^10.1f} {'-':^10s} {'-':^12s} {'-':^10s} {'-':^12s} {'-':^12s} "
        last_line += f"{optimized_segments[-1]['end_head']:^12.2f} {last_status:^10s}\n"
        text.insert(tk.END, last_line)

        # 详细计算结果
        print("显示详细计算结果...")
        text.insert(tk.END, "\n=== 详细计算结果 ===\n\n")
        for label, value, unit in [
            ('最远端滴灌带所需水头', PRINTA[0], 'm'),
            ('最远端滴灌带所需压力', PRINTA[1], 'MPa'),
            ('滴灌带水头', PRINTA[2], 'm'),
            ('单条农管入口所需水头', PRINTA[3] + PRINTA[2], 'm'),
            ('农管雷诺数', PRINTA[4], ''),
            ('农管沿程阻力系数', PRINTA[5], ''),
            ('农管水头损失', PRINTA[3], 'm'),
            ('农管轮灌最大流量', PRINTA[6], 'm³/s'),
            ('农管轮灌最大流速', PRINTA[7], 'm/s'),
            ('斗管雷诺数', PRINTA[8], ''),
            ('斗管沿程阻力系数', PRINTA[9], ''),
            ('斗管水头损失', PRINTA[10], 'm'),
            ('斗管轮灌最大流量', PRINTA[11], 'm³/s'),
            ('斗管第一段轮灌最大流速', PRINTA[12], 'm/s'),
            ('斗管第二段轮灌最大流速', PRINTA[13], 'm/s'),
            ('支管雷诺数', PRINTB[0], ''),
            ('支管沿程阻力系数', PRINTB[1], ''),
            ('支管总水头损失', PRINTB[2], 'm'),
            ('支管轮灌最大流量', PRINTB[3], 'm³/s'),
            ('支管轮灌最大流速', PRINTB[4], 'm/s'),
        ]:
            text.insert(tk.END, f"{label}: {value:.3f} {unit}\n")
        print("结果显示完成！")

        # 修改导出按钮的创建，使用静态方法
        export_button = ttk.Button(
            frame,
            text="导出数据",
            command=lambda: self.export_results(optimized_segments, irrigation_groups, DATAA[7])
        )
        export_button.pack(pady=10)

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
