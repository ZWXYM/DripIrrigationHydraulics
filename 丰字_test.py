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


def water_speed(diameter, flow_rate):
    if flow_rate == 0:
        return 0
    d = diameter / 1000
    speed = flow_rate / ((d / 2) ** 2 * math.pi)
    return speed


def friction_factor(diameter, flow_rate, pipe_roughness):
    if flow_rate == 0:
        return 0, 0

    d = diameter / 1000
    v = water_speed(diameter, flow_rate)
    Re = 1000 * v * d / 1.004e-3
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
    if flow_rate == 0:
        return 0

    f, Re = friction_factor(diameter, flow_rate, 1.5e-6)
    d = diameter / 1000
    v = water_speed(diameter, flow_rate)
    h_f = f * (length / d) * (v ** 2 / (2 * 9.81))
    return h_f


def calculate_flow_rates(lgz1, lgz2):
    dripper_min_flow = DATAA[0]
    dripper_length = DATAA[1]
    fuzhu_sub_length = DATAA[2]
    dripper_distance = DATAA[3]
    lgz0 = DATAA[12]

    dripper_flow = dripper_min_flow / 3600000
    num_drippers = math.floor(dripper_length / dripper_spacing)
    lateral_flow = dripper_flow * math.floor(fuzhu_sub_length / dripper_distance) * 2 * lgz0 * num_drippers
    sub_flow = lateral_flow * lgz1
    main_flow = sub_flow * lgz2

    return [lateral_flow, sub_flow, dripper_flow, main_flow]


def calculate_sub_head(lgz1, lgz2):
    flow = calculate_flow_rates(lgz1, lgz2)
    lateral_flow, sub_flow = flow[0:2]

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

    return [required_head, pressure, dripper_loss, lateral_loss, Re_lateral, lateral_loss,
            lateral_flow, lateral_speed, Re_sub, f_sub, sub_loss, sub_flow, sub_speed_a, sub_speed_b]


def calculate_irrigation_groups(lgz2, total_nodes):
    """计算轮灌组分配，保持原有逻辑但支持自定义节点数量"""
    if lgz2 % 2 != 0:
        raise ValueError("轮灌组数量必须为偶数")

    groups = []
    nodes_per_group = lgz2 // 2
    half_point = total_nodes // 2

    # 创建前部和后部节点列表，保持原有的分配逻辑
    front_nodes = list(range(1, half_point + 1))
    back_nodes = list(range(total_nodes, half_point, -1))
    group_id = 0

    while front_nodes or back_nodes:
        remaining_nodes = len(front_nodes) + len(back_nodes)
        if remaining_nodes < nodes_per_group:
            group = {
                'group_id': group_id,
                'nodes': sorted(front_nodes + back_nodes),
                'is_special': True,
                'sub_pipes_count': remaining_nodes * 2
            }
            groups.append(group)
            break

        group_nodes = []
        if back_nodes:
            group_nodes.append(back_nodes.pop(0))

        front_needed = nodes_per_group - len(group_nodes)
        while front_needed > 0 and front_nodes:
            group_nodes.append(front_nodes.pop(0))
            front_needed -= 1

        while len(group_nodes) < nodes_per_group and back_nodes:
            group_nodes.append(back_nodes.pop(0))

        group = {
            'group_id': group_id,
            'nodes': sorted(group_nodes),
            'is_special': False,
            'sub_pipes_count': len(group_nodes) * 2
        }
        groups.append(group)
        group_id += 1

    return groups


def optimize_pipe_diameters(first_group_flow_profile, main_diameter, lateral_length, total_nodes):
    """
    优化管网中的管段直径

    Args:
        first_group_flow_profile: 第一个轮灌组的流量分布
        main_diameter: 初始管径
        lateral_length: 农管长度
        total_nodes: 总节点数量（总管段数量相等）

    Returns:
        dict: 优化后的管段直径
    """
    segment_length = lateral_length * 2
    quarter_point = total_nodes // 4
    total_segments = total_nodes  # 管段数量等于节点数量
    step = 50
    min_diameter = 250

    print(f"\n开始管径优化...")
    print(f"初始管径: {main_diameter}mm")
    print(f"总节点数: {total_nodes}")
    print(f"总管段数: {total_segments}")

    optimized_diameters = {}

    # 优化每个管段的直径（0到total_nodes-1）
    for i in range(total_segments):
        if i < quarter_point:
            # 前1/4的管段保持原始管径
            optimized_diameters[i] = main_diameter
        else:
            current_flow = first_group_flow_profile.get(i, 0)
            current_diameter = main_diameter

            print(f"\n优化管段 {i} 的管径:")
            print(f"当前流量: {current_flow:.6f} m³/s")

            while current_diameter >= min_diameter:
                velocity = water_speed(current_diameter, current_flow)
                head_loss = pressure_loss(current_diameter, segment_length, current_flow)

                print(f"尝试管径: {current_diameter}mm")
                print(f"流速: {velocity:.2f} m/s")
                print(f"水头损失: {head_loss:.2f} m")

                if (0.5 <= velocity <= 2.5 and
                        head_loss <= DATAA[11] / (total_nodes - quarter_point)):
                    break

                current_diameter -= step

            current_diameter = max(current_diameter, min_diameter)
            optimized_diameters[i] = current_diameter

    return optimized_diameters


def calculate_group_hydraulics(group, sub_flow, optimized_diameters, input_head, lateral_length, total_nodes):
    """
    计算轮灌组的水力特性

    Args:
        group: 轮灌组信息
        sub_flow: 单个支管流量
        optimized_diameters: 优化后的管段直径字典
        input_head: 入口水头
        lateral_length: 农管长度
        total_nodes: 总节点数量

    Returns:
        list: 包含每个管段水力特性的列表
    """
    segment_length = lateral_length * 2
    active_nodes = set(group['nodes'])
    segments = []
    current_flow = sub_flow * len(group['nodes']) * 2
    current_head = input_head

    # 计算所有管段（0到total_nodes-1）的水力特性
    for i in range(total_nodes):
        diameter = optimized_diameters[i]  # 现在可以直接使用索引，因为我们确保了所有管段都有对应的直径
        current_flow = max(0, current_flow)

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

        current_head -= head_loss
        if i in active_nodes:
            current_flow = max(0, current_flow - sub_flow * 2)

    return segments


def calculate_main_head(main_diameter, lgz1, lgz2):
    flow = calculate_flow_rates(lgz1, lgz2)
    main_flow = flow[3]
    input_head = DATAA[14]
    main_losses = [pressure_loss(main_diameter, y, main_flow) for y in range(lgz2 * 400, DATAA[13] * 400 + 1, 100)]
    main_loss = sum(main_losses) / len(main_losses)
    f_main, Re_main = friction_factor(main_diameter, main_flow, 1.5e-6)
    main_speed = water_speed(main_diameter, main_flow)
    final_end_head = input_head - main_loss

    return [Re_main, f_main, main_loss, main_flow, main_speed, final_end_head]


def lgz_num_count(lgz1, lgz2):
    fuzhu_num = DATAA[9] / DATAA[2]
    lgz_num = fuzhu_num * (6 / lgz1) * (DATAA[13] / lgz2)
    t = 2
    worktime = 20
    TMAX = lgz_num * t
    T = TMAX / worktime

    return [t, worktime, TMAX, T]


def evaluate(individual):
    lgz1, lgz2 = map(int, individual)

    if lgz2 % 2 != 0:
        return (float('inf'),)

    try:
        PRINTA = calculate_sub_head(lgz1, lgz2)
        PRINTB = calculate_main_head(DATAA[6], lgz1, lgz2)
        PRINTC = lgz_num_count(lgz1, lgz2)

        required_head = PRINTA[0]
        main_loss = PRINTB[2]
        T = PRINTC[3]

        if (any(map(math.isnan, [required_head, main_loss, T])) or
                any(map(math.isinf, [required_head, main_loss, T]))):
            return (float('inf'),)

        lateral_speed = PRINTA[7]
        sub_speed = PRINTA[12]
        main_speed = PRINTB[4]

        if (any(map(math.isnan, [lateral_speed, sub_speed, main_speed])) or
                any(map(math.isinf, [lateral_speed, sub_speed, main_speed]))):
            return (float('inf'),)

        speed_penalty = 0
        for speed in [lateral_speed, sub_speed, main_speed]:
            if speed < 0.5:
                speed_penalty += (0.5 - speed) * 10
            elif speed > 2.5:
                speed_penalty += (speed - 2.5) * 10

        loss_distribution = abs(main_loss / (required_head + 0.01) - 0.3)

        fitness = (T * 0.5 +
                   main_loss * 0.1 +
                   speed_penalty * 0.3 +
                   loss_distribution * 0.1)

        if required_head <= DATAA[10]:
            if main_loss <= DATAA[11]:
                return (fitness,)

        return (float('inf'),)

    except (ValueError, ZeroDivisionError, TypeError):
        return (float('inf'),)


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

        input_fields = [
            ('滴灌带滴孔流量(L/h)', 2.5, float),
            ('滴灌带长度(m)', 66, float),
            ('辅助农管长度(m)', 50, float),
            ('滴灌带间距(m)', 0.8, float),
            ('斗管初始管径(mm)', 160, int),
            ('农管管径(mm)', 90, int),
            ('支管初始管径(mm)', 500, int),
            ('农管长度(m)', 150, float),
            ('斗管长度(m)', 350, float),
            ('小地块x方向的长(m)', 150, float),
            ('最远端滴灌带所需最小水头(m)', 20, float),
            ('干管水头损失最大值(m)', 50, float),
            ('一条农管上开启的辅助农管条数', 1, int),
            ('支管上斗管数量', 64, int),
            ('支管入口水头', 51.36, float),
        ]

        for label, default, var_type in input_fields:
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
            return random.randint(1, 10)

        def even_randint2():
            return random.randint(1, 10) * 2

        self.toolbox.register("attr_lgz1", even_randint)
        self.toolbox.register("attr_lgz2", even_randint2)

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

            # 首先计算基本流量和水力特性
            flow = calculate_flow_rates(best[0], best[1])
            total_nodes = DATAA[13] // 2  # 由于每个节点连接两条斗管，总节点数是斗管数的一半
            sub_flow = flow[3] / best[1]  # 计算单个支管流量

            PRINTA = calculate_sub_head(best[0], best[1])
            PRINTB = calculate_main_head(DATAA[6], best[0], best[1])
            PRINTC = lgz_num_count(best[0], best[1])

            # 计算轮灌组
            irrigation_groups = calculate_irrigation_groups(best[1], total_nodes)

            # 为第一个轮灌组计算流量分布
            first_group = irrigation_groups[0]
            first_group_flow = {}
            initial_flow = sub_flow * len(first_group['nodes']) * 2

            # 计算每个节点位置的流量
            for i in range(total_nodes):
                if i in first_group['nodes']:
                    first_group_flow[i] = initial_flow
                    initial_flow -= sub_flow * 2
                else:
                    first_group_flow[i] = initial_flow

            # 优化管径
            optimized_diameters = optimize_pipe_diameters(
                first_group_flow, DATAA[6], DATAA[7], total_nodes)

            # 计算每个轮灌组的水力特性
            all_groups_results = []
            for group in irrigation_groups:
                group_segments = calculate_group_hydraulics(
                    group, sub_flow, optimized_diameters, DATAA[14], DATAA[7], total_nodes)
                all_groups_results.append((group, group_segments))

            return best, all_groups_results, PRINTA, PRINTB, PRINTC

        except Exception as e:
            print(f"\n优化过程中出现错误: {str(e)}")
            messagebox.showerror("优化错误", f"优化过程中发生错误: {str(e)}")
            raise

    def show_results(self, results):
        print("\n开始生成结果显示...")
        best, all_groups_results, PRINTA, PRINTB, PRINTC = results

        result_window = tk.Toplevel(self)
        result_window.title("优化结果")
        result_window.geometry("1200x800")

        frame = ttk.Frame(result_window)
        frame.pack(fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        text = tk.Text(frame, wrap=tk.WORD, yscrollcommand=scrollbar.set)
        text.pack(fill=tk.BOTH, expand=True)
        scrollbar.config(command=text.yview)

        # 设置文本标签
        text.tag_configure("active_node", foreground="red")
        text.tag_configure("inactive_node", foreground="black")

        text.insert(tk.END, "=== 基本优化结果 ===\n\n")
        text.insert(tk.END, f"每斗管上开启的农管数量: {best[0]}\n")
        text.insert(tk.END, f"每支管上开启的斗管数量: {best[1]}\n")
        text.insert(tk.END, f"完成灌溉预计时间: {PRINTC[3]:.2f} 天\n")
        text.insert(tk.END, f"最佳适应度: {best.fitness.values[0]:.4f}\n\n")

        text.insert(tk.END, "=== 基础水力计算结果 ===\n\n")
        basic_results = [
            ('斗管入口所需水头', PRINTA[0], 'm'),
            ('斗管入口所需压力', PRINTA[1], 'MPa'),
            ('斗管水头损失', PRINTA[3], 'm'),
            ('斗管流量最大值', PRINTA[6], 'm³/s'),
            ('斗管最大流速', PRINTA[7], 'm/s'),
            ('支管雷诺数', PRINTA[8], ''),
            ('支管沿程阻力系数', PRINTA[9], ''),
            ('支管水头损失', PRINTA[10], 'm'),
            ('支管流量最大值', PRINTA[11], 'm³/s'),
            ('支管第一段最大流速', PRINTA[12], 'm/s'),
            ('支管第二段最大流速', PRINTA[13], 'm/s'),
        ]

        for label, value, unit in basic_results:
            text.insert(tk.END, f"{label}: {value:.3f} {unit}\n")
        text.insert(tk.END, "\n")

        text.insert(tk.END, "=== 轮灌组详细结果 ===\n\n")
        for group_index, (group, segments) in enumerate(all_groups_results):
            text.insert(tk.END, f"轮灌组 {group_index + 1}")
            if group['is_special']:
                text.insert(tk.END, " (特殊组)")
            text.insert(tk.END, "\n")
            text.insert(tk.END, f"活跃节点: {sorted(group['nodes'])}\n")
            text.insert(tk.END, f"支管数量: {group['sub_pipes_count']}\n\n")

            # 绘制管网示意图
            text.insert(tk.END, "管网节点状态示意图:\n")
            text.insert(tk.END, "0")
            for i in range(1, len(segments) + 1):
                text.insert(tk.END, "-")
                if i in group['nodes']:
                    text.insert(tk.END, str(i), "active_node")
                else:
                    text.insert(tk.END, str(i), "inactive_node")
            text.insert(tk.END, "\n\n")

            # 显示管段信息表格
            header = (f"{'管段编号':^8} {'距起点(m)':^12} {'管径(mm)':^10} "
                      f"{'起点水头(m)':^12} {'终点水头(m)':^12} {'流量(m³/s)':^12} "
                      f"{'流速(m/s)':^10} {'水头损失(m)':^12}\n")
            text.insert(tk.END, header)
            text.insert(tk.END, "=" * 90 + "\n")

            for segment in segments:
                line = (f"{segment['segment_id']:^8d} {segment['distance']:^12.1f} "
                        f"{segment['diameter']:^10.0f} {segment['start_head']:^12.2f} "
                        f"{segment['end_head']:^12.2f} {segment['flow']:^12.6f} "
                        f"{segment['velocity']:^10.2f} {segment['head_loss']:^12.2f}\n")
                text.insert(tk.END, line)

            text.insert(tk.END, "\n")

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
