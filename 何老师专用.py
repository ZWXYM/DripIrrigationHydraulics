import random
import math
import numpy as np
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
    # lateral_loss = pressure_loss(lateral_diameter, lateral_length, lateral_flow)
    lateral_losss = [pressure_loss(lateral_diameter, y, lateral_flow) for y in range(0, int(lateral_length) + 1, 1)]
    lateral_loss = sum(lateral_losss) / len(lateral_losss)
    sub_diameter_b = DATAA[4] - 20
    sub_length_a = sub_length / 3
    sub_length_b = (sub_length * 2) / 3
    # sub_loss_a = pressure_loss(sub_diameter, sub_length_a, sub_flow)
    sub_loss_as = [pressure_loss(sub_diameter, y, sub_flow) for y in range(0, int(sub_length_a) + 1, 100)]
    sub_loss_a = sum(sub_loss_as) / len(sub_loss_as)
    sub_loss_bs = [pressure_loss(sub_diameter_b, y, sub_flow) for y in range(0, int(sub_length_b) + 1, 100)]
    sub_loss_b = sum(sub_loss_bs) / len(sub_loss_bs)
    # sub_loss_b = pressure_loss(sub_diameter_b, sub_length_b, sub_flow)
    sub_loss = sub_loss_a + sub_loss_b
    # sub_losses = [pressure_loss(sub_diameter, y, sub_flow) for y in range(50, int(sub_length) + 1, 100)]
    # sub_loss = sum(sub_losses) / len(sub_losses)
    dripper_loss = 10
    required_head = lateral_loss + sub_loss + dripper_loss
    f_lateral, Re_lateral = friction_factor(lateral_diameter, lateral_flow, 1.5e-6)  # 沿程阻力系数
    f_sub, Re_sub = friction_factor(sub_diameter, sub_flow, 1.5e-6)  # 沿程阻力系数
    sub_speed_a = water_speed(sub_diameter, sub_flow)
    sub_speed_b = water_speed(sub_diameter_b, sub_flow)
    lateral_speed = water_speed(lateral_diameter, lateral_flow)
    pressure = (required_head * 1e3 * 9.81) / 1000000
    sub_PRINT = [required_head, pressure, dripper_loss, lateral_loss, Re_lateral, lateral_loss, lateral_flow,
                 lateral_speed, Re_sub, f_sub, sub_loss, sub_flow, sub_speed_a, sub_speed_b]
    return sub_PRINT


# 水头损失计算总函数
def calculate_main_head(main_diameter, lgz1, lgz2):
    flow = calculate_flow_rates(lgz1, lgz2)
    main_flow = flow[3]
    input_head = DATAA[17]
    main_losses = [pressure_loss(main_diameter, y, main_flow) for y in range(lgz2 * 400, DATAA[16] * 400 + 1, 100)]
    main_loss = sum(main_losses) / len(main_losses)
    f_main, Re_main = friction_factor(main_diameter, main_flow, 1.5e-6)
    main_speed = water_speed(main_diameter, main_flow)
    final_end_head = input_head - main_loss
    mian_PRINT = [Re_main, f_main, main_loss, main_flow, main_speed, final_end_head]
    return mian_PRINT


def guanjing(PRINTA, PRINTB, lgz1):
    # 初始化参数
    main_diameter = DATAA[6]  # 使用当前支管管径作为初始值
    length = DATAA[16] * 2 * DATAA[10]  # 管道长度
    flow_rate = PRINTA[11] * lgz1  # 管道流量

    # 初始化搜索参数
    diameter = main_diameter
    step = 50  # 管径调整步长
    min_diameter = 250  # 最小可能管径
    max_diameter = DATAA[6]  # 最大可能管径
    last_diameter = diameter  # 记录上一次的管径
    iteration_count = 0  # 添加迭代计数器
    max_iterations = 100  # 最大迭代次数

    while min_diameter <= diameter <= max_diameter and iteration_count < max_iterations:
        # 计算当前管径的水头损失
        head_loss = pressure_loss(diameter, length, flow_rate)
        end_head = DATAA[17] - head_loss

        if abs(end_head - PRINTA[0]) < 0.1:  # 如果误差在可接受范围内
            break

        if end_head > PRINTA[0]:
            # 水头充足，减小管径
            new_diameter = max(diameter - step, min_diameter)
            if new_diameter == diameter:  # 如果无法继续减小
                break
            diameter = new_diameter
        else:
            # 水头不足，增大管径
            new_diameter = min(diameter + step, max_diameter)
            if new_diameter == diameter:  # 如果无法继续增大
                break
            diameter = new_diameter

        if diameter == last_diameter:  # 如果管径没有变化
            break

        last_diameter = diameter
        iteration_count += 1

    # 确保最终结果满足水头要求
    final_head_loss = pressure_loss(diameter, length, flow_rate)
    final_end_head = DATAA[17] - final_head_loss

    if final_end_head < DATAA[13]:
        diameter = max_diameter  # 确保最终结果满足最小水头要求
        final_end_head = DATAA[17] - pressure_loss(diameter, length, flow_rate)
    # 使用最终确定的管径重新计算水力参数
    new_head_loss = pressure_loss(diameter, length, flow_rate)
    f_new, Re_new = friction_factor(diameter, flow_rate, 1.5e-6)
    new_speed = water_speed(diameter, flow_rate)

    # 更新PRINTA数组中的相关值
    PRINTB[0] = Re_new  # 更新雷诺数
    PRINTB[1] = f_new  # 更新沿程阻力系数
    PRINTB[2] = new_head_loss  # 更新水头损失
    PRINTB[4] = new_speed  # 更新流速
    PRINTB[5] = final_end_head
    return diameter


def lgz_num_count(lgz1, lgz2):
    fuzhu_num = DATAA[10] / DATAA[2]
    # 计算所需水头和灌水时间
    lgz_num = fuzhu_num * (6 / lgz1) * (DATAA[16] / lgz2)
    t = 2
    worktime = 20
    TMAX = lgz_num * t
    T = TMAX / worktime
    PRINTC = [t, worktime, TMAX, T]
    return PRINTC


# 定义评估函数
def evaluate(individual):
    lgz1, lgz2 = map(int, individual)
    PRINTA = calculate_sub_head(lgz1, lgz2)
    PRINTB = calculate_main_head(DATAA[6], lgz1, lgz2)
    PRINTC = lgz_num_count(lgz1, lgz2)

    required_head = PRINTA[0]
    main_loss = PRINTB[2]
    T = PRINTC[3]

    # 增加新的评估指标
    lateral_speed = PRINTA[7]  # 农管流速
    sub_speed = PRINTA[12]  # 斗管流速
    main_speed = PRINTB[4]  # 支管流速

    # 流速惩罚项(理想流速范围0.5-2.5m/s)
    speed_penalty = 0
    for speed in [lateral_speed, sub_speed, main_speed]:
        if speed < 0.5:
            speed_penalty += (0.5 - speed) * 10
        elif speed > 2.5:
            speed_penalty += (speed - 2.5) * 10

    # 水头损失分布均匀性惩罚项
    loss_distribution = abs(main_loss / (required_head + 0.01) - 0.3)  # 理想状态下主管水头损失约占30%

    # 最终的适应度函数
    fitness = (T * 0.5 +  # 灌溉时间权重
               main_loss * 0.1 +  # 水头损失权重
               speed_penalty * 0.3 +  # 流速惩罚权重
               loss_distribution * 0.1)  # 水头损失分布权重

    # 约束条件检查
    if required_head <= DATAA[13]:
        if main_loss <= DATAA[14]:
            return (fitness,)
        else:
            return (float('inf'),)
    else:
        return (float('inf'),)


# 主函数
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

    '''
    0. dripper_min_flow 2.1
    1. dripper_length 50
    2. fuzhu_sub_length 40
    3. dripper_distance 0.8
    4. sub_diameter 160
    5. lateral_diameter 90
    6. main_diameter 500
    7. lateral_length 200
    8. sub_length 750
    9. field_length 100
    10. field_wide 200
    11. Full_field_long 800
    12. Full_field_wide 800
    13. required_head_max 27
    14. main_loss_max 23
    15. lgz0 1
    16. douguan 22
    17. 支管入口处水头
    '''

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

        ttk.Button(frame, text="运行优化", command=self.run_optimization).grid(row=row, column=0, columnspan=2, pady=10)
        ttk.Button(frame, text="加载预设", command=self.load_preset).grid(row=row + 1, column=0, columnspan=2, pady=10)
        ttk.Button(frame, text="保存预设", command=self.save_preset).grid(row=row + 2, column=0, columnspan=2, pady=10)

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

    def setup_deap(self):
        self.toolbox.register("attr_lgz1", random.randint, 1, 10)
        self.toolbox.register("attr_lgz2", random.randint, 1, 20)
        self.toolbox.register("individual", tools.initCycle, creator.Individual,
                              (self.toolbox.attr_lgz1, self.toolbox.attr_lgz2), n=1)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("evaluate", evaluate)
        self.toolbox.register("mate", tools.cxTwoPoint)
        self.toolbox.register("mutate", tools.mutUniformInt, low=[1, 1], up=[10, 20], indpb=0.2)
        self.toolbox.register("select", tools.selTournament, tournsize=3)

    def optimize(self):
        random.seed(42)  # 使用固定的随机种子以获得可重复的结果

        # 增加种群大小和迭代次数
        pop = self.toolbox.population(n=200)
        hof = tools.HallOfFame(1)

        # 添加更多统计信息
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)

        # 调整遗传算法参数
        algorithms.eaSimple(pop, self.toolbox,
                            cxpb=0.9,  # 保持较高的交叉率
                            mutpb=0.2,  # 降低变异率以保持稳定性
                            ngen=30,  # 增加迭代次数
                            stats=stats,
                            halloffame=hof,
                            verbose=True)

        best = hof[0]

        PRINTA = calculate_sub_head(best[0], best[1])
        PRINTB = calculate_main_head(DATAA[6], best[0], best[1])
        PRINTC = lgz_num_count(best[0], best[1])
        DATAA[6] = guanjing(PRINTA, PRINTB, best[1])
        return best, PRINTA, PRINTB, PRINTC

    def show_results(self, results):
        best, PRINTA, PRINTB, PRINTC = results
        result_window = tk.Toplevel(self)
        result_window.title("优化结果")
        result_window.geometry("700x800")

        text = tk.Text(result_window, wrap=tk.WORD)
        text.pack(expand=True, fill=tk.BOTH)
        text.insert(tk.END, f"最佳轮灌组\n")
        text.insert(tk.END, f"每斗管上开启的农管数量:{best[0]}\n")
        text.insert(tk.END, f"每支管上开启的斗管数量:{best[1]}\n")
        text.insert(tk.END, f"完成灌溉预计时间:{PRINTC[3]}\n")
        text.insert(tk.END, f"最佳适应度: {best.fitness.values[0]}\n\n")

        for label, value, unit in [
            ('支管初始管径', DATAA[6], 'mm'),
            ('支管末端水头', PRINTB[5], 'm'),
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
            ('支管水头损失', PRINTB[2], 'm'),
            ('支管轮灌最大流量', PRINTB[3], 'm³/s'),
            ('支管轮灌最大流速', PRINTB[4], 'm/s'),
        ]:
            text.insert(tk.END, f"{label}: {value:.3f} {unit}\n")

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
        preset = {label: self.entries[label][1](entry.get()) for label, (entry, _) in self.entries.items()}
        with open("preset.json", "w") as f:
            json.dump(preset, f)
        messagebox.showinfo("成功", "预设保存成功")


def main():
    app = IrrigationApp()
    app.mainloop()


if __name__ == "__main__":
    main()
