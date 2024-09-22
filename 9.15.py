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


# 计算地块内植物数量，输入地块长度、辅助农管长度、
def calculate_plant_count(fuzhu_field_wide, fuzhu_sub_length, sr, st):
    # 计算行数
    num_rows = math.floor(fuzhu_sub_length / sr)
    # 计算每行植物数
    plants_per_row = math.floor(fuzhu_field_wide / st)  # 将cm转换为m
    # 计算总植物数
    total_plants = math.floor(num_rows * plants_per_row)
    return total_plants


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
def calculate_head(sub_diameter, lateral_diameter, main_diameter, lateral_length, sub_length, dripper_min_flow,
                   dripper_length,
                   fuzhu_sub_length, dripper_distance, lgz0, lgz1, lgz2):
    lateral_flow, sub_flow, dripper_flow, main_flow = calculate_flow_rates(dripper_min_flow, dripper_length,
                                                                           fuzhu_sub_length,
                                                                           dripper_distance, lgz0, lgz1, lgz2)
    lateral_loss = pressure_loss(lateral_diameter, lateral_length, lateral_flow)

    # 修改这里，确保 range() 函数的参数都是整数
    sub_losses = [pressure_loss(sub_diameter, y, sub_flow) for y in range(50, int(sub_length) + 1, 100)]
    sub_loss = sum(sub_losses) / len(sub_losses)
    dripper_loss = 10
    required_head = lateral_loss + sub_loss + dripper_loss

    # 同样修改这里
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
    lgz0 = int(lgz0)
    lgz1 = int(lgz1)
    lgz2 = int(lgz2)
    total_plants = calculate_plant_count(field_length, fuzhu_sub_length, sr, st)
    num_dripper = calculate_dripper_count(dripper_length, fuzhu_sub_length, dripper_distance)
    block_number = Full_field_long / field_length * Full_field_wide / field_wide
    fuzhu_number = round(field_wide / fuzhu_sub_length)
    plant = num_dripper / total_plants
    mmax = (Soil_bulk_density * 1000) * field_z * field_p * (field_max - field_min) * field_p_old
    # TMAX = mmax / ib
    m1 = mmax / nn
    if plant <= 1:
        t = (m1 * dripper_spacing * dripper_distance) / dripper_min_flow
    else:
        t = (m1 * sr * st) / plant * dripper_min_flow
    T = t * (fuzhu_number / lgz0) * math.ceil(block_number / 2 / lgz1) * round(21 / lgz2) / work_time
    total_water_v = num_dripper * dripper_min_flow * t / 1000
    total_field_s = dripper_length * fuzhu_sub_length
    water_z = total_water_v / total_field_s * 1000
    TMAX = water_z / ib
    PRANTB = [mmax, TMAX, t, T]
    return PRANTB


# 定义评估函数
def evaluate(individual):
    lgz1, lgz2 = map(int, individual)
    PRINTA = calculate_head(DATAA[4], DATAA[5], DATAA[6], DATAA[7], DATAA[8], DATAA[0], DATAA[1], DATAA[2], DATAA[3],
                            DATAA[26], lgz1, lgz2)
    required_head = PRINTA[0]
    dripper_loss = PRINTA[2]
    main_loss = PRINTA[15]
    # 计算所需水头和灌水时间
    PRANTB = shuili(DATAA[1], DATAA[0], DATAA[2], DATAA[9], DATAA[10], DATAA[11], DATAA[12], DATAA[13], DATAA[14],
                    DATAA[15], DATAA[16], DATAA[17], DATAA[18], DATAA[19], DATAA[20], DATAA[21], DATAA[26], lgz1, lgz2,
                    DATAA[22],
                    DATAA[23], DATAA[3])
    T = PRANTB[3]

    # 计算目标函数值（水头比）
    head_ratio = ((required_head + main_loss) / dripper_loss) * 0.4 + T * 0.6
    # 检查约束条件
    if required_head <= DATAA[24]:
        if lgz1 % 4 == 0:
            if main_loss <= DATAA[25]:
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
            ('滴灌带滴孔流量(L/h)', 2.1, float),
            ('滴灌带长度(m)', 50, float),
            ('辅助农管长度(m)', 40, float),
            ('滴灌带间距(m)', 0.8, float),
            ('斗管管径(mm)', 160, int),
            ('农管管径(mm)', 90, int),
            ('支管管径(mm)', 500, int),
            ('农管长度(m)', 200, float),
            ('斗管长度(m)', 750, float),
            ('小地块y方向的长(m)', 100, float),
            ('小地块x方向的长(m)', 200, float),
            ('土壤容重(g/cm3)', 1.46, float),
            ('设计浸润深度(m)', 0.6, float),
            ('设计土壤浸润比', 0.75, float),
            ('土壤持水量', 0.28, float),
            ('适宜土壤含水率上限', 0.9, float),
            ('适宜土壤含水率下限', 0.7, float),
            ('植物行距(m)', 0.8, float),
            ('植物一行上株距(m)', 0.1, float),
            ('设计耗水强度(mm)', 6, float),
            ('灌水利用效率', 0.95, float),
            ('日工作时长(h)', 20, float),
            ('地块全长(m)', 800, float),
            ('地块全宽(m)', 800, float),
            ('最远端滴灌带所需最小水头(m)', 27, float),
            ('干管水头损失最大值(m)', 23, float),
            ('一条农管上开启的辅助农管条数', 1, int)
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
        random.seed(10)
        pop = self.toolbox.population(n=100)
        hof = tools.HallOfFame(1)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("min", np.min)
        stats.register("max", np.max)
        algorithms.eaSimple(pop, self.toolbox, cxpb=0.9, mutpb=0.9, ngen=50, stats=stats, halloffame=hof, verbose=True)
        best = hof[0]

        PRINTA = calculate_head(int(DATAA[4]), int(DATAA[5]), int(DATAA[6]), DATAA[7], DATAA[8], DATAA[0], DATAA[1],
                                DATAA[2], DATAA[3],
                                int(DATAA[26]), int(best[0]), int(best[1]))
        PRINTB = shuili(DATAA[1], DATAA[0], DATAA[2], DATAA[9], DATAA[10], DATAA[11], DATAA[12], DATAA[13], DATAA[14],
                        DATAA[15], DATAA[16], DATAA[17], DATAA[18], DATAA[19], DATAA[20], DATAA[21], int(DATAA[26]),
                        int(best[0]),
                        int(best[1]), DATAA[22], DATAA[23], DATAA[3])

        return best, PRINTA, PRINTB

    def show_results(self, results):
        best, PRINTA, PRINTB = results
        result_window = tk.Toplevel(self)
        result_window.title("优化结果")
        result_window.geometry("700x800")

        text = tk.Text(result_window, wrap=tk.WORD)
        text.pack(expand=True, fill=tk.BOTH)

        text.insert(tk.END, f"最佳解决方案: lgz1 = {best[0]}, lgz2 = {best[1]}\n")
        text.insert(tk.END, f"最佳适应度: {best.fitness.values[0]}\n\n")

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
            ('斗管轮灌最大流速', PRINTA[12], 'm/s'),
            ('支管雷诺数', PRINTA[13], ''),
            ('支管沿程阻力系数', PRINTA[14], ''),
            ('支管水头损失', PRINTA[15], 'm'),
            ('支管轮灌最大流量', PRINTA[16], 'm³/s'),
            ('支管轮灌最大流速', PRINTA[17], 'm/s'),
            ('最大净灌水定额', PRINTB[0], 'mm'),
            ('设计灌水周期', PRINTB[1], '天'),
            ('一次灌水延续时间', PRINTB[2], 'h'),
            ('完成所有灌水所需时间', PRINTB[3], '天')
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
11. Soil_bulk_density 1.46
12. field_z 0.6
13. field_p 0.75
14. field_p_old 0.28
15. field_max 0.9
16. field_min 0.8
17. sr 0.8
18. st 0.1
19. ib 6
20. nn 0.95
21. work_time 20
22. Full_field_long 800
23. Full_field_wide 800
24. required_head_max 27
25. main_loss_max 23
26. lgz0 1
    PRINTA = calculate_head(160, 90, 500, 180, 750, 2.1, 50, 40, 0.8, 1, lgz1, lgz2)
    PRANTB = shuili(50, 2.1, 40, 100, 200, 1.46, 0.6, 0.75, 0.28, 0.9, 0.7, 0.8, 0.1, 6, 0.95, 20, 1, lgz1, lgz2, 800,800, 0.8)
    '''
