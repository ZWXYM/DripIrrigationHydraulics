import numpy as np
from importlib.machinery import SourceFileLoader
import os


def create_solution_directory():
    """创建解决方案目录"""
    if not os.path.exists("pareto_solutions"):
        os.makedirs("pareto_solutions")


def generate_solution_file(solution_id, config, irrigation_system):
    """为每个解决方案生成详细的输出文件"""
    filename = f"pareto_solutions/solution_{solution_id:03d}.txt"

    with open(filename, 'w', encoding='utf-8') as f:
        f.write(f"灌溉管网优化方案 {solution_id}\n")
        f.write("=" * 80 + "\n\n")

        # 基本信息
        f.write("方案基本信息\n")
        f.write("-" * 40 + "\n")
        f.write(f"系统总成本: {config['cost']:,.2f} 元\n")
        f.write(f"压力均方差: {config['variance']:.6f}\n\n")

        # 干管配置
        f.write("干管管径配置\n")
        f.write("-" * 40 + "\n")
        f.write("管段编号    管径(mm)\n")
        for i, diameter in enumerate(config['decoded_configuration']['main_pipe_diameters'], 1):
            f.write(f"{i:4d}         {diameter:4d}\n")
        f.write("\n")

        # 斗管配置
        f.write("斗管管径配置\n")
        f.write("-" * 40 + "\n")
        f.write("节点编号    第一段管径(mm)    第二段管径(mm)\n")
        for i, (first, second) in enumerate(config['decoded_configuration']['submain_diameters'], 1):
            f.write(f"{i:4d}         {first:4d}             {second:4d}\n")

        # 水力计算结果
        f.write("\n水力计算结果\n")
        f.write("-" * 40 + "\n")
        # ... (这里可以添加更多水力计算的详细信息，如果需要)


def generate_index_file(solutions):
    """生成解决方案索引文件"""
    with open("pareto_solutions/index.txt", 'w', encoding='utf-8') as f:
        f.write("帕累托最优解索引\n")
        f.write("=" * 80 + "\n\n")

        # 写入目录
        f.write("目录\n")
        f.write("-" * 80 + "\n")
        f.write("编号    系统成本(元)      压力均方差      文件名\n")
        f.write("-" * 80 + "\n")

        for i, sol in enumerate(solutions, 1):
            f.write(f"{i:3d}     {sol['cost']:12,.2f}    {sol['variance']:12.6f}    solution_{i:03d}.txt\n")

        # 添加统计信息
        f.write("\n统计信息\n")
        f.write("-" * 30 + "\n")
        f.write(f"解的数量: {len(solutions)}\n")
        f.write(f"成本范围: {min(s['cost'] for s in solutions):,.2f} - {max(s['cost'] for s in solutions):,.2f} 元\n")
        f.write(
            f"压力方差范围: {min(s['variance'] for s in solutions):.6f} - {max(s['variance'] for s in solutions):.6f}\n")
        f.write(f"平均成本: {sum(s['cost'] for s in solutions) / len(solutions):,.2f} 元\n")
        f.write(f"平均压力方差: {sum(s['variance'] for s in solutions) / len(solutions):.6f}\n")


def extract_and_save_pareto_data(pareto_front, irrigation_system, pipe_specs):
    """提取并保存帕累托前沿的数据点（包含实际管径）"""
    # 创建解决方案目录
    create_solution_directory()

    solutions = []
    for ind in pareto_front:
        if np.all(np.isfinite(ind.fitness.values)):
            # 解码管径配置
            decoded_config = decode_pipe_configuration(list(ind), irrigation_system, pipe_specs)
            solution = {
                'cost': ind.fitness.values[0],
                'variance': ind.fitness.values[1],
                'raw_configuration': list(ind),
                'decoded_configuration': decoded_config
            }
            solutions.append(solution)

    # 按成本排序
    solutions.sort(key=lambda x: x['cost'])

    # 生成每个解决方案的详细文件
    for i, solution in enumerate(solutions, 1):
        generate_solution_file(i, solution, irrigation_system)

    # 生成索引文件
    generate_index_file(solutions)

    return solutions


def decode_pipe_configuration(individual, irrigation_system, pipe_specs):
    """将配置数组解码为实际的管径尺寸"""
    # (保持原有的decode_pipe_configuration函数代码不变)
    main_indices = individual[:len(irrigation_system.main_pipe) - 1]
    submain_first_indices = individual[len(irrigation_system.main_pipe) - 1:
                                       len(irrigation_system.main_pipe) + len(irrigation_system.submains) - 1]
    submain_second_indices = individual[len(irrigation_system.main_pipe) +
                                        len(irrigation_system.submains) - 1:]

    decoded_config = {
        'main_pipe_diameters': [],  # 干管管径
        'submain_diameters': []  # 斗管管径（包含两段）
    }

    # 解码干管管径（确保管径递减）
    prev_diameter = None
    for index in main_indices:
        available_diameters = [d for d in pipe_specs["main"]["diameters"]
                               if prev_diameter is None or d <= prev_diameter]
        if not available_diameters:
            break

        normalized_index = min(index, len(available_diameters) - 1)
        diameter = available_diameters[normalized_index]
        decoded_config['main_pipe_diameters'].append(diameter)
        prev_diameter = diameter

    # 解码斗管管径
    for first_index, second_index in zip(submain_first_indices, submain_second_indices):
        if decoded_config['main_pipe_diameters']:
            main_connection_diameter = decoded_config['main_pipe_diameters'][0]
        else:
            main_connection_diameter = max(pipe_specs["main"]["diameters"])

        available_first_diameters = [d for d in pipe_specs["submain"]["diameters"]
                                     if d <= main_connection_diameter]
        if not available_first_diameters:
            continue

        normalized_first_index = min(first_index, len(available_first_diameters) - 1)
        first_diameter = available_first_diameters[normalized_first_index]

        available_second_diameters = [d for d in pipe_specs["submain"]["diameters"]
                                      if d <= first_diameter]
        if not available_second_diameters:
            continue

        normalized_second_index = min(second_index, len(available_second_diameters) - 1)
        second_diameter = available_second_diameters[normalized_second_index]

        decoded_config['submain_diameters'].append((first_diameter, second_diameter))

    return decoded_config


if __name__ == "__main__":
    # 加载原始程序
    original_program = SourceFileLoader("original", "NSGA.py").load_module()

    # 设置随机种子以确保结果可重复
    np.random.seed(42)

    # 创建灌溉系统
    irrigation_system = original_program.IrrigationSystem(
        node_count=23,
    )

    # 执行优化并获取帕累托前沿
    print("正在执行优化计算，这可能需要几分钟...")
    pareto_front, _ = original_program.multi_objective_optimization(irrigation_system, 6, 4)

    # 提取并保存数据
    if pareto_front:
        solutions = extract_and_save_pareto_data(pareto_front, irrigation_system, original_program.PIPE_SPECS)
        print(f"已成功提取并保存 {len(solutions)} 个帕累托最优解到 pareto_solutions 目录")
        print("索引文件已保存为 pareto_solutions/index.txt")
    else:
        print("未能获取到帕累托前沿数据")