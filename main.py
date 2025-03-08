import os
import sys
import argparse
import logging
import time
import subprocess
import threading
import psutil
from pathlib import Path
import re

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("optimization_manager.log"),
        logging.StreamHandler(sys.stdout)
    ]
)


def create_directory_structure():
    """确保目录结构存在"""
    # 创建PSO和GA目录（如果不存在）
    Path("PSO").mkdir(exist_ok=True)
    Path("GA").mkdir(exist_ok=True)

    # 检查文件是否都在正确位置
    required_files = {
        "PSO/PSO.py": "梳齿布局PSO算法",
        "PSO/shuangPSO.py": "丰字布局PSO算法",
        "GA/NSGA.py": "梳齿布局NSGA-II算法",
        "GA/shuangNSGA.py": "丰字布局NSGA-II算法"
    }

    missing_files = []
    for file_path, description in required_files.items():
        if not Path(file_path).exists():
            missing_files.append(f"{file_path} ({description})")

    if missing_files:
        logging.error("以下必需文件不存在，请确保它们在正确的位置:")
        for missing_file in missing_files:
            logging.error(f"  - {missing_file}")
        return False

    return True


def modify_algorithm_parameters(file_path, node_count, lgz1, lgz2):
    """修改算法文件中的参数"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()

        # 检查文件类型（梳齿还是丰字）
        is_shuang = "shuang" in file_path.lower()

        # 修改创建IrrigationSystem的节点数
        if is_shuang:
            # 丰字布局文件
            code = re.sub(r'irrigation_system = IrrigationSystem\(\s*\n\s*node_count=\d+',
                          f'irrigation_system = IrrigationSystem(\n            node_count={node_count}',
                          code)
        else:
            # 梳齿布局文件
            code = re.sub(r'irrigation_system = IrrigationSystem\(\s*\n\s*node_count=\d+',
                          f'irrigation_system = IrrigationSystem(\n            node_count={node_count}',
                          code)

        # 修改lgz1和lgz2参数
        code = re.sub(r'best_lgz1, best_lgz2 = \d+, \d+',
                      f'best_lgz1, best_lgz2 = {lgz1}, {lgz2}',
                      code)

        # 性能优化：调整显示效率
        code = re.sub(r'if self.show_dynamic_plots and (iteration|generation) % 5 == 0:',
                      r'if self.show_dynamic_plots and \1 % 10 == 0:',
                      code)

        # 创建临时文件
        temp_file = f"temp_{os.path.basename(file_path)}"
        with open(temp_file, 'w', encoding='utf-8') as f:
            f.write(code)

        return temp_file
    except Exception as e:
        logging.error(f"修改文件 {file_path} 参数时出错: {str(e)}")
        return None


def run_optimization(algorithm_file, node_count, lgz1, lgz2):
    """运行单个优化算法"""
    try:
        # 获取算法名称用于显示
        if "PSO" in algorithm_file:
            if "shuang" in algorithm_file:
                algo_name = "PSO丰字布局"
            else:
                algo_name = "PSO梳齿布局"
        else:
            if "shuang" in algorithm_file:
                algo_name = "NSGA-II丰字布局"
            else:
                algo_name = "NSGA-II梳齿布局"

        logging.info(f"准备运行 {algo_name} (节点: {node_count}, lgz1: {lgz1}, lgz2: {lgz2})")

        # 修改参数并创建临时文件
        temp_file = modify_algorithm_parameters(algorithm_file, node_count, lgz1, lgz2)
        if not temp_file:
            logging.error(f"无法为 {algo_name} 创建临时文件")
            return None

        logging.info(f"已创建临时文件: {temp_file}")

        # 获取Python解释器路径
        python_executable = sys.executable

        # 设置环境变量以提高性能
        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = "2"  # 限制OpenMP线程数
        env["MKL_NUM_THREADS"] = "2"  # 限制Intel MKL线程数
        env["PYTHONIOENCODING"] = "utf-8"  # 设置Python IO编码
        # 在Windows上添加额外编码设置
        if os.name == 'nt':
            env["PYTHONUTF8"] = "1"  # 强制使用UTF-8

        # 启动进程
        process = subprocess.Popen(
            [python_executable, temp_file],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding='utf-8',  # 明确指定编码
            errors='replace',  # 遇到解码错误时替换为替代字符
            bufsize=1  # 行缓冲
        )

        logging.info(f"已启动 {algo_name} 进程 (PID: {process.pid})")

        # 设置进程优先级
        try:
            p = psutil.Process(process.pid)
            p.nice(psutil.NORMAL_PRIORITY_CLASS)
        except Exception as e:
            logging.warning(f"设置进程优先级失败: {str(e)}")

        # 创建线程来读取并记录输出
        def read_output(stream, prefix, log_func):
            try:
                for line in stream:
                    line = line.strip()
                    if line:
                        log_func(f"[{prefix}] {line}")
            except UnicodeDecodeError as e:
                log_func(f"[{prefix}] 输出流解码错误: {e}")
                # 尝试以二进制模式读取剩余内容，避免中断
                try:
                    for binary_line in stream.buffer:
                        log_func(f"[{prefix}] [二进制数据]")
                except:
                    pass

        stdout_thread = threading.Thread(
            target=read_output,
            args=(process.stdout, algo_name, logging.info),
            daemon=False
        )
        stderr_thread = threading.Thread(
            target=read_output,
            args=(process.stderr, algo_name, logging.error),
            daemon=False
        )

        stdout_thread.start()
        stderr_thread.start()

        return {
            "process": process,
            "temp_file": temp_file,
            "algo_name": algo_name,
            "stdout_thread": stdout_thread,
            "stderr_thread": stderr_thread
        }
    except Exception as e:
        logging.error(f"运行 {algorithm_file} 时出错: {str(e)}")
        return None


def check_system_resources():
    """检查系统资源"""
    cpu_count = psutil.cpu_count(logical=True)
    available_memory = psutil.virtual_memory().available / (1024 * 1024 * 1024)  # GB
    return cpu_count, available_memory


def get_interactive_params():
    """交互式获取参数"""
    params = {}

    print("\n== 交互式参数输入 ==")
    print("(按Enter键使用默认值)")

    # 梳齿布局参数
    print("\n梳齿布局参数:")
    try:
        nodes_input = input("  节点数量 [默认:23]: ").strip()
        params['nodes_shuzi'] = int(nodes_input) if nodes_input else 23

        lgz1_input = input("  轮灌组参数lgz1 [默认:6]: ").strip()
        params['lgz1_shuzi'] = int(lgz1_input) if lgz1_input else 6

        lgz2_input = input("  轮灌组参数lgz2 [默认:4]: ").strip()
        params['lgz2_shuzi'] = int(lgz2_input) if lgz2_input else 4
    except ValueError:
        print("  输入无效，将使用默认值")
        params['nodes_shuzi'] = params.get('nodes_shuzi', 23)
        params['lgz1_shuzi'] = params.get('lgz1_shuzi', 6)
        params['lgz2_shuzi'] = params.get('lgz2_shuzi', 4)

    # 丰字布局参数
    print("\n丰字布局参数:")
    try:
        nodes_input = input("  节点数量 [默认:32]: ").strip()
        params['nodes_fengzi'] = int(nodes_input) if nodes_input else 32

        lgz1_input = input("  轮灌组参数lgz1 [默认:8]: ").strip()
        params['lgz1_fengzi'] = int(lgz1_input) if lgz1_input else 8

        lgz2_input = input("  轮灌组参数lgz2 [默认:2]: ").strip()
        params['lgz2_fengzi'] = int(lgz2_input) if lgz2_input else 2
    except ValueError:
        print("  输入无效，将使用默认值")
        params['nodes_fengzi'] = params.get('nodes_fengzi', 32)
        params['lgz1_fengzi'] = params.get('lgz1_fengzi', 8)
        params['lgz2_fengzi'] = params.get('lgz2_fengzi', 2)

    # 执行选项
    print("\n执行选项:")
    sequential_input = input("  顺序执行算法? (y/n) [默认:n]: ").strip().lower()
    params['sequential'] = sequential_input == 'y'

    timeout_input = input("  执行超时时间(秒，0表示不限时间) [默认:0]: ").strip()
    try:
        params['timeout'] = int(timeout_input) if timeout_input else 0
    except ValueError:
        print("  输入无效，将使用默认值")
        params['timeout'] = 0

    return params


def main():
    """主函数"""
    start_time = time.time()

    # 检查系统资源
    cpu_count, available_memory = check_system_resources()

    # 解析命令行参数
    parser = argparse.ArgumentParser(description='灌溉系统优化算法管理器')

    # 梳齿布局参数
    parser.add_argument('--nodes-shuzi', type=int, help='梳齿布局节点数量')
    parser.add_argument('--lgz1-shuzi', type=int, help='梳齿布局轮灌组参数lgz1')
    parser.add_argument('--lgz2-shuzi', type=int, help='梳齿布局轮灌组参数lgz2')

    # 丰字布局参数
    parser.add_argument('--nodes-fengzi', type=int, help='丰字布局节点数量')
    parser.add_argument('--lgz1-fengzi', type=int, help='丰字布局轮灌组参数lgz1')
    parser.add_argument('--lgz2-fengzi', type=int, help='丰字布局轮灌组参数lgz2')

    # 执行选项
    parser.add_argument('--sequential', action='store_true', help='顺序执行算法而不是并行执行')
    parser.add_argument('--timeout', type=int, help='执行超时时间（秒，0表示不限时间）')
    parser.add_argument('-n', '--no-interactive', action='store_true', help='不使用交互式输入，使用默认参数')

    # 解析命令行参数
    cmd_args = parser.parse_args()

    # 检查是否需要交互式输入
    if not cmd_args.no_interactive and (
            cmd_args.nodes_shuzi is None or
            cmd_args.lgz1_shuzi is None or
            cmd_args.lgz2_shuzi is None or
            cmd_args.nodes_fengzi is None or
            cmd_args.lgz1_fengzi is None or
            cmd_args.lgz2_fengzi is None or
            cmd_args.timeout is None
    ):
        interactive_params = get_interactive_params()
    else:
        interactive_params = {}

    # 创建最终参数对象
    class Args:
        pass

    args = Args()

    # 设置参数值（命令行参数优先，然后是交互式输入，最后是默认值）
    args.nodes_shuzi = cmd_args.nodes_shuzi if cmd_args.nodes_shuzi is not None else interactive_params.get(
        'nodes_shuzi', 23)
    args.lgz1_shuzi = cmd_args.lgz1_shuzi if cmd_args.lgz1_shuzi is not None else interactive_params.get('lgz1_shuzi',
                                                                                                         6)
    args.lgz2_shuzi = cmd_args.lgz2_shuzi if cmd_args.lgz2_shuzi is not None else interactive_params.get('lgz2_shuzi',
                                                                                                         4)

    args.nodes_fengzi = cmd_args.nodes_fengzi if cmd_args.nodes_fengzi is not None else interactive_params.get(
        'nodes_fengzi', 32)
    args.lgz1_fengzi = cmd_args.lgz1_fengzi if cmd_args.lgz1_fengzi is not None else interactive_params.get(
        'lgz1_fengzi', 8)
    args.lgz2_fengzi = cmd_args.lgz2_fengzi if cmd_args.lgz2_fengzi is not None else interactive_params.get(
        'lgz2_fengzi', 2)

    args.sequential = cmd_args.sequential if cmd_args.sequential else interactive_params.get('sequential', False)
    args.timeout = cmd_args.timeout if cmd_args.timeout is not None else interactive_params.get('timeout', 0)

    # 打印程序标题
    print("\n" + "=" * 60)
    print("   灌溉系统优化算法管理器")
    print("=" * 60)

    # 检查目录结构
    if not create_directory_structure():
        print("\n错误: 目录结构检查失败，请确保所有必需的文件都存在。")
        return

    # 打印系统信息
    print(f"\n系统资源信息:")
    print(f"  - CPU 核心数: {cpu_count}")
    print(f"  - 可用内存: {available_memory:.1f} GB")
    print(f"  - 运行模式: {'顺序' if args.sequential else '并行'}")

    # 打印运行参数
    print(f"\n运行参数:")
    print(f"  梳齿布局参数:")
    print(f"    - 节点数量: {args.nodes_shuzi}")
    print(f"    - 轮灌组参数lgz1: {args.lgz1_shuzi}")
    print(f"    - 轮灌组参数lgz2: {args.lgz2_shuzi}")
    print(f"  丰字布局参数:")
    print(f"    - 节点数量: {args.nodes_fengzi}")
    print(f"    - 轮灌组参数lgz1: {args.lgz1_fengzi}")
    print(f"    - 轮灌组参数lgz2: {args.lgz2_fengzi}")

    # 设置运行的优化算法列表
    algorithms = [
        {"file": "PSO/PSO.py", "type": "shuzi", "name": "PSO梳齿布局"},
        {"file": "PSO/shuangPSO.py", "type": "fengzi", "name": "PSO丰字布局"},
        {"file": "GA/NSGA.py", "type": "shuzi", "name": "NSGA-II梳齿布局"},
        {"file": "GA/shuangNSGA.py", "type": "fengzi", "name": "NSGA-II丰字布局"}
    ]

    # 运行优化算法
    running_processes = []

    print("\n开始运行优化算法...")

    try:
        if args.sequential:
            # 顺序执行
            for algo in algorithms:
                print(f"\n正在运行 {algo['name']}...")

                # 根据布局类型选择对应的参数
                if algo['type'] == 'shuzi':
                    nodes = args.nodes_shuzi
                    lgz1 = args.lgz1_shuzi
                    lgz2 = args.lgz2_shuzi
                else:
                    nodes = args.nodes_fengzi
                    lgz1 = args.lgz1_fengzi
                    lgz2 = args.lgz2_fengzi

                # 运行算法
                process_info = run_optimization(algo['file'], nodes, lgz1, lgz2)

                if process_info:
                    print(f"  {algo['name']} 已启动，等待完成...")
                    process_info['process'].wait()  # 等待当前进程完成
                    print(f"  {algo['name']} 已完成！")

                    # 清理临时文件
                    if os.path.exists(process_info['temp_file']):
                        os.remove(process_info['temp_file'])

                else:
                    print(f"  {algo['name']} 启动失败！")
        else:
            # 并行执行
            print("正在并行启动所有算法...")

            for algo in algorithms:
                # 根据布局类型选择对应的参数
                if algo['type'] == 'shuzi':
                    nodes = args.nodes_shuzi
                    lgz1 = args.lgz1_shuzi
                    lgz2 = args.lgz2_shuzi
                else:
                    nodes = args.nodes_fengzi
                    lgz1 = args.lgz1_fengzi
                    lgz2 = args.lgz2_fengzi

                # 运行算法
                process_info = run_optimization(algo['file'], nodes, lgz1, lgz2)

                if process_info:
                    print(f"  {algo['name']} 已启动！")
                    running_processes.append(process_info)
                    time.sleep(1)  # 短暂延迟，避免资源竞争
                else:
                    print(f"  {algo['name']} 启动失败！")

            # 等待所有进程完成或超时
            if running_processes:
                print("\n所有算法已启动，正在等待完成...")
                timeout_time = time.time() + args.timeout if args.timeout > 0 else None

                # 显示进度
                while running_processes:
                    completed = []
                    for proc_info in running_processes:
                        return_code = proc_info['process'].poll()
                        if return_code is not None:  # 进程已结束
                            print(f"  {proc_info['algo_name']} 已完成 (返回代码: {return_code})")
                            completed.append(proc_info)

                    # 从运行列表中移除已完成的进程
                    for proc_info in completed:
                        running_processes.remove(proc_info)
                        # 清理临时文件
                        if os.path.exists(proc_info['temp_file']):
                            os.remove(proc_info['temp_file'])

                    # 检查是否超时
                    if timeout_time and time.time() > timeout_time:
                        print("\n超时！正在终止剩余进程...")
                        for proc_info in running_processes:
                            print(f"  正在终止 {proc_info['algo_name']}...")
                            proc_info['process'].terminate()
                        break

                    # 如果还有正在运行的进程，等待一段时间
                    if running_processes:
                        # 显示运行状态
                        print(f"\r正在运行: {', '.join([p['algo_name'] for p in running_processes])}, "
                              f"已运行时间: {int(time.time() - start_time)}秒", end="")
                        time.sleep(5)

                print("\n所有算法执行完成！")

    except KeyboardInterrupt:
        print("\n用户中断，正在停止所有进程...")
        for proc_info in running_processes:
            try:
                proc_info['process'].terminate()
                print(f"  已停止 {proc_info['algo_name']}")
            except:
                pass

    finally:
        # 确保清理所有临时文件
        for proc_info in running_processes:
            if os.path.exists(proc_info['temp_file']):
                try:
                    os.remove(proc_info['temp_file'])
                except:
                    pass

    # 汇总结果
    end_time = time.time()
    total_time = end_time - start_time

    print("\n优化结果汇总:")
    result_files = []

    # 检查PSO结果文件
    pso_result_file = "optimization_results_PSO_DAN.txt"
    pso_shuang_result_file = "optimization_results_PSO_SHUANG.txt"

    if os.path.exists(pso_result_file):
        file_size = os.path.getsize(pso_result_file) / 1024  # KB
        print(f"  PSO梳齿布局: 结果文件 {pso_result_file} ({file_size:.1f} KB)")
        result_files.append((pso_result_file, "PSO梳齿布局"))
    else:
        print(f"  PSO梳齿布局: 未找到结果文件")

    if os.path.exists(pso_shuang_result_file):
        file_size = os.path.getsize(pso_shuang_result_file) / 1024  # KB
        print(f"  PSO丰字布局: 结果文件 {pso_shuang_result_file} ({file_size:.1f} KB)")
        result_files.append((pso_shuang_result_file, "PSO丰字布局"))
    else:
        print(f"  PSO丰字布局: 未找到结果文件")

    # 检查NSGA-II结果文件
    nsga_result_file = "optimization_results_NSGAⅡ_DAN.txt"
    nsga_shuang_result_file = "optimization_results_NSGAⅡ_SHUANG.txt"

    if os.path.exists(nsga_result_file):
        file_size = os.path.getsize(nsga_result_file) / 1024  # KB
        print(f"  NSGA-II梳齿布局: 结果文件 {nsga_result_file} ({file_size:.1f} KB)")
        result_files.append((nsga_result_file, "NSGA-II梳齿布局"))
    else:
        print(f"  NSGA-II梳齿布局: 未找到结果文件")

    if os.path.exists(nsga_shuang_result_file):
        file_size = os.path.getsize(nsga_shuang_result_file) / 1024  # KB
        print(f"  NSGA-II丰字布局: 结果文件 {nsga_shuang_result_file} ({file_size:.1f} KB)")
        result_files.append((nsga_shuang_result_file, "NSGA-II丰字布局"))
    else:
        print(f"  NSGA-II丰字布局: 未找到结果文件")

    # 查找并显示图像文件
    print("\n生成的图像文件:")
    image_files = []
    for file in os.listdir("."):
        if file.endswith(".png") and ("PSO" in file or "NSGA" in file):
            image_files.append(file)

    if image_files:
        for image_file in image_files:
            file_size = os.path.getsize(image_file) / 1024  # KB
            print(f"  {image_file} ({file_size:.1f} KB)")
    else:
        print("  未找到生成的图像文件")

    # 比较算法性能（如果结果文件存在）
    if result_files:
        system_costs = []
        variance_values = []

        for result_file, algo_name in result_files:
            try:
                with open(result_file, 'r', encoding='utf-8') as f:
                    content = f.read()

                    # 提取系统总成本
                    cost_match = re.search(r"系统总成本: (\d+\.\d+) 元", content)
                    if cost_match:
                        cost = float(cost_match.group(1))

                        # 提取压力均方差
                        variance_match = re.search(r"系统整体压力均方差: (\d+\.\d+)", content)
                        if variance_match:
                            variance = float(variance_match.group(1))
                            system_costs.append((algo_name, cost))
                            variance_values.append((algo_name, variance))
            except Exception as e:
                print(f"  无法解析 {result_file}: {str(e)}")

        if system_costs:
            # 按成本排序
            system_costs.sort(key=lambda x: x[1])
            print("\n成本排名 (从低到高):")
            for i, (algo_name, cost) in enumerate(system_costs, 1):
                print(f"  {i}. {algo_name}: {cost:.2f} 元")

            # 按均方差排序
            variance_values.sort(key=lambda x: x[1])
            print("\n压力均方差排名 (从低到高):")
            for i, (algo_name, variance) in enumerate(variance_values, 1):
                print(f"  {i}. {algo_name}: {variance:.2f}")

            # 计算综合排名
            combined_ranking = {}
            for algo_name, _ in system_costs:
                cost_rank = [a[0] for a in system_costs].index(algo_name) + 1
                variance_rank = [a[0] for a in variance_values].index(algo_name) + 1
                combined_ranking[algo_name] = cost_rank + variance_rank

            combined_ranking = sorted(combined_ranking.items(), key=lambda x: x[1])
            print("\n综合排名 (考虑成本和均方差):")
            for i, (algo_name, rank) in enumerate(combined_ranking, 1):
                print(f"  {i}. {algo_name}")

    print(f"\n总运行时间: {total_time:.1f}秒 ({total_time / 60:.1f}分钟)")
    print("\n" + "=" * 60)
    print("                  优化管理器执行完毕")
    print("=" * 60)


if __name__ == "__main__":
    main()