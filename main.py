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
import csv

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


def modify_algorithm_parameters(file_path, node_count, lgz1, lgz2, rukoushuitou):
    """修改算法文件中的参数"""
    try:
        with open(file_path, 'r', encoding='utf-8-sig') as f:
            code = f.read()

        # 修改IrrigationSystem初始化部分，直接查找并替换整个初始化行
        # 查找系统初始化的常规模式
        pattern1 = re.compile(r'irrigation_system\s*=\s*IrrigationSystem\s*\(\s*\n\s*node_count=\d+[^)]*\)')

        # 如果找到了匹配，进行替换
        if pattern1.search(code):
            code = pattern1.sub(
                f'irrigation_system = IrrigationSystem(\n            node_count={node_count},\n            rukoushuitou={rukoushuitou})',
                code)
        else:
            # 尝试另一种模式：单行初始化
            pattern2 = re.compile(r'irrigation_system\s*=\s*IrrigationSystem\s*\(\s*node_count=\d+[^)]*\)')
            if pattern2.search(code):
                code = pattern2.sub(
                    f'irrigation_system = IrrigationSystem(node_count={node_count}, rukoushuitou={rukoushuitou})', code)
            else:
                # 如果以上模式都不匹配，尝试更宽松的匹配
                pattern3 = re.compile(r'irrigation_system\s*=\s*IrrigationSystem\([^)]*\)')
                if pattern3.search(code):
                    code = pattern3.sub(
                        f'irrigation_system = IrrigationSystem(node_count={node_count}, rukoushuitou={rukoushuitou})',
                        code)

        # 修改lgz1和lgz2参数
        lgz_pattern = re.compile(r'best_lgz1,\s*best_lgz2\s*=\s*\d+,\s*\d+')
        if lgz_pattern.search(code):
            code = lgz_pattern.sub(f'best_lgz1, best_lgz2 = {lgz1}, {lgz2}', code)

        # 修改输出格式，使管段编号右对齐
        # 查找并替换write_line()调用中的格式字符串
        code = re.sub(r'{i:2d}', r'{i:>2d}', code)  # 将左对齐改为右对齐
        code = re.sub(r'f"{i:2d}', r'f"{i:>2d}', code)  # 包含字符串前缀的情况

        # 替换可能存在的其他格式
        code = re.sub(r'编号\s+后端距起点', r'编号    后端距起点', code)  # 确保表头对齐

        # 创建临时文件
        temp_file = f"temp_{os.path.basename(file_path)}"
        with open(temp_file, 'w', encoding='utf-8-sig') as f:
            f.write(code)

        return temp_file
    except Exception as e:
        logging.error(f"修改文件 {file_path} 参数时出错: {str(e)}")
        return None


def run_optimization(algorithm_file, node_count, lgz1, lgz2, rukoushuitou):
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

        logging.info(f"准备运行 {algo_name} (节点: {node_count}, lgz1: {lgz1}, lgz2: {lgz2}, 入口水头: {rukoushuitou})")

        # 修改参数并创建临时文件
        temp_file = modify_algorithm_parameters(algorithm_file, node_count, lgz1, lgz2, rukoushuitou)
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


def convert_txt_to_csv(txt_file):
    """
    将TXT结果文件转换为CSV格式的最终优化版本
    特点:
    - 精确处理节点编号和管径配置表格，保留其原始结构
    - 正确处理灌溉系统表格的所有列
    - 为所有轮灌组添加一致的表头
    - 统一编码处理，防止中文乱码
    """
    if not os.path.exists(txt_file):
        logging.warning(f"文件不存在，无法转换: {txt_file}")
        return None

    try:
        # 创建CSV文件名
        csv_file = txt_file.replace('.txt', '.csv')

        # 读取TXT文件，尝试多种编码
        content = None
        for encoding in ['utf-8', 'gbk', 'gb18030']:
            try:
                with open(txt_file, 'r', encoding=encoding) as f:
                    content = f.read()
                logging.info(f"使用 {encoding} 编码成功读取文件")
                break
            except UnicodeDecodeError:
                continue

        if content is None:
            with open(txt_file, 'rb') as f:
                content = f.read().decode('utf-8', errors='replace')
            logging.info("使用替换模式解码文件内容")

        # 分割为行
        lines = content.splitlines()

        # 打开CSV文件进行写入，使用UTF-8-SIG编码
        with open(csv_file, 'w', encoding='utf-8-sig', newline='') as csvfile:
            writer = csv.writer(csvfile)

            # 记录已找到的表头类型，用于确保为相同类型的表添加一致的表头
            irrigation_system_headers = None
            pipe_diameter_headers = None

            i = 0
            while i < len(lines):
                line = lines[i].strip()

                # 检测管径配置表格
                if "节点编号" in line and "管径" in line:
                    # 如果是第一次遇到管径配置表格，记录表头
                    if pipe_diameter_headers is None:
                        # 根据图片中的表格格式定义标准表头
                        pipe_diameter_headers = ["节点编号", "第一段管径", "第二段管径"]

                    # 写入表头
                    writer.writerow(pipe_diameter_headers)
                    i += 1

                    # 处理单位行（如果存在）
                    if i < len(lines) and "(mm)" in lines[i]:
                        i += 1  # 跳过单位行

                    # 处理分隔行（如果存在）
                    if i < len(lines) and "-----" in lines[i]:
                        i += 1  # 跳过分隔行

                    # 处理数据行
                    while i < len(lines):
                        data_line = lines[i].strip()

                        # 检查是否到达表格结束
                        if not data_line or "-----" in data_line:
                            if "-----" in data_line:
                                i += 1  # 跳过结束分隔线
                            break

                        # 分析数据行
                        # 针对格式 "# 1    140    140" 或 "1    140    140"
                        clean_line = re.sub(r'\s+', ' ', data_line)
                        parts = clean_line.split(' ')

                        # 处理可能以 # 开头的行
                        if parts and parts[0] == '#':
                            parts.pop(0)  # 移除单独的#号
                        elif parts and parts[0].startswith('#'):
                            parts[0] = parts[0][1:].strip()  # 移除#号

                        # 确保数据行格式正确
                        if parts and parts[0].isdigit():
                            row_data = []

                            # 节点编号
                            row_data.append(parts[0] if len(parts) > 0 else "")

                            # 第一段管径
                            row_data.append(parts[1] if len(parts) > 1 else "")

                            # 第二段管径
                            row_data.append(parts[2] if len(parts) > 2 else "")

                            # 写入解析后的数据行
                            writer.writerow(row_data)
                        else:
                            # 非数据行，作为注释写入
                            writer.writerow(['# ' + data_line])

                        i += 1

                    continue  # 继续处理下一行

                # 检测灌溉系统表格（编号、后端距起点等）
                elif "编号" in line and ("后端距起点" in line or "段前启用状态" in line or "水头损失" in line):
                    # 如果是第一次遇到灌溉系统表格，记录表头
                    if irrigation_system_headers is None:
                        # 根据图片和反馈定义标准表头
                        irrigation_system_headers = [
                            "编号", "后端距起点", "管径", "段前启用状态", "流量",
                            "流速", "水头损失", "段前水头压力", "压力富裕"
                        ]

                    # 写入标准表头
                    writer.writerow(irrigation_system_headers)
                    i += 1

                    # 跳过单位行和分隔行
                    while i < len(lines) and ("(" in lines[i] or "---" in lines[i]):
                        i += 1

                    # 处理数据行
                    while i < len(lines):
                        data_line = lines[i].strip()

                        # 检测表格结束
                        if not data_line or "这段" in data_line or "注:" in data_line or "压力均方差" in data_line:
                            if data_line:  # 保留统计信息行
                                writer.writerow(['# ' + data_line])
                            i += 1
                            break

                        # 检查是否是数据行（以数字开头）
                        if re.match(r'^\d+', data_line):
                            # 清理行，确保空格一致性
                            clean_line = re.sub(r'\s+', ' ', data_line)
                            parts = clean_line.split(' ')

                            # 初始化数据行
                            row_data = [""] * len(irrigation_system_headers)

                            # 解析数据行
                            # 编号
                            if len(parts) > 0:
                                row_data[0] = parts[0]

                            # 后端距起点
                            if len(parts) > 1:
                                row_data[1] = parts[1]

                            # 检查是否有星号
                            has_asterisk = '*' in parts
                            asterisk_index = parts.index('*') if has_asterisk else -1

                            # 管径
                            if len(parts) > 2:
                                row_data[2] = parts[2]

                            # 段前启用状态
                            row_data[3] = '*' if has_asterisk else ''

                            # 调整后续列的索引偏移
                            offset = 1 if has_asterisk and asterisk_index > 2 else 0

                            # 流量
                            if len(parts) > 3 + offset:
                                row_data[4] = parts[3 + offset]

                            # 流速
                            if len(parts) > 4 + offset:
                                row_data[5] = parts[4 + offset]

                            # 水头损失
                            if len(parts) > 5 + offset:
                                row_data[6] = parts[5 + offset]

                            # 段前水头压力
                            if len(parts) > 6 + offset:
                                row_data[7] = parts[6 + offset]

                            # 压力富裕
                            if len(parts) > 7 + offset:
                                row_data[8] = parts[7 + offset]

                            # 写入数据行
                            writer.writerow(row_data)
                        else:
                            # 检查是否是新的轮灌组标题行
                            if "轮灌组" in data_line or "灌溉组" in data_line:
                                # 写入轮灌组标题
                                writer.writerow(['# ' + data_line])

                                # 为新的轮灌组写入相同的表头
                                writer.writerow(irrigation_system_headers)
                            else:
                                # 其他非数据行，写为注释
                                writer.writerow(['# ' + data_line])

                        i += 1

                    continue  # 继续处理下一行

                # 检测到分隔线标记可能开始一个新的表格
                elif "===" in line or "---" in line and len(line) > 10:
                    # 写入分隔线
                    writer.writerow(['# ' + line])
                    i += 1

                    # 检查下一行是否是管径配置表头
                    if i < len(lines) and "节点编号" in lines[i] and "管径" in lines[i]:
                        # 写入管径配置表头
                        if pipe_diameter_headers is None:
                            pipe_diameter_headers = ["节点编号", "第一段管径", "第二段管径"]

                        # 不要在这里写入表头，让下一次循环处理
                        continue

                    # 检查下一行是否是灌溉系统表头
                    elif i < len(lines) and "编号" in lines[i] and "后端距起点" in lines[i]:
                        # 不要在这里写入表头，让下一次循环处理
                        continue

                    continue  # 继续处理下一行

                else:
                    # 非表格行，作为注释写入
                    if line.strip():
                        writer.writerow(['# ' + line.strip()])
                    i += 1

        logging.info(f"已成功将 {txt_file} 转换为 {csv_file}")
        return csv_file

    except Exception as e:
        logging.error(f"转换文件 {txt_file} 为CSV时出错: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        return None


def parse_irrigation_table(table_lines):
    """
    专门解析灌溉系统优化结果表格

    参数:
        table_lines: 表格文本行的列表

    返回:
        headers: 列标题列表
        rows: 数据行列表
    """
    if not table_lines or len(table_lines) < 2:
        return [], []

    # 查找表头行
    header_row_idx = None
    for i, line in enumerate(table_lines):
        if "编号" in line and ("后端距起点" in line or "管长" in line):
            header_row_idx = i
            break

    if header_row_idx is None:
        return [], []

    # 解析表头
    header_line = table_lines[header_row_idx]

    # 定义表格中应该存在的列名
    expected_columns = [
        "编号", "后端距起点", "管长", "管径", "节点启用状态", "流量",
        "节点水头压力", "压力富裕"
    ]

    # 在表头中查找这些列的位置
    column_positions = []
    for col in expected_columns:
        pos = header_line.find(col)
        if pos >= 0:
            column_positions.append((pos, col))

    # 如果没有找到足够的列，尝试使用空格分割
    if len(column_positions) < 3:
        # 根据多个空格分割
        headers = [h.strip() for h in re.split(r'\s{2,}', header_line) if h.strip()]
        rows = []
        for i in range(header_row_idx + 1, len(table_lines)):
            row = [cell.strip() for cell in re.split(r'\s{2,}', table_lines[i]) if cell.strip()]
            if row:
                rows.append(row)
        return headers, rows

    # 按位置排序
    column_positions.sort(key=lambda x: x[0])

    # 提取列名
    headers = [col for _, col in column_positions]

    # 检查是否缺少某些列
    found_columns = set(headers)
    missing_columns = set(expected_columns) - found_columns

    # 特别处理缺少的"流量"列
    if "流量" in missing_columns and "节点启用状态" in found_columns:
        status_idx = headers.index("节点启用状态")
        headers.insert(status_idx + 1, "流量")

    # 特别处理缺少的"压力富裕"列
    if "压力富裕" in missing_columns and "节点水头压力" in found_columns:
        pressure_idx = headers.index("节点水头压力")
        headers.insert(pressure_idx + 1, "压力富裕")

    # 计算列边界
    col_spans = []
    for i, (pos, _) in enumerate(column_positions):
        start = pos
        if i < len(column_positions) - 1:
            end = column_positions[i + 1][0]
        else:
            end = len(header_line) + 10  # 添加额外空间以确保捕获行尾
        col_spans.append((start, end))

    # 处理数据行
    rows = []
    for i in range(header_row_idx + 1, len(table_lines)):
        line = table_lines[i]
        if not line.strip():
            continue

        # 使用列边界切分行
        row_data = []

        # 确保行足够长
        line_padded = line.ljust(max(end for _, end in col_spans))

        for j, (start, end) in enumerate(col_spans):
            cell = line_padded[start:end].strip()

            # 处理星号（根据图片显示，星号是管径后的标记）
            if '*' in cell and headers[j] in ["管径", "管长"]:
                # 保留星号但分离数值
                numbers = re.findall(r'-?\d+\.?\d*', cell)
                if numbers:
                    cell = numbers[0] + ' *'

            row_data.append(cell)

        # 特殊处理"节点启用状态"和"流量"列合并的情况
        if "流量" in headers and "节点启用状态" in headers:
            status_idx = headers.index("节点启用状态")
            flow_idx = headers.index("流量")

            if flow_idx == status_idx + 1 and status_idx < len(row_data):
                # 检查"节点启用状态"列是否包含两个数字
                cell = row_data[status_idx]
                numbers = re.findall(r'-?\d+\.?\d*', cell)

                if len(numbers) >= 2:
                    # 第一个数字是启用状态
                    row_data[status_idx] = numbers[0]

                    # 插入流量值
                    if flow_idx >= len(row_data):
                        row_data.append(numbers[1])
                    else:
                        # 修正流量列
                        row_data[flow_idx] = numbers[1]

        # 特殊处理"节点水头压力"和"压力富裕"列合并的情况
        if "压力富裕" in headers and "节点水头压力" in headers:
            pressure_idx = headers.index("节点水头压力")
            margin_idx = headers.index("压力富裕")

            if margin_idx == pressure_idx + 1 and pressure_idx < len(row_data):
                # 检查"节点水头压力"列是否包含两个数字
                cell = row_data[pressure_idx]
                numbers = re.findall(r'-?\d+\.?\d*', cell)

                if len(numbers) >= 2:
                    # 第一个数字是水头压力
                    row_data[pressure_idx] = numbers[0]

                    # 插入压力富裕值
                    if margin_idx >= len(row_data):
                        row_data.append(numbers[1])
                    else:
                        # 修正压力富裕列
                        row_data[margin_idx] = numbers[1]

        # 确保行数据和表头列数一致
        while len(row_data) < len(headers):
            row_data.append("")

        # 截断过长的行
        if len(row_data) > len(headers):
            row_data = row_data[:len(headers)]

        rows.append(row_data)

    return headers, rows

def read_output(stream, prefix, log_func):
    """
    安全地读取进程输出流并记录到日志
    处理可能的编码错误
    """
    try:
        for line in stream:
            # 确保line是字符串
            if isinstance(line, bytes):
                try:
                    line = line.decode('utf-8')
                except UnicodeDecodeError:
                    try:
                        line = line.decode('gbk')
                    except UnicodeDecodeError:
                        line = line.decode('utf-8', errors='replace')

            line = line.strip()
            if line:
                try:
                    log_func(f"[{prefix}] {line}")
                except UnicodeEncodeError:
                    # 如果仍然出现编码错误，使用ASCII编码记录
                    safe_line = line.encode('ascii', 'replace').decode('ascii')
                    log_func(f"[{prefix}] {safe_line}")
    except Exception as e:
        log_func(f"[{prefix}] 读取输出时出错: {str(e)}")

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

    # 统一参数
    try:
        rukoushuitou_input = input("\n入口水头压力(米) [默认:50]: ").strip()
        params['rukoushuitou'] = float(rukoushuitou_input) if rukoushuitou_input else 50
    except ValueError:
        print("  输入无效，将使用默认值")
        params['rukoushuitou'] = 50

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

    # 通用参数
    parser.add_argument('--rukoushuitou', type=float, help='入口水头压力(米)')

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
            cmd_args.rukoushuitou is None or
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

    args.rukoushuitou = cmd_args.rukoushuitou if cmd_args.rukoushuitou is not None else interactive_params.get(
        'rukoushuitou', 50)

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
    print(f"  通用参数:")
    print(f"    - 入口水头压力: {args.rukoushuitou}米")
    print(f"  梳齿布局参数:")
    print(f"    - 节点数量: {args.nodes_shuzi}")
    print(f"    - 轮灌组参数lgz1: {args.lgz1_shuzi}")
    print(f"    - 轮灌组参数lgz2: {args.lgz2_shuzi}")
    print(f"  丰字布局参数:")
    print(f"    - 节点数量: {args.nodes_fengzi}")
    print(f"    - 轮灌组参数lgz1: {args.lgz1_fengzi}")
    print(f"    - 轮灌组参数lgz2: {args.lgz2_fengzi}")

    print("\n开始运行优化算法...\n")

    try:
        # 创建灌溉系统优化算法列表
        algorithms = [
            {"file": "PSO/PSO.py", "type": "shuzi", "name": "PSO梳齿布局"},
            {"file": "PSO/shuangPSO.py", "type": "fengzi", "name": "PSO丰字布局"},
            {"file": "GA/NSGA.py", "type": "shuzi", "name": "NSGA-II梳齿布局"},
            {"file": "GA/shuangNSGA.py", "type": "fengzi", "name": "NSGA-II丰字布局"}
        ]

        # 运行优化算法
        running_processes = []

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

                rukoushuitou = args.rukoushuitou

                # 运行算法
                process_info = run_optimization(algo['file'], nodes, lgz1, lgz2, rukoushuitou)

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
            print(f"正在并行启动所有算法...")

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

                rukoushuitou = args.rukoushuitou

                # 运行算法
                process_info = run_optimization(algo['file'], nodes, lgz1, lgz2, rukoushuitou)

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
        # 转换为CSV
        csv_file = convert_txt_to_csv(pso_result_file)
        if csv_file:
            print(f"    已转换为CSV: {csv_file}")
    else:
        print(f"  PSO梳齿布局: 未找到结果文件")

    if os.path.exists(pso_shuang_result_file):
        file_size = os.path.getsize(pso_shuang_result_file) / 1024  # KB
        print(f"  PSO丰字布局: 结果文件 {pso_shuang_result_file} ({file_size:.1f} KB)")
        result_files.append((pso_shuang_result_file, "PSO丰字布局"))
        # 转换为CSV
        csv_file = convert_txt_to_csv(pso_shuang_result_file)
        if csv_file:
            print(f"    已转换为CSV: {csv_file}")
    else:
        print(f"  PSO丰字布局: 未找到结果文件")

    # 检查NSGA-II结果文件
    nsga_result_file = "optimization_results_NSGAⅡ_DAN.txt"
    nsga_shuang_result_file = "optimization_results_NSGAⅡ_SHUANG.txt"

    if os.path.exists(nsga_result_file):
        file_size = os.path.getsize(nsga_result_file) / 1024  # KB
        print(f"  NSGA-II梳齿布局: 结果文件 {nsga_result_file} ({file_size:.1f} KB)")
        result_files.append((nsga_result_file, "NSGA-II梳齿布局"))
        # 转换为CSV
        csv_file = convert_txt_to_csv(nsga_result_file)
        if csv_file:
            print(f"    已转换为CSV: {csv_file}")
    else:
        print(f"  NSGA-II梳齿布局: 未找到结果文件")

    if os.path.exists(nsga_shuang_result_file):
        file_size = os.path.getsize(nsga_shuang_result_file) / 1024  # KB
        print(f"  NSGA-II丰字布局: 结果文件 {nsga_shuang_result_file} ({file_size:.1f} KB)")
        result_files.append((nsga_shuang_result_file, "NSGA-II丰字布局"))
        # 转换为CSV
        csv_file = convert_txt_to_csv(nsga_shuang_result_file)
        if csv_file:
            print(f"    已转换为CSV: {csv_file}")
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
                with open(result_file, 'r', encoding='utf-8-sig') as f:
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
