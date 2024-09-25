import cv2
import numpy as np
import math

# 设置参数
width_m, height_m = 400, 200  # 三角形的宽度和高度（米）
circle_radius_m = 0.15  # 圆的半径（米）
scale = 10  # 缩放因子 (1米 = 1000像素)

# 将米转换为像素
width_px = int(width_m * scale)
height_px = int(height_m * scale)
circle_radius_px = max(1, int(circle_radius_m * scale))

# 创建画布
img = np.ones((height_px, width_px, 3), dtype=np.uint8) * 255

# 绘制三角形
cv2.line(img, (0, height_px - 1), (width_px - 1, height_px - 1), (0, 0, 0), 2)
cv2.line(img, (0, height_px - 1), (0, 0), (0, 0, 0), 2)
cv2.line(img, (width_px - 1, height_px - 1), (0, 0), (0, 0, 0), 2)

# 计算圆心的间距
dx = circle_radius_m * 2
dy = circle_radius_m * math.sqrt(3)

# 计算斜率
slope = height_m / width_m

circle_count = 0

# 绘制圆
for i in range(int(width_m / dx) + 1):
    for j in range(int(height_m / dy) + 1):
        x_m = i * dx
        y_m = j * dy

        # 检查圆是否完全在三角形内
        if x_m + circle_radius_m <= width_m and y_m + circle_radius_m <= height_m:
            if y_m + circle_radius_m <= height_m - (x_m * slope):
                x_px = int(x_m * scale)
                y_px = height_px - int(y_m * scale)
                cv2.circle(img, (x_px, y_px), circle_radius_px, (0, 0, 255), -1)
                circle_count += 1

        # 打印进度，但不保存中间图像
        if i % 10 == 0:
            print(f"Progress: {i}/{int(width_m / dx) + 1} columns processed")

    # 保存最终图像
    cv2.imwrite("final_symmetric_triangle_with_circles.png", img)

    print(f"最终对称三角形图像已保存为 final_symmetric_triangle_with_circles.png")
    print(f"三角形内的圆的精确数量: {circle_count}")
