import math

dripper_spacing = 0.3


def dripper_max_length(dripper_min_flow, dripper_length, sub_diameter):
    x = 0.6
    k = 0.2
    h = 10
    hl = k / x * (1 + 0.15 * ((1 - x) / x) * k)  # 允许水头偏差率（%）
    hl_max = h * (1 + hl)  # 最大水头损失（m）
    nm = ((5.466 * hl_max * (sub_diameter ** 4.75)) / (1.2*dripper_spacing * (2*(dripper_min_flow * (dripper_length / dripper_spacing)) ** 1.75))) ** 0.364
    Nm = math.ceil(nm)
    L = Nm * dripper_spacing
    if L >= dripper_length:
        return 1, L
    else:
        return 0, L


def main():
    dripper_min_flow = 2.1
    dripper_length = 50
    sub_diameter = 90
    sign, L = dripper_max_length(dripper_min_flow, dripper_length, sub_diameter)
    print(sign)
    print(L)


if __name__ == "__main__":
    main()
