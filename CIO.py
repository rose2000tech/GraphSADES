import numpy as np

def calculate_importance_coefficient(att):
    """
    计算重要性系数矩阵

    参数:
    att (numpy.ndarray): 图模型的注意力系数的归一化值矩阵，大小为 n x n x 2

    返回:
    numpy.ndarray: 重要性系数矩阵 U，大小为 n x n
    """
    n = att.shape[0]  # 节点总数
    U = np.zeros((n, n))  # 初始化重要性系数矩阵
    total_sum = 0  # 总和初始化为0

    # 第一步：计算每个节点对的平均注意力系数并累加指数值
    for i in range(n):
        for j in range(n):
            s = att[i][j][0] + att[i][j][1]  # 计算两个注意力系数之和
            avg = s / 2  # 计算平均值
            total_sum += np.exp(avg)  # 累加指数值
            U[i][j] = avg  # 将平均值存储在U矩阵中

    # 第二步：对重要性系数矩阵进行归一化
    for i in range(n):
        for j in range(n):
            U[i][j] = np.exp(U[i][j]) / total_sum  # 对每个系数进行归一化

    return U  # 返回重要性系数矩阵

# 示例数据
att = np.random.rand(5, 5, 2)  # 生成一个随机的5x5x2的注意力系数矩阵
U = calculate_importance_coefficient(att)  # 调用函数计算重要性系数矩阵
print(U)  # 输出结果
