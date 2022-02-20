# 这是一个示例 Python 脚本。

# 按 Shift+F10 执行或将其替换为您的代码。
# 按 双击 Shift 在所有地方搜索类、文件、工具窗口、操作和设置。
from numpy import array
import numpy as np
import sklearn
import matplotlib.pyplot as plt


def print_hi(name):
    title = np.loadtxt('./diabetets/diabetes_012_health_indicators_BRFSS2015.csv', delimiter=',', dtype=str, max_rows=1)
    print(title)
    data = np.loadtxt('./diabetets/diabetes_012_health_indicators_BRFSS2015.csv', delimiter=',', skiprows=1)
    plt.scatter(data[:, 0], data[:, 4], s=4, alpha=0.3, c=data[:, 2], cmap='RdYlBu_r');



# 按间距中的绿色按钮以运行脚本。
if __name__ == '__main__':
    print_hi('PyCharm')

# 访问 https://www.jetbrains.com/help/pycharm/ 获取 PyCharm 帮助
