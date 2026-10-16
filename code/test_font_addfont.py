# -*- coding: utf-8 -*-
"""
修复后的字体配置 - 使用addfont方法
"""

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 清除字体缓存并注册字体
fm.fontManager.addfont('C:/Windows/Fonts/simhei.ttf')

# 设置字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 测试绘图
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot([1, 2, 3, 4], [1, 4, 2, 3], 'o-', linewidth=2, label='测试数据')
ax.set_xlabel('时间步长', fontsize=12)
ax.set_ylabel('特征值', fontsize=12)
ax.set_title('中文字体显示测试', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

plt.savefig('figures/test_font_fixed.png', dpi=150, bbox_inches='tight')
print("Test figure saved: figures/test_font_fixed.png")
