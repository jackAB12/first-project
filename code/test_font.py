# -*- coding: utf-8 -*-
"""
测试matplotlib中文字体配置
"""
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 测试1: 列出系统可用的中文字体
print("=" * 60)
print("系统中可用的中文字体:")
print("=" * 60)
fonts = fm.findSystemFonts(fontpaths=None, fontext='ttf')
chinese_fonts = []
for font_path in fonts:
    if 'sim' in font_path.lower() or 'hei' in font_path.lower() or 'kai' in font_path.lower():
        try:
            font_prop = fm.FontProperties(fname=font_path)
            font_name = font_prop.get_name()
            print(f"{font_name}: {font_path}")
            chinese_fonts.append((font_name, font_path))
        except:
            pass

# 测试2: 使用完整路径直接加载字体
print("\n" + "=" * 60)
print("测试字体加载:")
print("=" * 60)

font_paths_to_test = [
    'C:/Windows/Fonts/simhei.ttf',
    'C:/Windows/Fonts/simsun.ttc',
    'C:/Windows/Fonts/simkai.ttf'
]

for font_path in font_paths_to_test:
    try:
        font_prop = fm.FontProperties(fname=font_path)
        print(f"[OK] Load success: {font_path}")
        print(f"  Font name: {font_prop.get_name()}")
    except Exception as e:
        print(f"[FAIL] Load failed: {font_path}")
        print(f"  Error: {e}")

# 测试3: 绘制测试图
print("\n" + "=" * 60)
print("生成测试图表:")
print("=" * 60)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# 方法1: 直接使用字体文件路径
font_prop1 = fm.FontProperties(fname='C:/Windows/Fonts/simhei.ttf')
axes[0].text(0.5, 0.5, '测试中文显示\n黑体SimHei',
            fontproperties=font_prop1, fontsize=16, ha='center', va='center')
axes[0].set_title('方法1: FontProperties(fname=...)', fontproperties=font_prop1)
axes[0].axis('off')

# 方法2: 注册字体后使用
fm.fontManager.addfont('C:/Windows/Fonts/simhei.ttf')
plt.rcParams['font.family'] = 'SimHei'
axes[1].text(0.5, 0.5, '测试中文显示\nrcParams设置',
            fontsize=16, ha='center', va='center')
axes[1].set_title('方法2: rcParams设置')
axes[1].axis('off')

# 方法3: 使用备选字体列表
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun']
plt.rcParams['axes.unicode_minus'] = False
axes[2].text(0.5, 0.5, '测试中文显示\n备选字体列表',
            fontsize=16, ha='center', va='center')
axes[2].set_title('方法3: 备选字体列表')
axes[2].axis('off')

plt.tight_layout()
plt.savefig('figures/test_chinese_font.png', dpi=150, bbox_inches='tight')
print("[OK] Test figure saved: figures/test_chinese_font.png")
print("\nPlease check if Chinese characters display correctly in the image")
