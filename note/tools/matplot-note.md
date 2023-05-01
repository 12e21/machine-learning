
# 以下是一些常用的函数及其作用：
- `import matplotlib.pyplot as plt` : 导入matplotlib.pyplot模块

- `plt.figure(num=None, figsize=None, dpi=None, facecolor=None, edgecolor=None, frameon=True, clear=False, tight_layout=False, constrained_layout=False)`:  创建一个新的图形窗口，可以指定窗口编号、大小、分辨率、颜色、边框、清除旧窗口等参数
- `plt.plot(x, y, color=None, linestyle=None, marker=None, label=None)` : 在当前的图形窗口中绘制一条或多条线，可以指定x和y坐标、颜色、线型、标记、图例等参数
- `plt.scatter(x, y, s=None, c=None, marker=None, cmap=None, norm=None, vmin=None, vmax=None, alpha=None, linewidths=None, edgecolors=None)` : 在当前的图形窗口中绘制一组散点，可以指定x和y坐标、大小、颜色、标记、颜色映射、透明度、边框等参数
- `plt.bar(x, height, width=0.8, bottom=None, align='center', color=None, edgecolor=None)` : 在当前的图形窗口中绘制一组条形图，可以指定x坐标、高度、宽度、底部位置、对齐方式、颜色、边框等参数
- `plt.hist(x, bins=None, range=None, density=False, weights=None, cumulative=False, bottom=None, histtype='bar', align='mid', orientation='vertical', rwidth=None, log=False)` : 在当前的图形窗口中绘制一个直方图，可以指定数据集、分组数目或区间、范围、密度或频数、权重、累积频数、底部位置、直方图类型、对齐方式、方向、相对宽度、对数坐标等参数
- `plt.pie(x, explode=None, labels=None, colors=None, autopct=None)` : 在当前的图形窗口中绘制一个饼图，可以指定数据集、分离距离、标签、颜色、百分比格式等参数
- `plt.title(label)` : 给当前的图形窗口添加一个标题，可以指定标题文本
- `plt.xlabel(xlabel)` : 给当前的图形窗口的x轴添加一个标签，可以指定标签文本
- `plt.ylabel(ylabel)` : 给当前的图形窗口的y轴添加一个标签，可以指定标签文本
- `plt.legend(loc='best', ncol=1)` : 给当前的图形窗口添加一个图例，可以指定位置和列数等参数
- `plt.grid(b=True)` : 给当前的图形窗口添加网格线，可以指定是否显示网格线等参数
- `plt.xlim(left=None,right=None)` : 设置当前的图形窗口的x轴范围，可以指定左右端点值
- `plt.ylim(bottom=None,top=None)` : 设置当前的图形窗口的y轴范围，可以指定上下端点值
- `plt.xticks(ticks=None)` : 设置当前的图形窗口的x轴刻度，可以指定刻度位置等参数
- `plt.yticks(ticks=None)` : 设置当前的图形窗口的y轴刻度，可以指定刻度位置等参数
- `plt.xscale(value='linear')` : 设置当前的图形窗口的x轴比例，可以是线性（linear）、对数（log）或逻辑（logit）等
- `plt.yscale(value='linear')` : 设置当前的图形窗口的y轴比例，可以是线性（linear）、对数（log）或逻辑（logit等）
- `plt.show()` : 显示当前的图形窗口，可以指定是否阻塞程序运行等参数
- `plt.savefig(fname)` : 保存当前的图形窗口为一个文件，可以指定文件名、格式、分辨率、透明度等参数
- `plt.clf()` : 清除当前的图形窗口中的所有内容，以便重新绘制新的内容
- `plt.pause(interval)` : 暂停程序运行一段时间，以便观察绘制出来的图形，可以指定暂停的秒数