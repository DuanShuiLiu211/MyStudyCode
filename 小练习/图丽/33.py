import threading
import time
from datetime import datetime

import matplotlib.pyplot as plot
import numpy as np
from scipy import stats
from sklearn.cluster import KMeans


class TimeTracker:
    def __init__(self):
        self.timestamps = []
        self.start_time = None

    def record(self):
        current = time.time()
        if self.start_time is None:
            self.start_time = current
        self.timestamps.append(current - self.start_time)


def run_kmeans():
    """在子线程中运行k-means聚类"""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 开始k-means聚类...")
    # 增加数据量和维度，使计算更密集
    data = np.random.rand(200000, 200)
    kmeans = KMeans(n_clusters=20, max_iter=500)
    kmeans.fit(data)
    print(f"[{datetime.now().strftime('%H:%M:%S')}] k-means聚类完成")


def main_thread_task(tracker):
    """主线程任务：每秒记录一次时间戳"""
    count = 0
    max_count = 50  # 增加采样点数量

    while count < max_count:
        tracker.record()
        time.sleep(1)
        count += 1


def run_test(with_kmeans=True):
    """运行测试并返回时间戳列表"""
    tracker = TimeTracker()

    if with_kmeans:
        kmeans_thread = threading.Thread(target=run_kmeans)
        kmeans_thread.start()

    main_thread_task(tracker)

    if with_kmeans:
        kmeans_thread.join()

    return tracker.timestamps


def analyze_intervals(intervals, name=""):
    """详细分析时间间隔"""
    mean = np.mean(intervals)
    std = np.std(intervals)

    # 计算95%置信区间
    ci = stats.t.interval(
        0.95, len(intervals) - 1, loc=mean, scale=stats.sem(intervals)
    )

    print(f"\n=== {name} 统计分析 ===")
    print(f"样本数量: {len(intervals)}")
    print(f"平均间隔: {mean:.6f}秒")
    print(f"标准差: {std:.6f}秒")
    print(f"95%置信区间: [{ci[0]:.6f}, {ci[1]:.6f}]")

    # 进行单样本t检验，检验均值是否显著不等于1
    t_stat, p_value = stats.ttest_1samp(intervals, 1.0)
    print(f"单样本t检验 p值: {p_value:.6f}")
    print(f"t统计量: {t_stat:.6f}")

    return mean, std, ci, p_value


def plot_results(with_kmeans_times, without_kmeans_times):
    """绘制详细的对比图"""
    # 计算时间间隔
    with_intervals = np.diff(with_kmeans_times)
    without_intervals = np.diff(without_kmeans_times)

    # 进行统计分析
    print("\n=== 统计检验 ===")
    with_stats = analyze_intervals(with_intervals, "有K-means运行")
    without_stats = analyze_intervals(without_intervals, "无K-means运行")

    # 两样本t检验，比较两组数据是否有显著差异
    t_stat, p_value = stats.ttest_ind(with_intervals, without_intervals)
    print(f"\n两样本t检验 p值: {p_value:.6f}")
    print(f"t统计量: {t_stat:.6f}")

    # 创建一个2x2的子图布局
    plot.figure(figsize=(15, 12))

    # 1. 时间序列图
    plot.subplot(2, 2, 1)
    plot.plot(with_intervals, label="With K-means", marker="o", alpha=0.6)
    plot.plot(without_intervals, label="Without K-means", marker="o", alpha=0.6)
    plot.axhline(y=1.0, color="r", linestyle="--", alpha=0.3, label="Ideal (1s)")
    plot.xlabel("Measurement Point")
    plot.ylabel("Time Interval (s)")
    plot.title("Time Interval Comparison")
    plot.legend()
    plot.grid(True)

    # 2. 直方图
    plot.subplot(2, 2, 2)
    plot.hist(with_intervals, bins=20, alpha=0.5, label="With K-means", density=True)
    plot.hist(
        without_intervals, bins=20, alpha=0.5, label="Without K-means", density=True
    )
    plot.axvline(x=1.0, color="r", linestyle="--", alpha=0.3, label="Ideal (1s)")
    plot.xlabel("Time Interval (s)")
    plot.ylabel("Density")
    plot.title("Distribution of Time Intervals")
    plot.legend()
    plot.grid(True)

    # 3. 均值和置信区间对比图
    plot.subplot(2, 2, 3)
    labels = ["Without K-means", "With K-means"]
    means = [without_stats[0], with_stats[0]]
    cis = [without_stats[2], with_stats[2]]

    for i in range(2):
        plot.vlines(x=i, ymin=cis[i][0], ymax=cis[i][1], color="blue", alpha=0.5)
        plot.plot(i, means[i], "ro")

    plot.axhline(y=1.0, color="r", linestyle="--", alpha=0.3, label="Ideal (1s)")
    plot.xticks(range(2), labels)
    plot.ylabel("Time Interval (s)")
    plot.title("Mean and 95% Confidence Intervals")
    plot.grid(True)

    # 4. 标准差对比图
    plot.subplot(2, 2, 4)
    stds = [without_stats[1], with_stats[1]]
    plot.bar(range(2), stds, alpha=0.6)
    plot.xticks(range(2), labels)
    plot.ylabel("Standard Deviation (s)")
    plot.title("Standard Deviation Comparison")
    plot.grid(True)

    plot.tight_layout()
    plot.show()


if __name__ == "__main__":
    print("开始测试...")

    print("\n=== 测试1: 有k-means运行 ===")
    with_kmeans_times = run_test(with_kmeans=True)

    print("\n等待5秒后进行下一个测试...")
    time.sleep(5)

    print("\n=== 测试2: 无k-means运行 ===")
    without_kmeans_times = run_test(with_kmeans=False)

    # 绘制结果
    plot_results(with_kmeans_times, without_kmeans_times)
