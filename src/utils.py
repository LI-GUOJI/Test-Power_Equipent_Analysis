"""
设备振动数据分析工具模块 (src/utils.py)
用于电力设备试验数据的常见分析任务
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')  # 忽略警告信息，保持输出整洁
#[注意]如果VS Code的虚拟环境与Jupyter不同时，外部库会以波浪线显示。

# 设置中文显示和绘图风格
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")
sns.set_palette("husl")


def describe_data(df, datetime_col='timestamp', device_col='device_id', 
                  vibration_cols=None, print_output=True):
    """
    生成设备振动数据的详细描述性统计报告
    
    参数:
    ----------
    df : pandas.DataFrame
        输入的设备数据DataFrame
    datetime_col : str, 默认 'timestamp'
        时间戳列名
    device_col : str, 默认 'device_id'
        设备ID列名
    vibration_cols : list, 可选
        振动数据列名列表，如果为None则自动检测包含'vibration'的列
    print_output : bool, 默认 True
        是否打印输出报告
        
    返回:
    ----------
    dict: 包含各项统计指标的字典
    """
    
    print("=" * 70)
    print("设备振动数据详细分析报告")
    print("=" * 70)
    
    # 1. 基础信息
    if not pd.api.types.is_datetime64_any_dtype(df[datetime_col]):
        df[datetime_col] = pd.to_datetime(df[datetime_col])    
    print(f"\n1. 数据基础信息:")
    print(f"   数据形状: {df.shape[0]} 行 × {df.shape[1]} 列")
    print(f"   时间范围: {df[datetime_col].min()} 至 {df[datetime_col].max()}")
    print(f"   时间跨度: {(df[datetime_col].max() - df[datetime_col].min()).days} 天")
    
    # 2. 设备信息
    if device_col in df.columns:
        device_counts = df[device_col].value_counts()
        print(f"\n2. 设备分布:")
        print(f"   设备数量: {len(device_counts)} 台")
        for device, count in device_counts.items():
            percentage = (count / len(df)) * 100
            print(f"   - {device}: {count} 条记录 ({percentage:.1f}%)")
    
    # 3. 确定振动数据列
    if vibration_cols is None:
        vibration_cols = [col for col in df.columns if 'vibration' in col.lower()]
    
    if not vibration_cols:
        print("\n警告: 未检测到振动数据列!")
        return {}
    
    print(f"\n3. 振动数据分析列: {', '.join(vibration_cols)}")
    
    # 4. 缺失值统计
    missing_info = {}
    print(f"\n4. 数据完整性检查:")
    for col in vibration_cols + ['temperature', 'load']:
        if col in df.columns:
            missing_count = df[col].isnull().sum()
            missing_percent = (missing_count / len(df)) * 100
            missing_info[col] = missing_percent
            if missing_count > 0:
                print(f"   - {col}: {missing_count} 个缺失值 ({missing_percent:.2f}%)")
            else:
                print(f"   - {col}: 无缺失值")
    
    # 5. 详细统计描述
    print(f"\n5. 振动数据统计描述:")
    stats_dict = {}
    
    for col in vibration_cols:
        if col in df.columns:
            # 移除缺失值进行计算
            col_data = df[col].dropna()
            
            if len(col_data) > 0:
                stats_dict[col] = {
                    'mean': col_data.mean(),
                    'std': col_data.std(),
                    'min': col_data.min(),
                    '25%': col_data.quantile(0.25),
                    'median': col_data.median(),
                    '75%': col_data.quantile(0.75),
                    'max': col_data.max(),
                    'range': col_data.max() - col_data.min(),
                    'skewness': col_data.skew(),
                    'kurtosis': col_data.kurtosis()
                }
                
                print(f"\n   {col} (单位: mm/s):")
                print(f"      均值: {stats_dict[col]['mean']:.3f} | 标准差: {stats_dict[col]['std']:.3f}")
                print(f"      范围: [{stats_dict[col]['min']:.3f}, {stats_dict[col]['max']:.3f}]")
                print(f"      四分位: [{stats_dict[col]['25%']:.3f}, {stats_dict[col]['median']:.3f}, {stats_dict[col]['75%']:.3f}]")
                print(f"      偏度: {stats_dict[col]['skewness']:.3f} | 峰度: {stats_dict[col]['kurtosis']:.3f}")
                
                # 异常值检测（基于IQR）
                Q1 = stats_dict[col]['25%']
                Q3 = stats_dict[col]['75%']
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = col_data[(col_data < lower_bound) | (col_data > upper_bound)]
                outlier_percent = (len(outliers) / len(col_data)) * 100
                print(f"      异常值: {len(outliers)} 个 ({outlier_percent:.2f}%)")
    
    # 6. 相关性分析（如果有多列振动数据）
    if len(vibration_cols) > 1:
        print(f"\n6. 振动数据相关性矩阵:")
        vib_df = df[vibration_cols].dropna()
        if len(vib_df) > 1:
            correlation_matrix = vib_df.corr()
            print(f"   相关系数矩阵:")
            for i, col1 in enumerate(vibration_cols):
                for j, col2 in enumerate(vibration_cols):
                    if i < j:  # 只显示上三角部分，避免重复
                        corr_value = correlation_matrix.loc[col1, col2]
                        print(f"      {col1} vs {col2}: {corr_value:.3f}")
    
    print("\n" + "=" * 70)
    
    return stats_dict


def plot_vibration_trend(df, device_id=None, vibration_cols=None, 
                         time_window='6H', save_path=None, figsize=(14, 10)):
    """
    绘制设备振动趋势图，支持多轴对比和异常点标注
    
    参数:
    ----------
    df : pandas.DataFrame
        输入的设备数据DataFrame
    device_id : str, 可选
        指定设备ID，如果为None则绘制所有设备
    vibration_cols : list, 可选
        振动数据列名列表，如果为None则自动检测包含'vibration'的列
    time_window : str, 默认 '6H'
        时间重采样窗口，用于平滑曲线。可选: 'H'（小时）, '6H', '12H', 'D'（天）
    save_path : str, 可选
        图表保存路径，如果为None则不保存
    figsize : tuple, 默认 (14, 10)
        图表大小
        
    返回:
    ----------
    matplotlib.figure.Figure: 生成的图表对象
    """
    
    # 1. 数据准备
    plot_df = df.copy()
    
    # 筛选特定设备
    if device_id is not None and 'device_id' in plot_df.columns:
        plot_df = plot_df[plot_df['device_id'] == device_id]
        title_device = f" ({device_id})"
    else:
        title_device = ""
    
    # 确定振动数据列
    if vibration_cols is None:
        vibration_cols = [col for col in plot_df.columns if 'vibration' in col.lower()]
    
    if not vibration_cols:
        print("错误: 未找到振动数据列!")
        return None
    
    # 2. 创建图表
    fig, axes = plt.subplots(len(vibration_cols) + 1, 1, figsize=figsize, 
                             sharex=True, height_ratios=[3, 3, 3, 1])
    
    # 如果只有一列振动数据，调整axes结构
    if len(vibration_cols) == 1:
        axes = [axes] if len(vibration_cols) == 1 else axes
    
    # 设置时间索引
    if 'timestamp' not in plot_df.columns:
        print("错误: 数据中缺少'timestamp'列!")
        return None
    
    plot_df = plot_df.set_index('timestamp').sort_index()
    
    # 3. 绘制振动趋势
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']  # 红、青绿、蓝
    
    for idx, col in enumerate(vibration_cols):
        if idx >= len(axes) - 1:
            break
            
        ax = axes[idx]
        
        # 绘制原始数据点（浅色、半透明）
        ax.scatter(plot_df.index, plot_df[col], alpha=0.3, s=10, 
                  color=colors[idx % len(colors)], label='原始数据点')
        
        # 重采样并绘制平滑趋势线
        if time_window:
            try:
                resampled = plot_df[col].resample(time_window).mean()
                ax.plot(resampled.index, resampled.values, linewidth=2.5, 
                       color=colors[idx % len(colors)], label=f'{time_window}移动平均')
            except:
                pass
        
        # 检测并标注异常点（基于3σ原则）
        mean_val = plot_df[col].mean()
        std_val = plot_df[col].std()
        threshold_upper = mean_val + 3 * std_val
        threshold_lower = mean_val - 3 * std_val
        
        outliers = plot_df[(plot_df[col] > threshold_upper) | (plot_df[col] < threshold_lower)]
        
        if len(outliers) > 0:
            ax.scatter(outliers.index, outliers[col], color='red', s=80, 
                      marker='x', zorder=5, label='异常点 (3σ)')
        
        # 添加阈值线
        ax.axhline(y=threshold_upper, color='red', linestyle='--', alpha=0.7, linewidth=1)
        ax.axhline(y=threshold_lower, color='red', linestyle='--', alpha=0.7, linewidth=1)
        ax.axhline(y=mean_val, color='green', linestyle='-', alpha=0.5, linewidth=1, label='均值')
        
        # 设置子图属性
        ax.set_ylabel(f'{col} (mm/s)', fontsize=12)
        ax.set_title(f'{col} 振动趋势{title_device}', fontsize=13, fontweight='bold')
        ax.legend(loc='upper left', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # 添加统计信息文本框
        stats_text = (f"均值: {mean_val:.2f}\n"
                     f"标准差: {std_val:.2f}\n"
                     f"异常点: {len(outliers)}个")
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
               verticalalignment='top', horizontalalignment='left',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
               fontsize=9)
    
    # 4. 绘制工况参数（如果存在）
    if 'temperature' in plot_df.columns and 'load' in plot_df.columns:
        ax_env = axes[-1]
        
        # 温度曲线
        color_temp = 'tab:red'
        ax_env.plot(plot_df.index, plot_df['temperature'], color=color_temp, 
                   linewidth=1.5, label='温度 (°C)')
        ax_env.set_ylabel('温度 (°C)', color=color_temp, fontsize=11)
        ax_env.tick_params(axis='y', labelcolor=color_temp)
        
        # 负载曲线（次坐标轴）
        ax_load = ax_env.twinx()
        color_load = 'tab:blue'
        ax_load.plot(plot_df.index, plot_df['load'], color=color_load, 
                    linewidth=1.5, linestyle='--', alpha=0.7, label='负载 (%)')
        ax_load.set_ylabel('负载 (%)', color=color_load, fontsize=11)
        ax_load.tick_params(axis='y', labelcolor=color_load)
        
        ax_env.set_title(f'工况参数{title_device}', fontsize=13, fontweight='bold')
        ax_env.grid(True, alpha=0.3)
        
        # 合并图例
        lines_env, labels_env = ax_env.get_legend_handles_labels()
        lines_load, labels_load = ax_load.get_legend_handles_labels()
        ax_env.legend(lines_env + lines_load, labels_env + labels_load, 
                     loc='upper left', fontsize=9)
    
    # 5. 设置公共属性
    axes[-1].set_xlabel('时间', fontsize=12)
    plt.suptitle(f'设备振动监测综合趋势图{title_device}', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # 6. 保存图表
    if save_path:
        try:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"图表已保存至: {save_path}")
        except Exception as e:
            print(f"保存图表时出错: {e}")
    
    return fig


def filter_3sigma(df, columns=None, device_col='device_id', 
                  group_by_device=False, return_stats=False):
    """
    基于3σ原则过滤异常值，支持按设备分组处理
    
    参数:
    ----------
    df : pandas.DataFrame
        输入的设备数据DataFrame
    columns : list, 可选
        需要过滤的列名列表，如果为None则自动检测数值列
    device_col : str, 默认 'device_id'
        设备ID列名（用于分组过滤）
    group_by_device : bool, 默认 False
        是否按设备单独计算阈值（推荐True，因为不同设备基准可能不同）
    return_stats : bool, 默认 False
        是否返回过滤统计信息
        
    返回:
    ----------
    pandas.DataFrame: 过滤后的DataFrame
    如果return_stats=True，同时返回 (filtered_df, stats_dict)
    """
    
    # 创建副本避免修改原数据
    df_filtered = df.copy()
    
    # 确定需要过滤的列
    if columns is None:
        # 自动选择数值列，排除时间戳和设备ID
        exclude_cols = ['timestamp', device_col, 'date', 'time', 'id']
        columns = [col for col in df.select_dtypes(include=[np.number]).columns 
                  if col not in exclude_cols]
    
    print(f"正在基于3σ原则过滤异常值...")
    print(f"处理列: {', '.join(columns)}")
    print(f"分组处理: {'是' if group_by_device else '否'}")
    
    # 记录统计信息
    stats_dict = {
        'original_rows': len(df),
        'filtered_rows': 0,
        'removed_rows': 0,
        'removed_percentage': 0.0,
        'column_stats': {}
    }
    
    if group_by_device and device_col in df.columns:
        # 按设备分组处理
        devices = df[device_col].unique()
        print(f"检测到 {len(devices)} 台设备，将分别计算阈值")
        
        for device in devices:
            device_mask = df_filtered[device_col] == device
            device_count = device_mask.sum()
            
            for col in columns:
                if col not in df_filtered.columns:
                    continue
                    
                # 计算该设备该列的阈值
                device_data = df_filtered.loc[device_mask, col].dropna()
                
                if len(device_data) > 10:  # 确保有足够数据计算
                    mean_val = device_data.mean()
                    std_val = device_data.std()
                    
                    if std_val > 0:  # 避免除零
                        upper_limit = mean_val + 3 * std_val
                        lower_limit = mean_val - 3 * std_val
                        
                        # 记录异常点
                        outlier_mask = device_mask & (
                            (df_filtered[col] > upper_limit) | 
                            (df_filtered[col] < lower_limit)
                        )
                        
                        # 标记异常点（设为NaN）
                        df_filtered.loc[outlier_mask, col] = np.nan
                        
                        # 记录统计
                        if col not in stats_dict['column_stats']:
                            stats_dict['column_stats'][col] = {'outliers': 0, 'by_device': {}}
                        
                        stats_dict['column_stats'][col]['outliers'] += outlier_mask.sum()
                        stats_dict['column_stats'][col]['by_device'][device] = {
                            'mean': mean_val,
                            'std': std_val,
                            'upper_limit': upper_limit,
                            'lower_limit': lower_limit,
                            'outliers': outlier_mask.sum(),
                            'outlier_percent': (outlier_mask.sum() / device_count * 100)
                        }
    else:
        # 全局处理（不分组）
        for col in columns:
            if col not in df_filtered.columns:
                continue
                
            # 计算全局阈值
            col_data = df_filtered[col].dropna()
            
            if len(col_data) > 10:
                mean_val = col_data.mean()
                std_val = col_data.std()
                
                if std_val > 0:
                    upper_limit = mean_val + 3 * std_val
                    lower_limit = mean_val - 3 * std_val
                    
                    # 标记异常点
                    outlier_mask = (df_filtered[col] > upper_limit) | (df_filtered[col] < lower_limit)
                    df_filtered.loc[outlier_mask, col] = np.nan
                    
                    # 记录统计
                    if col not in stats_dict['column_stats']:
                        stats_dict['column_stats'][col] = {'outliers': 0}
                    
                    stats_dict['column_stats'][col]['outliers'] = outlier_mask.sum()
                    stats_dict['column_stats'][col].update({
                        'global_mean': mean_val,
                        'global_std': std_val,
                        'global_upper_limit': upper_limit,
                        'global_lower_limit': lower_limit,
                        'outlier_percent': (outlier_mask.sum() / len(df) * 100)
                    })
    
    # 计算总体统计
    stats_dict['filtered_rows'] = len(df_filtered.dropna(subset=columns))
    stats_dict['removed_rows'] = stats_dict['original_rows'] - stats_dict['filtered_rows']
    stats_dict['removed_percentage'] = (stats_dict['removed_rows'] / stats_dict['original_rows'] * 100)
    
    # 打印摘要
    print(f"\n过滤完成!")
    print(f"原始数据行数: {stats_dict['original_rows']}")
    print(f"过滤后行数: {stats_dict['filtered_rows']}")
    print(f"移除行数: {stats_dict['removed_rows']} ({stats_dict['removed_percentage']:.2f}%)")
    
    if stats_dict['column_stats']:
        print(f"\n各列异常点统计:")
        for col, col_stats in stats_dict['column_stats'].items():
            if 'outliers' in col_stats:
                print(f"  {col}: {col_stats['outliers']} 个异常点")
    
    print("=" * 50)
    
    if return_stats:
        return df_filtered, stats_dict
    else:
        return df_filtered


# 额外的实用函数
def calculate_vibration_metrics(df, device_col='device_id', vibration_cols=None):
    """
    计算设备振动指标的聚合统计
    """
    if vibration_cols is None:
        vibration_cols = [col for col in df.columns if 'vibration' in col.lower()]
    
    metrics = {}
    for col in vibration_cols:
        if col in df.columns:
            metrics[f'{col}_mean'] = df[col].mean()
            metrics[f'{col}_std'] = df[col].std()
            metrics[f'{col}_max'] = df[col].max()
            metrics[f'{col}_min'] = df[col].min()
            metrics[f'{col}_range'] = df[col].max() - df[col].min()
    
    return metrics


def detect_anomalous_devices(df, vibration_cols=None, threshold_std=2.0):
    """
    检测表现异常的设备（振动水平显著高于平均）
    """
    if vibration_cols is None:
        vibration_cols = [col for col in df.columns if 'vibration' in col.lower()]
    
    anomalous_devices = {}
    
    for col in vibration_cols:
        if col in df.columns and 'device_id' in df.columns:
            device_stats = df.groupby('device_id')[col].agg(['mean', 'std']).reset_index()
            overall_mean = device_stats['mean'].mean()
            overall_std = device_stats['mean'].std()
            
            # 识别异常设备（均值超过总体均值+threshold_std*标准差）
            threshold = overall_mean + threshold_std * overall_std
            anomalous = device_stats[device_stats['mean'] > threshold]
            
            if not anomalous.empty:
                anomalous_devices[col] = anomalous['device_id'].tolist()
    
    return anomalous_devices

def check_temperature_alert(temperature, warning_threshold=75, critical_threshold=85):
    """
    检查设备温度并返回预警等级。
    
    参数:
        temperature (float): 当前温度值
        warning_threshold (float): 警告阈值
        critical_threshold (float): 严重警告阈值
        
    返回:
        str: 'normal', 'warning', 'critical'
    """
    if temperature >= critical_threshold:
        return "critical"
    elif temperature >= warning_threshold:
        return "warning"
    else:
        return "normal"