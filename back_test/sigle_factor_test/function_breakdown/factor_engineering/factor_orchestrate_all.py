r"""
╔═══════════════════════════════════════════════════════════════════╗
║  因子处理流程编排器 (Factor Processing Pipeline Orchestrator)    ║
║  模块: factor_engineering / 阶段: p02-因子处理                   ║
╚═══════════════════════════════════════════════════════════════════╝

功能:
  一键执行所有因子处理步骤（缓尾→标准化→中性化）
  自动执行串联的处理步骤、管理中间文件、生成清晰的文件命名

缓尾→标准化→中性化管道:
  [1] 3MAD极值缓尾 (Optional)
  [2] Zscore标准化 (Optional, 支持3种方法)
  [3] 市值行业中性化 (Optional)

CLI 用法:
  python p02_factor_processing__orchestrate_all.py \
    --input <path>                  # 原始因子文件（必填）
    --factor-col <name>             # 因子列名（必填）
    [--factor-name <name>]           # 因子简称用于驾名，默认使用factor-col
    [--output-dir <path>]            # 输出目录，默认为原文件所在目录
    [--enable-mad3 {yes|no}]         # 是否启用3MAD，默认 yes
    [--enable-zscore {yes|no}]       # 是否启用Zscore，默认 yes
    [--zscore-method {cross_sectional|time_series|global}] # Zscore方法
    [--enable-neutral {yes|no}]      # 是否启用中性化，默认 yes
    [--mv-col <name>]                # 市值列名，默认 lncap
    [--ind-col <name>]               # 行业列名，默认 ind_code
    [--keep-intermediate {yes|no}]   # 是否保留中间文件，默认 no

示例:
  # 执行所有处理步骤（推荐）
  python p02_factor_processing__orchestrate_all.py \
    --input "factor_raw.parquet" \
    --factor-col "momentum" \
    --factor-name "mom" \
    --enable-mad3 yes \
    --enable-zscore yes \
    --zscore-method cross_sectional \
    --enable-neutral yes

  # 仅执行缓尾 + 标准化（跳过中性化）
  python p02_factor_processing__orchestrate_all.py \
    --input "factor_raw.parquet" \
    --factor-col "momentum" \
    --enable-mad3 yes \
    --enable-zscore yes \
    --enable-neutral no

  # 定义推师路径、中间文件不保留
  python p02_factor_processing__orchestrate_all.py \
    --input "factor_raw.parquet" \
    --factor-col "momentum" \
    --output-dir "E:\\factor_processed" \
    --keep-intermediate no

输出文件命名规范:
  factor_{factor_name}_{processing_stage}.parquet
  例如: factor_momentum_mad3_zscore_neutral.parquet
  表示滋慈力竟灯经过3MAD缓尾 + Zscore标准化 + 中性化处理

需要scikit-learn:
  pip install scikit-learn
"""
import os
import subprocess
import pandas as pd
import argparse
import sys
from datetime import datetime
from pathlib import Path


def run_processing_step(step_name, script_path, input_file, output_file, **kwargs):
    """
    执行单个处理步骤
    
    参数:
        step_name (str): 步骤名称
        script_path (str): 处理脚本路径
        input_file (str): 输入文件
        output_file (str): 输出文件
        **kwargs: 其他CLI参数
    
    返回:
        bool: 是否成功
    """
    print(f"\n{'='*60}")
    print(f"[Pipeline] 执行步骤: {step_name}")
    print(f"{'='*60}")
    
    # 构建命令
    cmd = [
        sys.executable,
        script_path,
        '--input', input_file,
        '--output', output_file
    ]
    
    # 添加其他参数
    for key, value in kwargs.items():
        if value is not None:
            cmd.append(f'--{key}')
            if isinstance(value, bool):
                cmd.append('yes' if value else 'no')
            else:
                cmd.append(str(value))
    
    print(f"[Pipeline] 命令: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print(f"[Pipeline] ✓ {step_name} 完成")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[Pipeline] ✗ {step_name} 失败: {e}")
        return False


def build_filename_with_history(base_path, factor_name, steps_applied):
    """
    生成带有处理历史的文件名
    
    参数:
        base_path (str): 基础路径
        factor_name (str): 因子名称
        steps_applied (list): 已应用的处理步骤列表
    
    返回:
        str: 新文件名
    """
    dir_path = os.path.dirname(base_path)
    ext = os.path.splitext(base_path)[1]
    
    # 构建处理历史标签
    history_tag = '_'.join([s.lower() for s in steps_applied]) if steps_applied else 'raw'
    
    # 生成文件名
    filename = f"factor_{factor_name}_{history_tag}{ext}"
    return os.path.join(dir_path, filename)


def main():
    """CLI入口"""
    parser = argparse.ArgumentParser(
        description="因子处理流程编排器 - 自动执行所有处理步骤",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='原始因子文件路径 (parquet或csv)'
    )
    
    parser.add_argument(
        '--factor-col',
        type=str,
        required=True,
        help='因子列名'
    )
    
    parser.add_argument(
        '--factor-name',
        type=str,
        default=None,
        help='因子简称 (用于文件命名，默认使用factor-col)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='输出目录 (默认: 原文件所在目录)'
    )
    
    parser.add_argument(
        '--enable-mad3',
        type=str,
        choices=['yes', 'no'],
        default='yes',
        help='是否启用3MAD极值处理 (默认: yes)'
    )
    
    parser.add_argument(
        '--enable-zscore',
        type=str,
        choices=['yes', 'no'],
        default='yes',
        help='是否启用Zscore标准化 (默认: yes)'
    )
    
    parser.add_argument(
        '--zscore-method',
        type=str,
        choices=['cross_sectional', 'time_series', 'global'],
        default='cross_sectional',
        help='Zscore标准化方法 (默认: cross_sectional)'
    )
    
    parser.add_argument(
        '--enable-neutral',
        type=str,
        choices=['yes', 'no'],
        default='yes',
        help='是否启用市值行业中性化 (默认: yes)'
    )
    
    parser.add_argument(
        '--mv-col',
        type=str,
        default='lncap',
        help='市值列名 (默认: lncap)'
    )
    
    parser.add_argument(
        '--ind-col',
        type=str,
        default='ind_code',
        help='行业列名 (默认: ind_code)'
    )
    
    parser.add_argument(
        '--keep-intermediate',
        type=str,
        choices=['yes', 'no'],
        default='no',
        help='是否保留中间文件 (默认: no)'
    )
    
    args = parser.parse_args()
    
    try:
        # 验证输入文件
        if not os.path.exists(args.input):
            raise FileNotFoundError(f"输入文件不存在: {args.input}")
        
        # 确定输出目录
        output_dir = args.output_dir or os.path.dirname(args.input) or '.'
        os.makedirs(output_dir, exist_ok=True)
        
        # 确定因子简称
        factor_name = args.factor_name or args.factor_col
        
        # 获取脚本所在目录
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        print(f"\n{'='*60}")
        print(f"[Pipeline] 因子处理流程编排器")
        print(f"{'='*60}")
        print(f"输入文件: {args.input}")
        print(f"输出目录: {output_dir}")
        print(f"因子列: {args.factor_col}")
        print(f"因子简称: {factor_name}")
        print(f"处理步骤:")
        print(f"  - 3MAD极值处理: {args.enable_mad3}")
        print(f"  - Zscore标准化: {args.enable_zscore} ({args.zscore_method})")
        print(f"  - 市值行业中性化: {args.enable_neutral}")
        
        # 追踪当前文件和处理步骤
        current_file = args.input
        steps_applied = []
        
        # 步骤1: 3MAD极值处理
        if args.enable_mad3 == 'yes':
            mad3_file = build_filename_with_history(
                os.path.join(output_dir, 'temp.parquet'),
                factor_name, steps_applied + ['mad3']
            )
            script = os.path.join(script_dir, 'factor_mad3_winsorization.py')
            
            if run_processing_step(
                '3MAD极值处理',
                script,
                current_file,
                mad3_file,
                **{'factor-col': args.factor_col, 'enable': 'yes'}
            ):
                current_file = mad3_file
                steps_applied.append('mad3')
            else:
                print("[Pipeline] ✗ 处理终止")
                return 1
        
        # 步骤2: Zscore标准化
        if args.enable_zscore == 'yes':
            zscore_file = build_filename_with_history(
                os.path.join(output_dir, 'temp.parquet'),
                factor_name, steps_applied + ['zscore']
            )
            script = os.path.join(script_dir, 'factor_zscore_normalization.py')
            
            if run_processing_step(
                'Zscore标准化',
                script,
                current_file,
                zscore_file,
                **{
                    'factor-col': args.factor_col,
                    'method': args.zscore_method,
                    'enable': 'yes'
                }
            ):
                current_file = zscore_file
                steps_applied.append('zscore')
            else:
                print("[Pipeline] ✗ 处理终止")
                return 1
        
        # 步骤3: 市值行业中性化
        if args.enable_neutral == 'yes':
            neutral_file = build_filename_with_history(
                os.path.join(output_dir, 'temp.parquet'),
                factor_name, steps_applied + ['neutral']
            )
            script = os.path.join(script_dir, 'factor_market_cap_industry_neutralization.py')
            
            if run_processing_step(
                '市值行业中性化',
                script,
                current_file,
                neutral_file,
                **{
                    'factor-col': args.factor_col,
                    'mv-col': args.mv_col,
                    'ind-col': args.ind_col,
                    'enable': 'yes'
                }
            ):
                current_file = neutral_file
                steps_applied.append('neutral')
            else:
                print("[Pipeline] ✗ 处理终止")
                return 1
        
        # 最终文件
        final_file = build_filename_with_history(
            os.path.join(output_dir, 'temp.parquet'),
            factor_name, steps_applied
        )
        
        # 如果最后一个处理的输出就是最终文件，无需复制
        if current_file != final_file:
            import shutil
            shutil.copy(current_file, final_file)
        
        # 清理中间文件
        if args.keep_intermediate == 'no' and len(steps_applied) > 0:
            print(f"\n[Pipeline] 清理中间文件...")
            # 这里可以添加清理逻辑
        
        # 最终统计
        print(f"\n{'='*60}")
        print(f"[Pipeline] ✓ 处理完成！")
        print(f"{'='*60}")
        print(f"最终输出文件: {final_file}")
        print(f"应用处理步骤: {', '.join(steps_applied)}")
        
        # 预览
        if final_file.endswith('.parquet'):
            df_final = pd.read_parquet(final_file)
        else:
            df_final = pd.read_csv(final_file)
        
        print(f"\n数据预览 (前5行):")
        print(df_final.head())
        print(f"\n数据统计:")
        print(f"  总行数: {len(df_final)}")
        print(f"  列数: {len(df_final.columns)}")
        print(f"  文件大小: {os.path.getsize(final_file) / 1024 / 1024:.2f} MB")
        
        return 0
        
    except Exception as e:
        print(f"\n[ERROR] {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
