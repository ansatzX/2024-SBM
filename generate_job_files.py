"""
作业提交文件生成器（面向过程风格）

生成作业提交流程所需的所有文件：
- run.sh: 包含所有轨迹运行命令
- p_run.make: makefile 定义
- slurm 文件: 90个批处理作业脚本
"""

import os
from typing import List, Tuple


# ============================================================================
# 生成 run.sh
# ============================================================================

def generate_run_script(params: List[dict], output_path: str = "run.sh") -> None:
    """
    生成 run.sh 文件，包含所有轨迹运行命令

    Args:
        params: 参数列表，每个元素是包含 s, alpha, nmodes, nsteps, rho_type 等的字典
        output_path: 输出文件路径
    """
    with open(output_path, 'w') as f:
        for i, p in enumerate(params):
            s = p.get('s', 1.0)
            alpha = p.get('alpha', 0.5)
            nmodes = p.get('nmodes', 1000)
            nsteps = p.get('nsteps', 300)
            rho_type = p.get('rho_type', 0)
            Omega = p.get('Omega', 1)
            omega_c = p.get('omega_c', 10)
            bond_dims = p.get('bond_dims', 20)
            td_method = p.get('td_method', 0)

            run_cmd = f'python3 traj_run.py --s {s:.3f} --alpha {alpha:.3f} --omega {Omega} --rho_type {rho_type} --nmodes {nmodes} --nstep {nsteps}'

            if p.get('calc_mutual_info', False):
                run_cmd += ' --calc_mutual_info 1'
            if p.get('calc_1site_entropy', False):
                run_cmd += ' --calc_1sites_entropy 1'
            if p.get('calc_spin_entropy', False):
                run_cmd += ' --calc_spin_entropy 1'

            run_cmd += '\n'
            f.write(run_cmd)


# ============================================================================
# 生成 p_run.make
# ============================================================================

def generate_makefile(n_jobs: int, ntasks_per_batch: int = 8, output_path: str = "p_run.make") -> None:
    """
    生成 Makefile

    Args:
        n_jobs: 总作业数
        ntasks_per_batch: 每个 batch 包含的作业数
        output_path: 输出文件路径
    """
    with open(output_path, 'w') as f:
        # 定义 'all' 目标
        all_targets = '\t'.join([f'run{i}' for i in range(n_jobs)])
        f.write(f'all:\t{all_targets}\n\n')

        # 定义单个 run 目标
        jobs = []
        for i in range(n_jobs):
            s = i % 90  # 周期，假设有90个不同参数组合
            jobs.append(f'run{i}:\n\tpython3 traj_run.py --s {s*0.01:.2f} --alpha 0.05 --rho_type 0 --nmodes 1000 --nstep 300 --calc_1sites_entropy 1\n')

        # 如果有自定义参数列表，应该从参数列表生成
        # 这里简化，使用默认模式

        f.write('\n'.join(jobs))

        # 定义 batch 目标
        n_batches = (n_jobs + ntasks_per_batch - 1) // ntasks_per_batch
        for batch_idx in range(n_batches):
            start = batch_idx * ntasks_per_batch
            end = min((batch_idx + 1) * ntasks_per_batch, n_jobs)

            targets = '\t'.join([f'run{i}' for i in range(start, end)])
            f.write(f'\nbatch_{batch_idx}:\n\t{targets}\n')
            f.write(f'\techo batch_{batch_idx} calc\n')

        # 处理剩余作业
        if n_jobs % ntasks_per_batch != 0:
            start = (n_jobs // ntasks_per_batch) * ntasks_per_batch
            targets = '\t'.join([f'run{i}' for i in range(start, n_jobs)])
            f.write(f'\nbatch_remain:\n\t{targets}\n')
            f.write(f'\techo batch_remain calc\n')


# ============================================================================
# 生成 slurm 文件
# ============================================================================

def generate_slurm_templates(cpu_per_task: int = 8,
                            conda_env: str = "reno-py310",
                            output_prefix: str = "reno_run_batch_") -> List[str]:
    """
    生成 slurm 模板字符串

    Args:
        cpu_per_task: 每个 CPU 任务的核心数
        conda_env: conda 环境名
        output_prefix: slurm 文件名前缀

    Returns:
        slurm 模板字符串列表
    """
    slurm_header = f'''#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={cpu_per_task}
#SBATCH --qos=normal
#SBATCH -p CPU

fastfetch

echo

df -Th


echo "Assigned GPU(s): $CUDA_VISIBLE_DEVICES"

echo '-------------------------------'
source  /curie/home/ansatz/.bashrc

# Below are executing commands
nvidia-smi dmon -s pucvmte -o T > nvdmon_job-$SLURM_JOB_ID.log &

source /software/envs/anaconda3.env

conda activate {conda_env}


export RENO_NUM_THREADS={cpu_per_task}
'''

    return slurm_header


def generate_slurm_files(n_batches: int,
                          cpu_per_task: int = 8,
                          conda_env: str = "reno-py310",
                          output_prefix: str = "reno_run_batch_") -> None:
    """
    生成所有 slurm 批处理文件

    Args:
        n_batches: 批处理数量
        cpu_per_task: 每个 CPU 任务的核心数
        conda_env: conda 环境名
        output_prefix: slurm 文件名前缀
    """
    slurm_header = generate_slurm_templates(cpu_per_task, conda_env, output_prefix)

    for batch_idx in range(n_batches):
        slurm_script = slurm_header + f'make -f p_run.make batch_{batch_idx} -j 1\n'

        output_file = f'{output_prefix}{batch_idx}.slurm'
        with open(output_file, 'w') as f:
            f.write(slurm_script)


# ============================================================================
# 清理旧文件
# ============================================================================

def cleanup_old_files(prefix: str = "reno_run_batch_") -> None:
    """
    清理旧的 slurm 文件

    Args:
        prefix: 文件名前缀
    """
    for filename in os.listdir('.'):
        if filename.startswith(prefix) and filename.endswith('.slurm'):
            os.remove(filename)


# ============================================================================
# 主生成函数
# ============================================================================

def generate_all_job_files(n_jobs: int,
                          ntasks_per_batch: int = 8,
                          cpu_per_task: int = 8,
                          conda_env: str = "reno-py310") -> None:
    """
    生成作业提交流程所需的所有文件

    Args:
        n_jobs: 总作业数
        ntasks_per_batch: 每个 batch 包含的作业数
        cpu_per_task: 每个 CPU 任务的核心数
        conda_env: conda 环境名
    """
    print(f"生成作业提交文件...")
    print(f"  总作业数: {n_jobs}")
    print(f"  每批作业数: {ntasks_per_batch}")

    n_batches = (n_jobs + ntasks_per_batch - 1) // ntasks_per_batch
    print(f"  批处理数: {n_batches}")

    # 生成 run.sh（示例）
    print(f"\n生成 run.sh...")
    # generate_run_script([...], "run.sh")

    # 生成 p_run.make
    print(f"生成 p_run.make...")
    generate_makefile(n_jobs, ntasks_per_batch, "p_run.make")

    # 清理旧 slurm 文件
    print(f"清理旧 slurm 文件...")
    cleanup_old_files()

    # 生成新的 slurm 文件
    print(f"生成 {n_batches} 个 slurm 文件...")
    generate_slurm_files(n_batches, cpu_per_task, conda_env)

    print(f"\n完成！")
    print(f"  提交命令示例:")
    for i in range(min(5, n_batches)):
        print(f"    sbatch reno_run_batch_{i}.slurm")


# ============================================================================
# 示例参数网格生成
# ============================================================================

def generate_phase_diagram_params(s_values: List[float],
                                alpha_values: List[float],
                                nmodes: int = 1000,
                                nsteps: int = 300) -> List[dict]:
    """
    生成相图扫描的参数网格

    Args:
        s_values: s 参数值列表
        alpha_values: alpha 参数值列表
        nmodes: 模式数
        nsteps: 时间步数

    Returns:
        参数列表
    """
    params = []
    for s in s_values:
        for alpha in alpha_values:
            params.append({
                's': s,
                'alpha': alpha,
                'nmodes': nmodes,
                'nsteps': nsteps,
                'rho_type': 0,
                'Omega': 1,
                'omega_c': 10,
                'bond_dims': 20,
                'td_method': 0,
                'calc_1site_entropy': True,
            })
    return params


# ============================================================================
# 命令行接口
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="生成作业提交文件")
    parser.add_argument('--n-jobs', type=int, default=90, help='总作业数')
    parser.add_argument('--ntasks-per-batch', type=int, default=8, help='每批作业数')
    parser.add_argument('--cpu-per-task', type=int, default=8, help='CPU核心数')
    parser.add_argument('--conda-env', type=str, default='reno-py310', help='conda环境名')
    args = parser.parse_args()

    generate_all_job_files(
        n_jobs=args.n_jobs,
        ntasks_per_batch=args.ntasks_per_batch,
        cpu_per_task=args.cpu_per_task,
        conda_env=args.conda_env
    )
