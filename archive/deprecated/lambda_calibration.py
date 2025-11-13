"""
λ↔bpf标定表工具
避免每个样本都进行二分搜索
"""

import torch
import numpy as np
from typing import List, Tuple, Callable
import logging

logger = logging.getLogger(__name__)


class LambdaCalibrator:
    """
    λ↔bpf标定器
    
    在校准集上建立λ→bpf的映射表，后续样本直接查表
    """
    
    def __init__(self):
        self.lambda_grid = []  # λ值列表（递减）
        self.bpf_grid = []     # 对应的bpf值列表（递增）
        self.calibrated = False
    
    def calibrate(
        self,
        encoder_fn: Callable,
        target_rates_bpf: List[float],
        num_lambda_points: int = 20,
        lambda_range: Tuple[float, float] = (0.0001, 16.0)
    ):
        """
        校准λ↔bpf映射表
        
        Args:
            encoder_fn: 编码函数，输入λ返回(indices, actual_bpf)
            target_rates_bpf: 目标码率列表
            num_lambda_points: λ采样点数
            lambda_range: λ范围
        """
        logger.info(f"开始λ↔bpf标定（{num_lambda_points}个点）...")
        
        # 对数均匀采样λ
        lambda_min, lambda_max = lambda_range
        lambda_samples = np.logspace(
            np.log10(lambda_min),
            np.log10(lambda_max),
            num=num_lambda_points
        )[::-1]  # 从大到小（对应bpf从小到大）
        
        # 测量每个λ对应的bpf
        bpf_samples = []
        valid_lambdas = []
        
        for lam in lambda_samples:
            try:
                _, actual_bpf = encoder_fn(lam)
                bpf_samples.append(actual_bpf)
                valid_lambdas.append(lam)
                logger.debug(f"  λ={lam:.4f} → {actual_bpf:.2f} bpf")
            except Exception as e:
                logger.warning(f"  λ={lam:.4f} 编码失败: {e}")
                continue
        
        # 存储（确保bpf单调递增）
        if len(valid_lambdas) > 0:
            # 按bpf排序
            sorted_pairs = sorted(zip(bpf_samples, valid_lambdas))
            self.bpf_grid, self.lambda_grid = zip(*sorted_pairs)
            self.bpf_grid = list(self.bpf_grid)
            self.lambda_grid = list(self.lambda_grid)
            self.calibrated = True
            
            logger.info(f"✅ 标定完成：")
            logger.info(f"  bpf范围: {min(self.bpf_grid):.2f} - {max(self.bpf_grid):.2f}")
            logger.info(f"  λ范围: {min(self.lambda_grid):.4f} - {max(self.lambda_grid):.4f}")
        else:
            logger.error("❌ 标定失败：无有效λ-bpf对")
            self.calibrated = False
    
    def lookup_lambda(self, target_bpf: float) -> float:
        """
        查找目标bpf对应的λ
        
        Args:
            target_bpf: 目标码率
        
        Returns:
            optimal_lambda: 对应的λ值
        """
        if not self.calibrated:
            logger.warning("标定表未初始化，返回默认λ=0.5")
            return 0.5
        
        # 线性插值查找
        if target_bpf <= self.bpf_grid[0]:
            # 低于最小bpf，返回最大λ
            return self.lambda_grid[0]
        elif target_bpf >= self.bpf_grid[-1]:
            # 高于最大bpf，返回最小λ
            return self.lambda_grid[-1]
        else:
            # 线性插值
            optimal_lambda = np.interp(
                target_bpf,
                self.bpf_grid,
                self.lambda_grid
            )
            return float(optimal_lambda)
    
    def lookup_with_refinement(
        self,
        encoder_fn: Callable,
        target_bpf: float,
        tolerance: float = 5.0,  # bpf容差
        max_refine_steps: int = 3
    ) -> Tuple[float, float]:
        """
        查表 + 可选微调
        
        Args:
            encoder_fn: 编码函数
            target_bpf: 目标码率
            tolerance: bpf容差（如果偏差>tolerance则微调）
            max_refine_steps: 最大微调步数
        
        Returns:
            (optimal_lambda, achieved_bpf)
        """
        # 查表
        initial_lambda = self.lookup_lambda(target_bpf)
        
        # 测试实际码率
        try:
            _, actual_bpf = encoder_fn(initial_lambda)
        except Exception as e:
            logger.error(f"编码失败: {e}")
            return initial_lambda, target_bpf
        
        error = abs(actual_bpf - target_bpf)
        
        # 如果误差在容差内，直接返回
        if error <= tolerance:
            return initial_lambda, actual_bpf
        
        # 否则微调
        logger.debug(f"微调λ: 初始λ={initial_lambda:.4f}, bpf={actual_bpf:.2f}, 目标={target_bpf:.2f}")
        
        current_lambda = initial_lambda
        for step in range(max_refine_steps):
            # 根据偏差方向调整λ
            if actual_bpf > target_bpf:
                # 码率过高，增大λ
                current_lambda *= 1.2
            else:
                # 码率过低，减小λ
                current_lambda *= 0.8
            
            # 测试新λ
            try:
                _, actual_bpf = encoder_fn(current_lambda)
                error = abs(actual_bpf - target_bpf)
                
                if error <= tolerance:
                    logger.debug(f"  微调成功（步数={step+1}）: λ={current_lambda:.4f}, bpf={actual_bpf:.2f}")
                    return current_lambda, actual_bpf
            except:
                break
        
        logger.debug(f"  微调后: λ={current_lambda:.4f}, bpf={actual_bpf:.2f}")
        return current_lambda, actual_bpf

