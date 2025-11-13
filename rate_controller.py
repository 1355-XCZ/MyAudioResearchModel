"""
码率控制器
实现对偶更新和二分搜索来达到目标码率
"""

import torch
import numpy as np
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class RateController:
    """
    码率控制器
    
    通过调整拉格朗日乘子 λ 来控制码率:
        - 训练期：对偶更新 λ ← [λ + η(R - r_target)]₊
        - 推理期：二分搜索找到最优 λ
    """
    
    def __init__(self, config):
        """
        Args:
            config: RateControlConfig实例
        """
        self.config = config
        
        # 当前 λ 值
        self.lambda_current = config.lambda_init
        
        # 对偶更新参数
        self.dual_lr = config.dual_lr
        self.dual_momentum = config.dual_momentum
        self.lambda_velocity = 0.0  # 动量项
        
        # 二分搜索范围
        self.lambda_min = config.lambda_min
        self.lambda_max = config.lambda_max
        
        logger.info(f"✅ RateController 初始化:")
        logger.info(f"   - λ 初始值: {self.lambda_current:.4f}")
        logger.info(f"   - λ 范围: [{self.lambda_min:.4f}, {self.lambda_max:.4f}]")
        logger.info(f"   - 对偶学习率: {self.dual_lr}")
    
    def dual_update(self, current_rate_bpf: float, target_rate_bpf: float) -> float:
        """
        对偶更新 λ（训练期使用）
        
        λ ← [λ + η(R_current - R_target)]₊
        
        Args:
            current_rate_bpf: 当前码率（bits/frame）
            target_rate_bpf: 目标码率（bits/frame）
        
        Returns:
            new_lambda: 更新后的 λ
        """
        # 计算梯度（码率误差）
        grad = current_rate_bpf - target_rate_bpf
        
        # 动量更新
        self.lambda_velocity = (
            self.dual_momentum * self.lambda_velocity +
            (1 - self.dual_momentum) * grad
        )
        
        # 更新 λ
        self.lambda_current = max(
            self.lambda_min,
            self.lambda_current + self.dual_lr * self.lambda_velocity
        )
        
        logger.debug(f"对偶更新: R={current_rate_bpf:.2f} (目标={target_rate_bpf:.2f}), "
                     f"λ={self.lambda_current:.4f}")
        
        return self.lambda_current
    
    def binary_search(
        self,
        encoder_fn,
        target_rate_bpf: float,
        max_iters: Optional[int] = None,
        tolerance_bpf: Optional[float] = None,
        use_cache: bool = True,
        lambda_hint: Optional[float] = None  # 先验λ值（来自第一个样本）
    ) -> Tuple[float, float]:
        """
        二分搜索找到达到目标码率的 λ（推理期使用，支持缓存）
        
        Args:
            encoder_fn: 函数 λ -> (indices, rate_bpf)
            target_rate_bpf: 目标码率（bpf = bits per frame）
            max_iters: 最大迭代次数
            tolerance_bpf: 码率容差（bpf）
            use_cache: 是否使用缓存（避免重复编码）
        
        Returns:
            (optimal_lambda, achieved_rate_bpf)
        """
        if max_iters is None:
            max_iters = self.config.max_binary_search_iters
        if tolerance_bpf is None:
            tolerance_bpf = self.config.rate_tolerance_bpf
        
        # 缓存：避免重复编码同一个λ（提速3-5倍）
        # 注意：缓存应该在每次新的目标码率搜索时清空
        cache = {} if use_cache else None
        
        def cached_encoder_fn(lam):
            # 量化缓存键，提高缓存命中率（避免浮点精度问题）
            key = round(lam, 6)  # 保留6位小数
            if cache is not None and key in cache:
                return cache[key]
            result = encoder_fn(lam)
            if cache is not None:
                cache[key] = result
            return result
        
        # 如果有先验λ（从第一个样本），优先测试它
        if lambda_hint is not None:
            _, hint_rate = cached_encoder_fn(lambda_hint)
            if abs(hint_rate - target_rate_bpf) < tolerance_bpf:
                # 先验λ已满足，直接返回
                logger.debug(f"  使用先验λ={lambda_hint:.4f}, R={hint_rate:.2f} bpf (目标={target_rate_bpf:.2f})")
                self.lambda_current = lambda_hint
                return lambda_hint, hint_rate
            # 否则以hint为中心缩小搜索范围
            lambda_low = max(self.lambda_min, lambda_hint * 0.5)
            lambda_high = min(self.lambda_max, lambda_hint * 2.0)
        else:
            lambda_low = self.lambda_min
            lambda_high = self.lambda_max
        
        best_lambda = lambda_hint if lambda_hint else self.lambda_current
        best_rate = None
        
        for iter_idx in range(max_iters):
            # 中点
            lambda_mid = (lambda_low + lambda_high) / 2
            
            # 编码并测量码率（使用缓存）
            _, rate_bpf = cached_encoder_fn(lambda_mid)
            
            # 记录最接近的
            if best_rate is None or abs(rate_bpf - target_rate_bpf) < abs(best_rate - target_rate_bpf):
                best_lambda = lambda_mid
                best_rate = rate_bpf
            
            # 检查收敛
            if abs(rate_bpf - target_rate_bpf) < tolerance_bpf:
                logger.info(f"✅ 二分搜索收敛（iter={iter_idx+1}）: "
                           f"λ={lambda_mid:.4f}, R={rate_bpf:.2f} bpf (目标={target_rate_bpf:.2f})")
                self.lambda_current = lambda_mid  # 提交最优λ到状态
                return lambda_mid, rate_bpf
            
            # 更新搜索区间
            # 注意：λ越大，码率越低（SKIP更多）
            if rate_bpf > target_rate_bpf:
                # 码率太高，需要增大 λ
                lambda_low = lambda_mid
            else:
                # 码率太低，需要减小 λ
                lambda_high = lambda_mid
            
            logger.debug(f"二分搜索 iter={iter_idx+1}: λ={lambda_mid:.4f}, "
                        f"R={rate_bpf:.2f} bpf (目标={target_rate_bpf:.2f}), "
                        f"区间=[{lambda_low:.4f}, {lambda_high:.4f}]")
        
        logger.warning(f"⚠️ 二分搜索未收敛（{max_iters}次迭代，可接受）: "
                      f"最佳λ={best_lambda:.4f}, R={best_rate:.2f} bpf (目标={target_rate_bpf:.2f})")
        
        # 打印缓存效率
        if cache is not None and len(cache) > 0:
            logger.debug(f"二分搜索缓存: {len(cache)} 个λ值被缓存（避免重复编码）")
        
        self.lambda_current = best_lambda  # 即使未收敛，也提交最佳λ
        return best_lambda, best_rate
    
    def reset(self):
        """重置控制器状态"""
        self.lambda_current = self.config.lambda_init
        self.lambda_velocity = 0.0


class RateDistortionCurve:
    """
    率失真曲线
    记录和分析 (λ, R, D) 的关系
    """
    
    def __init__(self):
        self.records = []  # [(lambda, rate_bps, distortion, info), ...]
    
    def add_point(
        self,
        lambda_val: float,
        rate_bps: float,
        distortion: float,
        info: Optional[dict] = None
    ):
        """添加一个点"""
        record = {
            'lambda': lambda_val,
            'rate_bps': rate_bps,
            'distortion': distortion,
            'info': info or {}
        }
        self.records.append(record)
    
    def get_curve(self, sort_by: str = 'rate_bps'):
        """
        获取排序后的曲线
        
        Args:
            sort_by: 'rate_bps' or 'lambda'
        
        Returns:
            (lambdas, rates, distortions)
        """
        sorted_records = sorted(self.records, key=lambda x: x[sort_by])
        
        lambdas = np.array([r['lambda'] for r in sorted_records])
        rates = np.array([r['rate_bps'] for r in sorted_records])
        distortions = np.array([r['distortion'] for r in sorted_records])
        
        return lambdas, rates, distortions
    
    def interpolate_lambda(self, target_rate_bps: float) -> float:
        """
        线性插值找到达到目标码率的 λ
        
        Args:
            target_rate_bps: 目标码率
        
        Returns:
            interpolated_lambda
        """
        if len(self.records) < 2:
            raise ValueError("至少需要2个点才能插值")
        
        lambdas, rates, _ = self.get_curve(sort_by='rate_bps')
        
        # 找到包围目标的两个点
        if target_rate_bps <= rates[0]:
            return lambdas[0]
        if target_rate_bps >= rates[-1]:
            return lambdas[-1]
        
        # 线性插值
        idx = np.searchsorted(rates, target_rate_bps)
        r0, r1 = rates[idx-1], rates[idx]
        l0, l1 = lambdas[idx-1], lambdas[idx]
        
        alpha = (target_rate_bps - r0) / (r1 - r0)
        lambda_interp = l0 + alpha * (l1 - l0)
        
        return lambda_interp
    
    def save(self, filepath: str):
        """保存曲线数据"""
        import json
        with open(filepath, 'w') as f:
            json.dump(self.records, f, indent=2)
        logger.info(f"✅ R-D曲线已保存: {filepath}")
    
    def load(self, filepath: str):
        """加载曲线数据"""
        import json
        with open(filepath, 'r') as f:
            self.records = json.load(f)
        logger.info(f"✅ R-D曲线已加载: {filepath} ({len(self.records)}个点)")


def test_rate_controller():
    """测试码率控制器"""
    from config import RateControlConfig
    
    config = RateControlConfig()
    controller = RateController(config)
    
    print("\n=== 测试对偶更新 ===")
    target_rate = 5.0
    for step in range(10):
        # 模拟当前码率（逐渐逼近目标）
        current_rate = 10.0 - step * 0.5
        lambda_new = controller.dual_update(current_rate, target_rate)
        print(f"Step {step+1}: R={current_rate:.2f}, λ={lambda_new:.4f}")
    
    print("\n=== 测试二分搜索 ===")
    controller.reset()
    
    # 模拟编码函数（λ越大，码率越低）
    def mock_encoder(lam):
        rate = 20.0 / (1 + lam)  # 反比关系
        return None, rate
    
    target_rate_bps = 5.0
    optimal_lambda, achieved_rate = controller.binary_search(
        mock_encoder,
        target_rate_bps,
        max_iters=10,
        tolerance_bps=0.5
    )
    print(f"最优 λ={optimal_lambda:.4f}, 达到码率={achieved_rate:.2f} bps")


if __name__ == "__main__":
    test_rate_controller()

