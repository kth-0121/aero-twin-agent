"""
601Solutions
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import math
import torch


@dataclass
class PhysicsFeedback:
    residual: torch.Tensor
    confidence: torch.Tensor | None = None


class PhysicsEvaluator(ABC):
    @abstractmethod
    def evaluate(self, process_x: torch.Tensor, predicted_state: torch.Tensor) -> PhysicsFeedback:
        raise NotImplementedError


class ZeroPhysicsEvaluator(PhysicsEvaluator):
    """아키텍처 검증 전용"""

    def evaluate(self, process_x, predicted_state):
        return PhysicsFeedback(residual=torch.zeros_like(predicted_state))


class RosenthalPhysicsEvaluator(PhysicsEvaluator): #Rosenthal의 열역학 모델
    """
    물리 모델:
        W = sqrt( 8/(pi*e) * eta*P / (rho*Cp*v*DeltaT) )
        R = D/W
        A = pi/4 * W * D

    평가:
        L=λW*LW​+λD​*LD​+λA*​LA​
    """

    def __init__(
        self,
        x_mean,
        x_scale,
        y_mean,
        y_scale,
        absorptivity: float = 0.34,
        density: float = 7800.0,
        heat_capacity: float = 460.0,
        thermal_diffusivity: float = 5.0e-6,
        melting_temperature: float = 1678.0,
        preheat_temperature: float = 293.15,
        scaling_k: float = 0.60,
        residual_clip: float = 5.0,
    ):
        self.x_mean = torch.as_tensor(x_mean, dtype=torch.float32)
        self.x_scale = torch.as_tensor(x_scale, dtype=torch.float32)
        self.y_mean = torch.as_tensor(y_mean, dtype=torch.float32)
        self.y_scale = torch.as_tensor(y_scale, dtype=torch.float32)
        self.absorptivity = absorptivity
        self.density = density
        self.heat_capacity = heat_capacity
        self.thermal_diffusivity = thermal_diffusivity
        self.melting_temperature = melting_temperature
        self.preheat_temperature = preheat_temperature
        self.scaling_k = scaling_k
        self.residual_clip = residual_clip

    def _physical_x(self, x: torch.Tensor) -> torch.Tensor:
        mean = self.x_mean.to(device=x.device, dtype=x.dtype)
        scale = self.x_scale.to(device=x.device, dtype=x.dtype)
        return x * scale + mean

    def _physical_y(self, y: torch.Tensor) -> torch.Tensor:
        mean = self.y_mean.to(device=y.device, dtype=y.dtype)
        scale = self.y_scale.to(device=y.device, dtype=y.dtype)
        return y * scale + mean

    def thermal_features(self, process_x: torch.Tensor):
        x = self._physical_x(process_x)
        P = x[:, 0].clamp_min(1e-6)
        v = (x[:, 1] / 1000.0).clamp_min(1e-9)
        diameter = (x[:, 2] / 1000.0).clamp_min(1e-9)

        Pe = v * diameter / self.thermal_diffusivity
        hs_vol = self.density * self.heat_capacity * (
            self.melting_temperature - self.preheat_temperature
        )
        h_star = self.absorptivity * P / (
            hs_vol * torch.sqrt(self.thermal_diffusivity * diameter**3 * v)
        )
        return P, v, diameter, Pe, h_star

    def rosenthal_width(self, process_x: torch.Tensor) -> torch.Tensor:
        """P 및 v 원시 입력을 사용하여 로젠탈 폭을 마이크로미터 단위로 반환합니다."""
        P, v, _, _, _ = self.thermal_features(process_x)
        delta_T = self.melting_temperature - self.preheat_temperature
        width_m = torch.sqrt(
            (8.0 / (math.pi * math.e))
            * self.absorptivity * P
            / (self.density * self.heat_capacity * v * delta_T)
        )
        return width_m * 1e6

    def scaling_depth(self, process_x: torch.Tensor) -> torch.Tensor:
        """마이크로미터 단위의 soft 스케일링 depth 추정치를 반환합니다.

        Using beta = H*/pi and L_th = Pe^(-1/2),
            D / d_laser ~= K * beta * L_th.

        분석용이지 Target은 아님에 주의.
        """
        _, _, diameter, Pe, h_star = self.thermal_features(process_x)
        beta = h_star / math.pi
        normalized_depth = self.scaling_k * beta / torch.sqrt(Pe.clamp_min(1e-12))
        return diameter * normalized_depth * 1e6

    def evaluate(self, process_x: torch.Tensor, predicted_state: torch.Tensor) -> PhysicsFeedback:
        pred = self._physical_y(predicted_state)
        pred_w, pred_r, pred_a = pred.unbind(dim=-1)
        eps = torch.tensor(1e-6, device=pred.device, dtype=pred.dtype)

        ros_w = self.rosenthal_width(process_x)
        phys_d = self.scaling_depth(process_x)
        phys_r = phys_d / pred_w.clamp_min(eps)
        phys_r = phys_r.clamp(min=0.15, max=2.5)
        pred_d = pred_w * pred_r

        r_width = (pred_w - ros_w) / (ros_w.abs() + eps)

        # 로그-비 잔차는 큰 종횡비 값이 피드백을 지배하는 것을 방지하고
        # 기하학적 오차를 대칭적으로 처리합니다.
        raw_r_residual = (
            torch.log(pred_r.clamp_min(1e-4))
            - torch.log(phys_r.clamp_min(1e-4))
        )

        # 분석 법칙이 모델 학습을 압도하지 않도록 기하 신뢰도를 조절합니다.
        R_PHYSICS_CONFIDENCE = 0.25
        r_aspect = R_PHYSICS_CONFIDENCE * raw_r_residual

        # 타원 면적 일관성 식: A ≈ pi/4 * W^2 * R (W * D)
        ellipse_area = math.pi / 4.0 * pred_w.clamp_min(eps) * pred_d.clamp_min(eps)
        r_area = (pred_a - ellipse_area) / (ellipse_area.abs() + eps)

        residual = torch.stack([r_width, r_aspect, r_area], dim=-1)
        residual = torch.clamp(residual, -self.residual_clip, self.residual_clip)

        # Balling 또는 Extreme R 상태(예: R < 0.2 또는 R > 2.0)에서는 단순 타원 가정의 신뢰도를 감쇄합니다.
        regime_penalty = torch.ones_like(pred_r)
        extreme_mask = (pred_r < 0.2) | (pred_r > 2.0)
        regime_penalty[extreme_mask] = 0.1

        # R_PHYSICS_CONFIDENCE 및 면적 일관성 신뢰도에 물리적 영역 패널티를 곱해줍니다.
        confidence_r_tensor = torch.full_like(r_width, R_PHYSICS_CONFIDENCE) * regime_penalty
        confidence_area = torch.ones_like(r_area) * regime_penalty

        confidence = torch.stack(
            [torch.ones_like(r_width), confidence_r_tensor, confidence_area], dim=-1
        )
        return PhysicsFeedback(residual=residual, confidence=confidence)

class FluidThermalPhysicsEvaluator(RosenthalPhysicsEvaluator): # 유체역학 모델
    """
    Thermal + Fluid Physics Evaluator.
    Computes Marangoni (Ma), Reynolds (Re), Weber (We), and Capillary (Ca) numbers
    alongside Normalized Enthalpy and Rosenthal Width discrepancies.

    Feedback vector structure to LoopedTransformer:
        [width_discrepancy, aspect_ratio_residual, area_discrepancy, Ma_normalized, Re_normalized, We_normalized, Ca_normalized]
    """
    def __init__(
        self,
        x_mean,
        x_scale,
        y_mean,
        y_scale,
        absorptivity: float = 0.34,
        density: float = 7800.0,              # kg/m^3
        heat_capacity: float = 460.0,          # J/(kg K)
        thermal_diffusivity: float = 5.0e-6,   # m^2/s
        melting_temperature: float = 1678.0,   # K
        preheat_temperature: float = 293.15,   # K
        viscosity: float = 6.0e-3,             # Pa s (liquid metal viscosity)
        surface_tension: float = 1.2,          # N/m
        dsigma_dT: float = -2.6e-4,            # N/(m K) (Marangoni coefficient)
        scaling_k: float = 0.60,
        residual_clip: float = 5.0
    ):
        super().__init__(
            x_mean, x_scale, y_mean, y_scale,
            absorptivity, density, heat_capacity,
            thermal_diffusivity, melting_temperature, preheat_temperature,
            scaling_k, residual_clip
        )
        self.viscosity = viscosity
        self.surface_tension = surface_tension
        self.dsigma_dT = abs(dsigma_dT)

    def evaluate(self, process_x: torch.Tensor, predicted_state: torch.Tensor) -> PhysicsFeedback:
        # Thermal features and physical values reconstruction
        pred = self._physical_y(predicted_state)
        pred_w, pred_r, pred_a = pred.unbind(dim=-1)
        eps = torch.tensor(1e-6, device=pred.device, dtype=pred.dtype)

        P, v, diameter, Pe, h_star = self.thermal_features(process_x)

        # Temperature rise estimation (Delta T)
        # Reference DeltaT = (Tm - T0) scaled with normalized enthalpy to account for power input
        delta_T_ref = self.melting_temperature - self.preheat_temperature
        DeltaT = (delta_T_ref * (1.0 + 0.1 * h_star)).clamp_min(100.0)

        # Melt pool characteristic length (L = melt width in meters)
        L = (pred_w / 1e6).clamp_min(1e-6)

        # Fluid dimensionless numbers calculation
        # Ma = |dsigma_dT| * DeltaT * L / (viscosity * alpha)
        Ma = (self.dsigma_dT * DeltaT * L) / (self.viscosity * self.thermal_diffusivity + eps)

        # Estimated Marangoni velocity U = |dsigma_dT| * DeltaT / viscosity
        U_marangoni = (self.dsigma_dT * DeltaT / (self.viscosity + eps)).clamp_max(10.0) # bound unrealistic speeds

        # Re = rho * U * L / viscosity
        Re = (self.density * U_marangoni * L) / (self.viscosity + eps)

        # We = rho * U^2 * L / surface_tension
        We = (self.density * (U_marangoni ** 2) * L) / (self.surface_tension + eps)

        # Ca = viscosity * U / surface_tension
        Ca = (self.viscosity * U_marangoni) / (self.surface_tension + eps)

        # Rosenthal core evaluations (discrepancy & aspect ratio & area consistency)
        ros_w = self.rosenthal_width(process_x)
        r_width = (pred_w - ros_w) / (ros_w.abs() + eps)

        phys_d = self.scaling_depth(process_x)
        phys_r = (phys_d / pred_w.clamp_min(eps)).clamp(0.15, 2.5)
        raw_r_residual = torch.log(pred_r.clamp_min(1e-4)) - torch.log(phys_r.clamp_min(1e-4))
        r_aspect = 0.25 * raw_r_residual

        pred_d = pred_w * pred_r
        ellipse_area = (math.pi / 4.0) * pred_w.clamp_min(eps) * pred_d.clamp_min(eps)
        r_area = (pred_a - ellipse_area) / (ellipse_area.abs() + eps)

        # Normalize fluid features to avoid high scalar values overpowering gradient update
        Ma_scaled = torch.log10(Ma.clamp_min(1.0)) / 4.0   # target scale: 0 ~ 1 for Ma ~ 10^4
        Re_scaled = torch.log10(Re.clamp_min(0.1)) / 3.0   # target scale
        We_scaled = We.clamp_max(10.0) / 10.0
        Ca_scaled = Ca.clamp_max(0.1) * 10.0

        # Composite Feedback Tensor construction [B, 7]
        residual = torch.stack([
            r_width,
            r_aspect,
            r_area,
            Ma_scaled,
            Re_scaled,
            We_scaled,
            Ca_scaled
        ], dim=-1)
        residual = torch.clamp(residual, -self.residual_clip, self.residual_clip)

        # Geometric and physical regime penalty masking
        regime_penalty = torch.ones_like(pred_r)
        extreme_mask = (pred_r < 0.2) | (pred_r > 2.0)
        regime_penalty[extreme_mask] = 0.1

        # Assign confidence multipliers
        confidence = torch.stack([
            torch.ones_like(r_width),
            torch.full_like(r_width, 0.25) * regime_penalty,
            torch.ones_like(r_area) * regime_penalty,
            torch.full_like(r_width, 0.1),  # fluid numbers are guides
            torch.full_like(r_width, 0.1),
            torch.full_like(r_width, 0.1),
            torch.full_like(r_width, 0.1)
        ], dim=-1)

        return PhysicsFeedback(residual=residual, confidence=confidence)

class FluidThermalPhysicsEvaluator(RosenthalPhysicsEvaluator): # 열-유체 모델
    """
    Thermal + Fluid Physics Evaluator with explicit progressive loop-wise reasoning:
      - Loop 0: Initial physical-state estimation (No feedback or zero feedback)
      - Loop 1: Thermal consistency refinement (Focus on Rosenthal width discrepancy direction)
      - Loop 2: Fluid/Morphology consistency refinement (Focus on flow regime & aspect ratio discrepancy)
      - Loop 3+: Joint physical consistency refinement (Coupled thermal, fluid, and geometric area consistency)
    """
    def __init__(
        self,
        x_mean,
        x_scale,
        y_mean,
        y_scale,
        absorptivity: float = 0.34,
        density: float = 7800.0,              # kg/m^3
        heat_capacity: float = 460.0,          # J/(kg K)
        thermal_diffusivity: float = 5.0e-6,   # m^2/s
        melting_temperature: float = 1678.0,   # K
        preheat_temperature: float = 293.15,   # K
        viscosity: float = 6.0e-3,             # Pa s
        surface_tension: float = 1.2,          # N/m
        dsigma_dT: float = -2.6e-4,            # N/(m K)
        scaling_k: float = 0.60,
        residual_clip: float = 5.0
    ):
        super().__init__(
            x_mean, x_scale, y_mean, y_scale,
            absorptivity, density, heat_capacity,
            thermal_diffusivity, melting_temperature, preheat_temperature,
            scaling_k, residual_clip
        )
        self.viscosity = viscosity
        self.surface_tension = surface_tension
        self.dsigma_dT = abs(dsigma_dT)

    def evaluate_loop_wise(self, process_x: torch.Tensor, predicted_state: torch.Tensor, loop_idx: int) -> PhysicsFeedback:
        # Unpack predictions in physical scale
        pred = self._physical_y(predicted_state)
        pred_w, pred_r, pred_a = pred.unbind(dim=-1)
        eps = torch.tensor(1e-6, device=pred.device, dtype=pred.dtype)

        # Initialize zero residuals
        r_width = torch.zeros_like(pred_w)
        r_aspect = torch.zeros_like(pred_r)
        r_area = torch.zeros_like(pred_a)
        Ma_scaled = torch.zeros_like(pred_w)
        Re_scaled = torch.zeros_like(pred_w)
        We_scaled = torch.zeros_like(pred_w)
        Ca_scaled = torch.zeros_like(pred_w)

        # Confidence mappings
        conf_width = torch.zeros_like(pred_w)
        conf_aspect = torch.zeros_like(pred_r)
        conf_area = torch.zeros_like(pred_a)
        conf_fluid = torch.zeros_like(pred_w)

        # Loop 0: Initial physical-state estimation (Pure data-driven, zero physics feedback influence)
        if loop_idx == 0:
            residual = torch.stack([r_width, r_aspect, r_area, Ma_scaled, Re_scaled, We_scaled, Ca_scaled], dim=-1)
            confidence = torch.stack([conf_width, conf_aspect, conf_area, conf_fluid, conf_fluid, conf_fluid, conf_fluid], dim=-1)
            return PhysicsFeedback(residual=residual, confidence=confidence)

        # Extract process parameters
        P, v, diameter, Pe, h_star = self.thermal_features(process_x)

        # Loop 1: Thermal consistency refinement
        if loop_idx == 1:
            ros_w = self.rosenthal_width(process_x)
            # Directional correction: positive if pred_w is smaller than thermal scale, negative if larger
            r_width = (ros_w - pred_w) / (ros_w.abs() + eps)
            conf_width = torch.ones_like(r_width)

        # Loop 2: Fluid/Morphology consistency refinement
        elif loop_idx == 2:
            # Calculate fluid dimensionless numbers for guidance
            delta_T_ref = self.melting_temperature - self.preheat_temperature
            DeltaT = (delta_T_ref * (1.0 + 0.1 * h_star)).clamp_min(100.0)
            L = (pred_w / 1e6).clamp_min(1e-6)

            Ma = (self.dsigma_dT * DeltaT * L) / (self.viscosity * self.thermal_diffusivity + eps)
            U_marangoni = (self.dsigma_dT * DeltaT / (self.viscosity + eps)).clamp_max(10.0)
            Re = (self.density * U_marangoni * L) / (self.viscosity + eps)
            We = (self.density * (U_marangoni ** 2) * L) / (self.surface_tension + eps)
            Ca = (self.viscosity * U_marangoni) / (self.surface_tension + eps)

            Ma_scaled = torch.log10(Ma.clamp_min(1.0)) / 4.0
            Re_scaled = torch.log10(Re.clamp_min(0.1)) / 3.0
            We_scaled = We.clamp_max(10.0) / 10.0
            Ca_scaled = Ca.clamp_max(0.1) * 10.0

            # Directional correction for aspect ratio based on fluid dynamics scaling
            phys_d = self.scaling_depth(process_x)
            phys_r = (phys_d / pred_w.clamp_min(eps)).clamp(0.15, 2.5)
            # Correction signal: positive if pred_r is smaller than physical aspect ratio, negative if larger
            r_aspect = torch.log(phys_r.clamp_min(1e-4)) - torch.log(pred_r.clamp_min(1e-4))

            # Apply regime penalty to confidence
            regime_penalty = torch.ones_like(pred_r)
            extreme_mask = (pred_r < 0.2) | (pred_r > 2.0)
            regime_penalty[extreme_mask] = 0.1

            conf_aspect = torch.full_like(r_aspect, 0.25) * regime_penalty
            conf_fluid = torch.full_like(r_width, 0.1)

        # Loop 3+: Joint physical consistency refinement (Coupled loop)
        else:
            # Thermal width discrepancy
            ros_w = self.rosenthal_width(process_x)
            r_width = (ros_w - pred_w) / (ros_w.abs() + eps)

            # Fluid numbers & aspect ratio discrepancy
            delta_T_ref = self.melting_temperature - self.preheat_temperature
            DeltaT = (delta_T_ref * (1.0 + 0.1 * h_star)).clamp_min(100.0)
            L = (pred_w / 1e6).clamp_min(1e-6)
            Ma = (self.dsigma_dT * DeltaT * L) / (self.viscosity * self.thermal_diffusivity + eps)
            U_marangoni = (self.dsigma_dT * DeltaT / (self.viscosity + eps)).clamp_max(10.0)
            Re = (self.density * U_marangoni * L) / (self.viscosity + eps)
            We = (self.density * (U_marangoni ** 2) * L) / (self.surface_tension + eps)
            Ca = (self.viscosity * U_marangoni) / (self.surface_tension + eps)

            Ma_scaled = torch.log10(Ma.clamp_min(1.0)) / 4.0
            Re_scaled = torch.log10(Re.clamp_min(0.1)) / 3.0
            We_scaled = We.clamp_max(10.0) / 10.0
            Ca_scaled = Ca.clamp_max(0.1) * 10.0

            phys_d = self.scaling_depth(process_x)
            phys_r = (phys_d / pred_w.clamp_min(eps)).clamp(0.15, 2.5)
            r_aspect = torch.log(phys_r.clamp_min(1e-4)) - torch.log(pred_r.clamp_min(1e-4))

            # Geometric area consistency
            pred_d = pred_w * pred_r
            ellipse_area = (math.pi / 4.0) * pred_w.clamp_min(eps) * pred_d.clamp_min(eps)
            # Correction signal: positive if pred_a is smaller than geometric ellipse area, negative if larger
            r_area = (ellipse_area - pred_a) / (ellipse_area.abs() + eps)

            regime_penalty = torch.ones_like(pred_r)
            extreme_mask = (pred_r < 0.2) | (pred_r > 2.0)
            regime_penalty[extreme_mask] = 0.1

            conf_width = torch.ones_like(r_width)
            conf_aspect = torch.full_like(r_aspect, 0.25) * regime_penalty
            conf_area = torch.ones_like(r_area) * regime_penalty
            conf_fluid = torch.full_like(r_width, 0.1)

        # Stack and construct feedback
        residual = torch.stack([r_width, r_aspect, r_area, Ma_scaled, Re_scaled, We_scaled, Ca_scaled], dim=-1)
        residual = torch.clamp(residual, -self.residual_clip, self.residual_clip)

        confidence = torch.stack([conf_width, conf_aspect, conf_area, conf_fluid, conf_fluid, conf_fluid, conf_fluid], dim=-1)

        return PhysicsFeedback(residual=residual, confidence=confidence)

    def evaluate(self, process_x: torch.Tensor, predicted_state: torch.Tensor) -> PhysicsFeedback:
        # Default fallback to coupled joint evaluation
        return self.evaluate_loop_wise(process_x, predicted_state, loop_idx=3)

class DeltaFeedbackEvaluator(PhysicsEvaluator):
    """Debug 전용"""

    def __init__(self, state_dim: int = 3):
        self.state_dim = state_dim
        self.previous = None

    def reset(self):
        self.previous = None

    def evaluate(self, process_x, predicted_state):
        if self.previous is None:
            residual = torch.zeros_like(predicted_state)
        else:
            residual = predicted_state - self.previous
        self.previous = predicted_state.detach()
        return PhysicsFeedback(residual=residual)
