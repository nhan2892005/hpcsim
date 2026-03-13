"""
Renewable Energy Module — GAS-MARL Green-Aware HPC Scheduling.

Models solar and wind power generation following:
  Chen et al., "GAS-MARL: Green-Aware job Scheduling for HPC clusters
  based on Multi-Action Deep Reinforcement Learning", FGCS 2025.

Power supply:
  P_solar(s) = α × A × irr(s)
  P_wind(s)  = piecewise function of wind speed
  P_cluster(s) = P_idle + Σ pow_j(s)   [per-job power consumption]
  ReUtil = ∫ min(P_green, P_cluster) dt / ∫ P_cluster dt

Synthesis approach: diurnal + stochastic patterns calibrated so that
average ReUtil hovers ~0.65–0.80 when jobs are spread across the day.
"""

from __future__ import annotations
import math
import random
from dataclasses import dataclass, field
from typing import Optional


# Solar model parameters 

@dataclass
class SolarModel:
    """Photovoltaic panel parameters (GAS-MARL Table 2)."""
    efficiency: float = 0.20          # α — PV conversion efficiency
    area_m2: float = 200.0            # A — effective irradiated area (m²)
    peak_irradiance_w_m2: float = 1000.0   # clear-sky peak

    def power_watts(self, irradiance: float) -> float:
        return self.efficiency * self.area_m2 * irradiance


# Wind model parameters 

@dataclass
class WindModel:
    """Wind turbine parameters (GAS-MARL Table 2)."""
    rated_power_w: float  = 7_200.0   # P_r
    rated_speed_ms: float = 15.0       # v_r
    cut_in_ms: float = 2.5             # v_in
    cut_out_ms: float = 30.0           # v_out

    def power_watts(self, wind_speed_ms: float) -> float:
        v = wind_speed_ms
        if v <= self.cut_in_ms or v >= self.cut_out_ms:
            return 0.0
        if v < self.rated_speed_ms:
            return self.rated_power_w * (v - self.cut_in_ms) / (self.rated_speed_ms - self.cut_in_ms)
        return self.rated_power_w


# Configuration 

@dataclass
class RenewableConfig:
    solar: SolarModel = field(default_factory=SolarModel)
    wind: WindModel = field(default_factory=WindModel)
    slot_duration_sec: float = 3600.0    # 1-hour slots
    forecast_horizon_slots: int = 24     # 24-hour forecast
    # Scale factor applied proportionally to cluster size (# GPUs / ref)
    scale_ref_gpus: int = 256
    rng_seed: Optional[int] = None


# Renewable Energy Module 

class RenewableEnergyModule:
    """
    Generates renewable energy time-series for a simulation run.

    Usage:
        re = RenewableEnergyModule(config, total_gpus=112, sim_duration=7200.0)
        power = re.available_power_watts(current_time)
        forecast = re.get_forecast(current_time)   # list of (remaining_sec, power_w)

    The module generates synthetic solar + wind traces with:
      - Solar: diurnal bell curve peaking ~noon with daily variation
      - Wind: Weibull-distributed speeds with gentle autocorrelation
    Both are scaled proportionally to cluster size.
    """

    def __init__(
        self,
        config: Optional[RenewableConfig] = None,
        total_gpus: int = 112,
        sim_duration: float = 7200.0,
    ):
        self.config = config or RenewableConfig()
        self.total_gpus = total_gpus
        self.sim_duration = sim_duration
        self.rng = random.Random(self.config.rng_seed or 2025)

        # GPU scale factor vs reference 256 processors
        self._scale = total_gpus / self.config.scale_ref_gpus

        # Pre-generate hourly power series for the whole simulation + 24-h ahead
        total_slots = math.ceil(sim_duration / self.config.slot_duration_sec) + 48
        self._solar_kw: list[float] = []
        self._wind_kw: list[float] = []
        self._total_kw: list[float] = []
        self._generate(total_slots)

    # Generation 

    def _solar_irradiance(self, hour_of_day: float) -> float:
        """
        Diurnal solar irradiance (W/m²) – bell curve 6:00–20:00.
        Includes day-to-day variability and cloud cover factor.
        """
        if hour_of_day < 6 or hour_of_day > 20:
            return 0.0
        angle = math.pi * (hour_of_day - 6) / 14.0  # 0 at 6:00, π at 20:00
        clear_sky = self.config.solar.peak_irradiance_w_m2 * math.sin(angle)
        cloud_cover = 0.6 + 0.4 * self.rng.random()   # 60–100% clear
        return max(0.0, clear_sky * cloud_cover)

    def _wind_speed(self, prev_speed: Optional[float], hour_of_day: float) -> float:
        """
        Autocorrelated wind speed with diurnal pattern (m/s).
        Weibull-distributed base with gentle autocorrelation.
        """
        # Diurnal pattern: slightly higher wind in afternoon
        base = 6.0 + 3.0 * math.sin(math.pi * (hour_of_day - 6) / 18.0)
        sample = self.rng.weibullvariate(base, 2.0)
        if prev_speed is not None:
            sample = 0.7 * prev_speed + 0.3 * sample   # AR(1) smoothing
        return max(0.0, min(sample, 35.0))

    def _generate(self, total_slots: int) -> None:
        prev_wind = None
        # Simulate starting at a random hour of day for variety
        start_hour = self.rng.uniform(0, 23)
        for slot in range(total_slots):
            hour_of_day = (start_hour + slot) % 24
            irr = self._solar_irradiance(hour_of_day)
            wind_speed = self._wind_speed(prev_wind, hour_of_day)
            prev_wind = wind_speed

            solar_w = self.config.solar.power_watts(irr) * self._scale
            wind_w  = self.config.wind.power_watts(wind_speed) * self._scale

            self._solar_kw.append(solar_w / 1000.0)
            self._wind_kw.append(wind_w  / 1000.0)
            self._total_kw.append((solar_w + wind_w) / 1000.0)

    # Public API 

    def _slot_index(self, sim_time_sec: float) -> int:
        idx = int(sim_time_sec / self.config.slot_duration_sec)
        return min(idx, len(self._total_kw) - 1)

    def available_power_watts(self, sim_time_sec: float) -> float:
        """Instantaneous renewable power available at sim_time_sec."""
        return self._total_kw[self._slot_index(sim_time_sec)] * 1000.0

    def get_forecast(
        self, sim_time_sec: float
    ) -> list[tuple[float, float]]:
        """
        Returns 24 forecast slots from current time:
            [(remaining_duration_sec, renewable_power_w), ...]
        Cycles past horizon like GAS-MARL §4.2.1.
        """
        base_slot = self._slot_index(sim_time_sec)
        dt = self.config.slot_duration_sec
        elapsed_in_slot = sim_time_sec % dt
        result = []
        for i in range(self.config.forecast_horizon_slots):
            slot = base_slot + i
            if slot >= len(self._total_kw):
                slot = slot % len(self._total_kw)
            remaining = dt - elapsed_in_slot if i == 0 else dt
            result.append((remaining, self._total_kw[slot] * 1000.0))
        return result

    def compute_utilization(
        self,
        snapshots: list[tuple[float, float]],   # (sim_time_sec, cluster_power_w)
    ) -> float:
        """
        ReUtil = ∫ min(P_green, P_cluster) dt / ∫ P_cluster dt
        snapshots: list of (time, cluster_total_power_w)
        """
        if len(snapshots) < 2:
            return 0.0
        green_consumed = 0.0
        total_consumed = 0.0
        for i in range(1, len(snapshots)):
            dt = snapshots[i][0] - snapshots[i - 1][0]
            avg_cluster = (snapshots[i][1] + snapshots[i - 1][1]) / 2.0
            t_mid = (snapshots[i][0] + snapshots[i - 1][0]) / 2.0
            green_avail = self.available_power_watts(t_mid)
            green_used = min(green_avail, avg_cluster)
            green_consumed += green_used * dt
            total_consumed += avg_cluster * dt
        if total_consumed <= 0:
            return 0.0
        return green_consumed / total_consumed

    def idle_power_watts(self, total_gpus: int) -> float:
        """Approximate cluster idle power (all GPUs at idle)."""
        # ~25W idle per GPU on average across heterogeneous types
        return total_gpus * 25.0

    def job_power_watts(
        self,
        num_gpus: int,
        utilization: float = 1.0,
        tdp_per_gpu: float = 300.0,
        idle_per_gpu: float = 25.0,
    ) -> float:
        """
        Power consumed by a running job using GOGH model:
        P(u) = P_idle + (P_tdp - P_idle) * (0.3u + 0.7u²)
        """
        u = max(0.0, min(1.0, utilization))
        p_per_gpu = idle_per_gpu + (tdp_per_gpu - idle_per_gpu) * (0.3 * u + 0.7 * u * u)
        return p_per_gpu * num_gpus
