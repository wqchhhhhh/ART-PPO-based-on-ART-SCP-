"""
Configuration module for Air-breathing Hypersonic Vehicle (AHV) Trajectory Optimization.
Contains planetary constants, vehicle geometry, aerodynamics, propulsion,
constraints, boundary conditions, and no-fly zone definitions.
"""

import numpy as np
#===============================================================================
#===============================================================================

# ==============================================================================
# 1. Planetary Constants (WGS-84 / Spherical Earth approximation)
# ==============================================================================
R_E = 6371000.0           # Earth mean radius [m]地球半径
MU_E = 3.986004418e14     # Earth gravitational parameter [m^3/s^2]地球引力常数
OMEGA_E = 7.292115e-5     # Earth rotation angular velocity [rad/s]地球自转角速度
G0 = 9.80665              # Standard gravity acceleration [m/s^2]标准重力加速度

# ==============================================================================
# 2. Vehicle Parameters (Generic Waverider / Air-breathing Hypersonic Cruiser)
# ==============================================================================
S_REF = 60.0              # Aerodynamic reference area [m^2]气动参考面积
M_INIT = 20000.0          # Initial mass [kg]初始质量
M_DRY = 12000.0           # Dry mass (empty mass) [kg]空载质量，不含消耗燃油
NOSE_RADIUS = 0.05        # Nose bluntness radius for heat flux [m]用于热流估计的头部钝化半径
K_HEAT = 1.7415e-4        # Detra-Kemp-Riddell / Sutton-Graves heating constant 当前热流近似公式的系数

# Aerodynamic coefficients: CL = CL0 + CL_alpha * alpha
#                           CD = CD0 + K_ind * CL^2
CL0 = -0.010              # Zero-alpha lift coefficient 0攻角升力系数
CL_ALPHA = 1.55           # Lift curve slope [1/rad]
#=================================================
#C_L=C_{L0}+C_{L\alpha}alpha
#==========================================
CD0 = 0.016               # Zero-lift parasitic drag coefficient
K_IND = 0.75              # Induced drag factor
#======================================
#C_D=C_{D0}+K_{\mathrm{IND}}C_L^2 二次升力相关阻力项
#======================================
# Propulsion parameters (Scramjet)发动机参数
# T_max = CT_MAX * q * S_REF, T = eta * T_max
CT_MAX = 0.040            # Maximum thrust coefficient 最大推力系数
ISP = 1500.0              # Specific impulse at cruise condition [s] 当前巡航工况下采用的比冲

# ==============================================================================
# 3. Mission Parameters & Boundary Conditions
# ==============================================================================
# Coordinates: Longitude theta, Latitude phi
# Total great-circle range ~ 4000 km

#起点的经纬度
LON_0_DEG = 105.0         # Initial longitude [deg]
LAT_0_DEG = 20.0          # Initial latitude [deg]

#终点的经纬度
LON_F_DEG = 143.0         # Final longitude [deg]
LAT_F_DEG = 32.0          # Final latitude [deg]

LON_0 = np.radians(LON_0_DEG)
LAT_0 = np.radians(LAT_0_DEG)
LON_F = np.radians(LON_F_DEG)
LAT_F = np.radians(LAT_F_DEG)

# Cruise altitude and speed (Ma 7, 30 km +- 5 km)
ALT_CRUISE = 30000.0      # Nominal cruise altitude [m]生成初始猜测时使用的名义巡航高度
ALT_0 = 30000.0           # Initial altitude [m]初始高度
ALT_F = 30000.0           # Final altitude [m]	最终高度

VEL_CRUISE = 2110.0       # Cruise velocity [m/s] (~Ma 7 at 30km)	初始猜测使用的名义巡航速度
VEL_0 = 2110.0            # Initial velocity [m/s]	初始速度
VEL_F = 2110.0            # Final velocity [m/s] 最终速度

GAMMA_0 = 0.0             # Initial flight path angle [rad]初始航迹倾角
GAMMA_F = 0.0             # Final flight path angle [rad]最终航迹倾角

# Time horizon estimation (~4000 km at ~2100 m/s -> ~1900 s)
#飞行时间的范围
TF_MIN = 1600.0           # Minimum flight time [s]
TF_MAX = 2400.0           # Maximum flight time [s]
TF_GUESS = 1920.0         # Initial guess of flight time [s]

# ==============================================================================
# 4. State & Control Boundaries
#状态和控制边界
# ==============================================================================
# Altitude bounds: 30 km +- 5 km -> [25 km, 35 km]高度范围
ALT_MIN = 25000.0         # [m]
ALT_MAX = 35000.0         # [m]

#对用的地心距范围
R_MIN = R_E + ALT_MIN     # [m]
R_MAX = R_E + ALT_MAX     # [m]

#速度范围
VEL_MIN = 1850.0          # [m/s]
VEL_MAX = 2350.0          # [m/s]

#航迹倾角范围
GAMMA_MIN = np.radians(-4.0)   # [rad]
GAMMA_MAX = np.radians(4.0)    # [rad]

#航向角范围
PSI_MIN = np.radians(-180.0)   # [rad] Heading angle
PSI_MAX = np.radians(180.0)    # [rad]

#质量范围
MASS_MIN = M_DRY               # [kg]
MASS_MAX = M_INIT              # [kg]

# Control bounds
#攻角范围
ALPHA_MIN = np.radians(-1.5)   # Angle of attack min [rad] (~ -1.5 deg)
ALPHA_MAX = np.radians(8.0)    # Angle of attack max [rad] (~ +8.0 deg)

#倾侧角允许范围
SIGMA_MIN = np.radians(-60.0)  # Bank angle min [rad] (-60 deg)
SIGMA_MAX = np.radians(60.0)   # Bank angle max [rad] (+60 deg)

#油门比例范围
ETA_MIN = 0.05                 # Minimum throttle ratio (idle combustion)
ETA_MAX = 1.0                  # Maximum throttle ratio

# ==============================================================================
# 5. Path Constraints 动压约束、过载约束、最大驻点热流
# ==============================================================================
Q_MIN = 18.0e3            # Min dynamic pressure [Pa] (inlet starting)
Q_MAX = 80.0e3            # Max dynamic pressure [Pa] (structural limit)
N_MAX = 2.5               # Max normal load factor [-]
Q_HEAT_MAX = 1000.0e3     # Max stagnation heat flux [W/m^2] (1000 kW/m^2)

# ==============================================================================
# 6. No-Fly Zones (NFZs)
# Multiple circular exclusion zones along the flight corridor禁飞区参数
# ==============================================================================
NO_FLY_ZONES = [
    {
        "id": 1,
        "name": "NFZ-A",
        "lon_deg": 116.0,
        "lat_deg": 23.5,
        "radius_km": 180.0,
    },
    {
        "id": 2,
        "name": "NFZ-B",
        "lon_deg": 126.0,
        "lat_deg": 27.0,
        "radius_km": 200.0,
    },
    {
        "id": 3,
        "name": "NFZ-C",
        "lon_deg": 134.0,
        "lat_deg": 29.5,
        "radius_km": 180.0,
    },
]

# Precompute NFZ angles in radians and cosines of angular radii
for nfz in NO_FLY_ZONES:
    nfz["lon_rad"] = np.radians(nfz["lon_deg"])
    nfz["lat_rad"] = np.radians(nfz["lat_deg"])
    nfz["angular_radius_rad"] = (nfz["radius_km"] * 1000.0) / R_E
    # Threshold cosine: cos(dist) <= cos(radius)
    nfz["cos_radius"] = np.cos(nfz["angular_radius_rad"])

# ==============================================================================
# 7. Collocation & Scaling Constants
# ==============================================================================
N_NODES = 45              # LGL initial-solution polynomial degreeLGL 初始解的多项式阶数设置

# Continuous refinement: piecewise-linear controls and RK4 multiple shooting.
#
SHOOTING_INTERVALS = 120    #	多重打靶控制区间数
RK4_MAX_STEP = 2.0        # Maximum shooting integrator step at TF_MAX [s]多重打靶积分精度设置，注释以 TF_MAX 为基准
NFZ_BUFFER_M = 100.0      # Numerical clearance reserve in the NLP [m]
ALT_BUFFER_M = 2.0        # Interior state-constraint reserve [m]
VEL_BUFFER_MPS = 0.05
PATH_BUFFER_FRACTION = 1e-3
VALIDATION_MAX_STEP = 1.0 # Independent DOP853 replay/check spacing [s]
CONTROL_REG_WEIGHT = 1e-6 # Integral of squared normalized control rate
MODEL_VERSION = "ahv-us76-heading-v2"

# Scaling factors for well-conditioned NLP 进行缩放，原始变量量级差别很大
SCALE_R = 10000.0         # r_scaled = (r - R_E) / SCALE_R  (~2.5 to 3.5)
SCALE_V = 1000.0          # V_scaled = V / SCALE_V          (~1.8 to 2.4)
SCALE_GAMMA = 0.1         # gamma_scaled = gamma / 0.1      (~ -0.7 to 0.7)
SCALE_MASS = 10000.0      # mass_scaled = mass / 10000.0    (~1.2 to 2.0)
SCALE_TF = 1000.0         # tf_scaled = tf / 1000.0         (~1.6 to 2.4)
SCALE_ALPHA = 0.1         # alpha_scaled = alpha / 0.1      (~ -0.2 to 0.8)
SCALE_SIGMA = 1.0         # sigma_scaled = sigma / 1.0      (~ -1.0 to 1.0)
SCALE_ETA = 1.0           # eta_scaled = eta / 1.0          (~ 0.1 to 1.0)
