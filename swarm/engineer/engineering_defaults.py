"""
Engineering Defaults Library — v2 (50 domains)

Static lookup tables of well-known engineering defaults and physical bounds.
Used by engineer_mode.py to fill missing design parameters with documented,
citable assumptions.

Domain list:
    rocket, motor, structure, thermal, power, fluid,          # original 6
    ballistics, orbital_mechanics, aerodynamics, aircraft,
    rotorcraft, automotive, robotics, controls, electronics,
    rf, optics, acoustics, chemical, combustion,
    gas_turbine, steam_turbine, refrigeration, heat_exchanger,
    cryogenics, nuclear, solar_pv, wind_turbine, hvac, civil,
    geotechnical, marine, materials, composites, manufacturing,
    welding, gearbox, bearing, vibration, fatigue,
    impact, pneumatics, hydraulics, vacuum, semiconductor,
    battery_chem, nuclear_power, geophysics, biomedical, mining
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


# ─── ENGINEERING DEFAULTS ────────────────────────────────────────────────────
# Structure: domain → { key → (value, unit, rationale) }

ENGINEERING_DEFAULTS: Dict[str, Dict[str, Tuple]] = {

    # ── 1. ROCKET / LAUNCH VEHICLE ─────────────────────────────────────────
    "rocket": {
        "Isp_LOX_RP1_sl":             (311.0,  "s",      "Merlin 1D sea-level Isp"),
        "Isp_LOX_RP1_vac":            (358.0,  "s",      "Merlin 1D vacuum Isp"),
        "Isp_LOX_LH2_vac":            (450.0,  "s",      "RL-10 class vacuum Isp"),
        "Isp_LOX_LH2_sl":             (380.0,  "s",      "SSME sea-level Isp"),
        "Isp_N2O4_UDMH_vac":          (315.0,  "s",      "Proton/Ariane4 storable propellants"),
        "Isp_LOX_CH4_vac":            (380.0,  "s",      "Raptor methane engine, vacuum"),
        "Isp_LOX_CH4_sl":             (330.0,  "s",      "Raptor sea-level Isp"),
        "eps_first_stage":            (0.06,   "",       "Structural fraction, expendable 1st stage"),
        "eps_second_stage":           (0.10,   "",       "Structural fraction, upper stage"),
        "eps_reusable_first":         (0.12,   "",       "Structural fraction, reusable 1st stage"),
        "dV_gravity_loss":            (1200.0, "m/s",    "Gravity drag loss, typical ascent"),
        "dV_drag_loss":               (200.0,  "m/s",    "Atmospheric drag loss, ascent"),
        "dV_LEO_ideal":               (9100.0, "m/s",    "Ideal delta-v to LEO 200 km"),
        "dV_LEO_with_losses":         (9500.0, "m/s",    "Total delta-v to LEO with losses"),
        "dV_GTO":                     (12000.0,"m/s",    "Delta-v to GTO, equatorial launch"),
        "TW_liftoff":                 (1.4,    "",       "Thrust-to-weight at liftoff (typical)"),
        "LOX_RP1_mixture_ratio":      (2.56,   "",       "LOX/RP-1 oxidizer-to-fuel ratio"),
        "LOX_LH2_mixture_ratio":      (6.0,    "",       "LOX/LH2 oxidizer-to-fuel ratio"),
        "LOX_CH4_mixture_ratio":      (3.55,   "",       "LOX/methane oxidizer-to-fuel (Raptor)"),
    },

    # ── 2. ELECTRIC MOTOR / DRIVE ──────────────────────────────────────────
    "motor": {
        "efficiency_BLDC":            (0.92,   "",       "Brushless DC motor efficiency"),
        "efficiency_PMSM":            (0.95,   "",       "Permanent magnet synchronous motor"),
        "efficiency_induction":       (0.90,   "",       "Induction motor efficiency"),
        "thermal_resistance_shaft":   (0.5,    "°C/W",   "Shaft-to-case thermal resistance"),
        "switching_freq_kHz":         (20.0,   "kHz",    "PWM inverter switching frequency"),
        "voltage_DC_bus":             (48.0,   "V",      "48V DC bus, robotics/EV drives"),
        "power_factor":               (0.9,    "",       "Power factor, AC motor drive"),
        "torque_density":             (5.0,    "N·m/kg", "Typical BLDC torque density"),
        "rpm_per_volt_kv":            (100.0,  "rpm/V",  "Typical outrunner KV rating"),
    },

    # ── 3. STRUCTURE / MECHANICS ───────────────────────────────────────────
    "structure": {
        "yield_Al6061":               (276.0,  "MPa",    "6061-T6 aluminium yield strength"),
        "yield_Ti6Al4V":              (880.0,  "MPa",    "Ti-6Al-4V yield strength"),
        "yield_A36":                  (250.0,  "MPa",    "A36 structural steel yield"),
        "yield_304SS":                (215.0,  "MPa",    "304 stainless steel yield"),
        "E_steel":                    (200e3,  "MPa",    "Young's modulus steel"),
        "E_Al":                       (69e3,   "MPa",    "Young's modulus aluminium"),
        "E_CFRP":                     (70e3,   "MPa",    "Young's modulus CFRP laminate"),
        "safety_factor_static":       (1.5,    "",       "Static safety factor, aerospace"),
        "safety_factor_fatigue":      (4.0,    "",       "Fatigue safety factor, aerospace"),
        "density_Al":                 (2700.0, "kg/m³",  "Aluminium density"),
        "density_steel":              (7850.0, "kg/m³",  "Steel density"),
        "density_CFRP":               (1600.0, "kg/m³",  "CFRP density"),
        "density_Ti":                 (4430.0, "kg/m³",  "Titanium density"),
    },

    # ── 4. THERMAL ─────────────────────────────────────────────────────────
    "thermal": {
        "h_natural_air":              (10.0,   "W/(m²·K)","Natural convection in air"),
        "h_forced_air":               (50.0,   "W/(m²·K)","Forced convection in air"),
        "h_water":                    (5000.0, "W/(m²·K)","Liquid water convection"),
        "k_Al":                       (205.0,  "W/(m·K)", "Thermal conductivity aluminium"),
        "k_steel":                    (50.0,   "W/(m·K)", "Thermal conductivity steel"),
        "k_insulation":               (0.04,   "W/(m·K)", "Mineral wool / foam insulation"),
        "emissivity_painted":         (0.9,    "",        "Emissivity, painted metal"),
        "emissivity_polished_Al":     (0.05,   "",        "Emissivity, polished aluminium"),
        "T_ambient":                  (293.15, "K",       "Standard ambient (20°C)"),
        "stefan_boltzmann":           (5.67e-8,"W/(m²·K⁴)","Stefan-Boltzmann constant"),
    },

    # ── 5. ELECTRICAL POWER ────────────────────────────────────────────────
    "power": {
        "energy_density_LiPo":        (200.0,  "Wh/kg",  "LiPo battery energy density"),
        "energy_density_LiFePO4":     (130.0,  "Wh/kg",  "LiFePO4 battery energy density"),
        "energy_density_NMC":         (240.0,  "Wh/kg",  "NMC lithium battery energy density"),
        "efficiency_inverter":        (0.97,   "",       "Modern IGBT/SiC inverter efficiency"),
        "efficiency_DC_DC":           (0.95,   "",       "DC-DC converter efficiency"),
        "solar_irradiance_AM0":       (1350.0, "W/m²",   "Solar irradiance, LEO (AM0)"),
        "efficiency_solar_cell":      (0.30,   "",       "High-efficiency space solar cell"),
        "cable_voltage_drop_max":     (0.02,   "",       "Max allowed cable voltage drop fraction"),
    },

    # ── 6. FLUID MECHANICS ─────────────────────────────────────────────────
    "fluid": {
        "pipe_velocity_water":        (2.0,    "m/s",    "Typical water pipe velocity"),
        "pipe_velocity_gas":          (20.0,   "m/s",    "Typical gas pipe velocity"),
        "friction_factor":            (0.02,   "",       "Darcy friction factor, turbulent"),
        "density_water":              (1000.0, "kg/m³",  "Water at 20°C"),
        "density_air_sl":             (1.225,  "kg/m³",  "Air at sea level ISA"),
        "viscosity_water":            (1e-3,   "Pa·s",   "Dynamic viscosity water 20°C"),
        "viscosity_air":              (1.81e-5,"Pa·s",   "Dynamic viscosity air 20°C"),
        "pump_efficiency":            (0.75,   "",       "Centrifugal pump efficiency"),
        "bulk_modulus_water":         (2.2e9,  "Pa",     "Bulk modulus of water"),
    },

    # ── 7. BALLISTICS / PROJECTILE MOTION ─────────────────────────────────
    "ballistics": {
        "g0":                         (9.80665,"m/s²",   "Standard gravitational acceleration"),
        # FRC robot typical values
        "frc_robot_max_speed":        (5.0,    "m/s",    "FRC drivetrain max speed (typical 2024)"),
        "frc_field_length":           (16.46,  "m",      "FRC field length 2024/2025/2026 (54 ft)"),
        "frc_field_width":            (8.23,   "m",      "FRC field width (27 ft)"),
        "frc_ball_mass_kg":           (0.27,   "kg",     "FRC fuel ball approximate mass ~9.5 oz"),
        "frc_shooter_exit_vel":       (15.0,   "m/s",    "Typical FRC flywheel shooter exit speed"),
        # General ballistics
        "Cd_sphere":                  (0.47,   "",       "Drag coefficient, smooth sphere"),
        "Cd_bullet":                  (0.30,   "",       "Drag coefficient, bullet-shaped projectile"),
        "max_range_angle_deg":        (45.0,   "deg",    "Angle for max range, no drag"),
        # Lead compensation
        "lead_angle_formula":         (0.0,    "",       "θ_lead = atan(v_robot_perp / v_projectile)"),
        "time_of_flight_formula":     (0.0,    "",       "t = distance / v_horizontal (flat approx)"),
        "air_density_sl":             (1.225,  "kg/m³",  "Air density at sea level"),
        "spin_gyroscopic_factor":     (1.0,    "",       "Gyroscopic stability factor (1=stable)"),
    },

    # ── 8. ORBITAL MECHANICS ───────────────────────────────────────────────
    "orbital_mechanics": {
        "mu_Earth":                   (3.986004418e14,"m³/s²","Earth gravitational parameter GM"),
        "mu_Sun":                     (1.32712440018e20,"m³/s²","Sun gravitational parameter GM"),
        "mu_Moon":                    (4.9048695e12,"m³/s²","Moon gravitational parameter GM"),
        "R_Earth":                    (6.3781e6,"m",     "Earth mean radius"),
        "R_Moon":                     (1.7374e6,"m",     "Moon mean radius"),
        "v_LEO":                      (7784.0, "m/s",    "LEO circular orbital speed ~400 km"),
        "v_escape_Earth":             (11186.0,"m/s",    "Earth escape velocity from surface"),
        "alt_LEO_typical":            (400e3,  "m",      "Typical LEO altitude (ISS ~410 km)"),
        "alt_GEO":                    (35786e3,"m",      "Geostationary orbit altitude"),
        "T_LEO":                      (5560.0, "s",      "LEO orbital period ~92 min"),
        "T_GEO":                      (86164.0,"s",      "GEO orbital period (sidereal day)"),
        "dV_Hohmann_LEO_GEO":         (3900.0, "m/s",    "Total delta-v Hohmann LEO→GEO"),
        "J2_Earth":                   (1.08263e-3,"",    "Earth J2 oblateness coefficient"),
    },

    # ── 9. AERODYNAMICS ────────────────────────────────────────────────────
    "aerodynamics": {
        "air_density_sl":             (1.225,  "kg/m³",  "ISA sea-level air density"),
        "air_density_10km":           (0.4135, "kg/m³",  "ISA air density at 10 km"),
        "speed_of_sound_sl":          (340.3,  "m/s",    "Speed of sound at sea level, ISA"),
        "speed_of_sound_10km":        (299.5,  "m/s",    "Speed of sound at 10 km, ISA"),
        "dynamic_viscosity_air_sl":   (1.789e-5,"Pa·s",  "Air dynamic viscosity at sea level"),
        "Cl_max_clean":               (1.5,    "",       "Max lift coefficient, clean wing"),
        "Cl_max_flaps":               (2.5,    "",       "Max lift coefficient, flaps extended"),
        "Cd0_clean":                  (0.025,  "",       "Zero-lift drag coefficient, clean aircraft"),
        "oswald_factor":              (0.80,   "",       "Oswald span efficiency factor (typical)"),
        "AR_typical_subsonic":        (8.0,    "",       "Aspect ratio, typical subsonic transport"),
        "AR_typical_fighter":         (3.5,    "",       "Aspect ratio, typical fighter jet"),
        "Mach_critical":              (0.72,   "",       "Critical Mach for typical swept wing"),
        "stall_speed_margin":         (1.3,    "",       "V_stall safety margin factor (CS-25)"),
    },

    # ── 10. FIXED-WING AIRCRAFT ────────────────────────────────────────────
    "aircraft": {
        "wing_loading_GA":            (650.0,  "N/m²",   "Wing loading, general aviation"),
        "wing_loading_airliner":      (6000.0, "N/m²",   "Wing loading, large airliner"),
        "wing_loading_fighter":       (3500.0, "N/m²",   "Wing loading, fighter jet"),
        "thrust_to_weight_airliner":  (0.30,   "",       "T/W ratio at takeoff, airliner"),
        "thrust_to_weight_fighter":   (1.1,    "",       "T/W ratio, fighter (supersonic capable)"),
        "fuel_fraction_cruise":       (0.45,   "",       "Fuel mass fraction, medium-haul airliner"),
        "SFC_turbofan_cruise":        (1.6e-5, "kg/(N·s)","Specific fuel consumption, turbofan cruise"),
        "L_D_max_airliner":           (18.0,   "",       "Max L/D ratio, modern airliner"),
        "L_D_max_GA":                 (12.0,   "",       "Max L/D ratio, piston GA aircraft"),
        "cruise_Mach_airliner":       (0.82,   "",       "Cruise Mach number, narrow-body airliner"),
        "takeoff_field_length":       (2000.0, "m",      "Typical sea-level TOFL, airliner"),
    },

    # ── 11. ROTORCRAFT / MULTIROTOR ────────────────────────────────────────
    "rotorcraft": {
        "disk_loading_helicopter":    (400.0,  "N/m²",   "Disk loading, medium helicopter"),
        "disk_loading_drone":         (80.0,   "N/m²",   "Disk loading, consumer multirotor"),
        "figure_of_merit":            (0.75,   "",       "Rotor figure of merit (hover efficiency)"),
        "blade_Cl_operating":         (0.6,    "",       "Typical rotor blade operating Cl"),
        "solidity_helicopter":        (0.07,   "",       "Rotor solidity, helicopter main rotor"),
        "tip_speed_helicopter":       (215.0,  "m/s",    "Blade tip speed, helicopter (Mach limit)"),
        "tip_speed_drone":            (80.0,   "m/s",    "Blade tip speed, consumer drone"),
        "hover_power_per_kg":         (150.0,  "W/kg",   "Hover power per GTOW, small multirotor"),
        "flight_time_typical_drone":  (1200.0, "s",      "Typical consumer drone flight time ~20 min"),
        "payload_fraction_drone":     (0.20,   "",       "Payload fraction, commercial delivery drone"),
    },

    # ── 12. AUTOMOTIVE / VEHICLES ──────────────────────────────────────────
    "automotive": {
        "Cd_sedan":                   (0.30,   "",       "Drag coefficient, modern sedan"),
        "Cd_SUV":                     (0.38,   "",       "Drag coefficient, SUV"),
        "Cd_sports_car":              (0.25,   "",       "Drag coefficient, sports car"),
        "frontal_area_sedan":         (2.2,    "m²",     "Frontal area, typical sedan"),
        "rolling_resistance_coeff":   (0.015,  "",       "Rolling resistance coefficient, asphalt"),
        "tire_friction_dry":          (0.8,    "",       "Tire-road friction coefficient, dry"),
        "tire_friction_wet":          (0.5,    "",       "Tire-road friction coefficient, wet"),
        "engine_efficiency_ICE":      (0.35,   "",       "Typical ICE thermal efficiency"),
        "drivetrain_efficiency":      (0.90,   "",       "Drivetrain mechanical efficiency"),
        "fuel_energy_density_petrol": (8.76e7, "J/kg",   "Energy density, petrol (gasoline)"),
        "fuel_energy_density_diesel": (9.8e7,  "J/kg",   "Energy density, diesel fuel"),
        "battery_EV_specific_energy": (250.0,  "Wh/kg",  "EV battery pack energy density (pack level)"),
        "regen_braking_efficiency":   (0.65,   "",       "Regenerative braking round-trip efficiency"),
        "aero_drag_onset_speed":      (50.0,   "km/h",   "Speed above which aero drag dominates"),
    },

    # ── 13. ROBOTICS ───────────────────────────────────────────────────────
    "robotics": {
        # FRC competition specifics
        "frc_max_robot_mass":         (54.43,  "kg",     "FRC robot mass limit (120 lb) + bumpers"),
        "frc_bumper_mass":            (6.0,    "kg",     "FRC bumper mass estimate"),
        "frc_battery_mass":           (5.9,    "kg",     "FRC MK ES17-12 battery mass"),
        "frc_battery_voltage":        (12.0,   "V",      "FRC nominal battery voltage"),
        "frc_battery_capacity":       (18.0,   "Ah",     "FRC standard battery capacity"),
        "frc_max_current_main":       (120.0,  "A",      "FRC main breaker rating"),
        "frc_wheel_diameter_typical": (0.1524, "m",      "FRC 6-inch wheel diameter"),
        # General robotics
        "servo_torque_typical":       (2.5,    "N·m",    "Typical hobby servo stall torque"),
        "encoder_resolution_typical": (4096,   "counts/rev","Typical quadrature encoder resolution"),
        "loop_rate_fast":             (200.0,  "Hz",     "Fast control loop rate"),
        "loop_rate_normal":           (50.0,   "Hz",     "Normal control loop rate"),
        "localization_accuracy_GPS":  (2.0,    "m",      "GPS position accuracy (open sky)"),
        "localization_accuracy_SLAM": (0.05,   "m",      "SLAM/lidar localization accuracy"),
        "joint_gear_ratio_arm":       (50.0,   "",       "Typical robot arm joint gear reduction"),
    },

    # ── 14. CONTROLS ───────────────────────────────────────────────────────
    "controls": {
        # PID tuning starting points
        "Kp_start":                   (1.0,    "",       "Initial proportional gain guess"),
        "Ki_start":                   (0.0,    "",       "Initial integral gain (start at zero)"),
        "Kd_start":                   (0.1,    "",       "Initial derivative gain guess"),
        "phase_margin_min":           (45.0,   "deg",    "Minimum acceptable phase margin"),
        "gain_margin_min":            (6.0,    "dB",     "Minimum acceptable gain margin"),
        "damping_ratio_typical":      (0.7,    "",       "Damping ratio for good transient response"),
        "settling_time_factor":       (4.0,    "",       "Settling time ≈ 4/damping*omega_n"),
        "bandwidth_rule":             (0.1,    "",       "Control bandwidth ≈ 0.1 × disturbance freq"),
        "overshoot_limit_pct":        (10.0,   "%",      "Max overshoot for 2nd-order system"),
        "sample_rate_rule":           (10.0,   "",       "Sample rate ≥ 10× bandwidth (Nyquist+margin)"),
        "actuator_saturation_margin": (0.8,    "",       "Keep actuator below 80% saturation"),
        "observer_bandwidth_factor":  (5.0,    "",       "Observer bandwidth ≈ 5× controller bandwidth"),
    },

    # ── 15. ELECTRONICS / CIRCUITS ─────────────────────────────────────────
    "electronics": {
        "V_forward_diode_Si":         (0.7,    "V",      "Silicon diode forward voltage"),
        "V_forward_LED":              (2.0,    "V",      "Typical LED forward voltage"),
        "V_CE_sat_BJT":               (0.2,    "V",      "BJT collector-emitter saturation voltage"),
        "V_GS_th_MOSFET":             (3.0,    "V",      "N-channel MOSFET threshold voltage"),
        "R_DS_on_typical":            (0.05,   "Ω",      "MOSFET on-state resistance, power FET"),
        "op_amp_GBW":                 (1e6,    "Hz",     "Op-amp gain-bandwidth product, general"),
        "op_amp_slew_rate":           (1.0,    "V/µs",   "Op-amp slew rate, standard type"),
        "PCB_trace_current_1oz":      (1.0,    "A",      "Max current, 1oz copper 1mm trace"),
        "PCB_trace_current_2oz":      (2.0,    "A",      "Max current, 2oz copper 1mm trace"),
        "decoupling_cap_100MHz":      (100e-9, "F",      "Decoupling capacitor near IC, 100 MHz"),
        "ADC_resolution_12bit":       (4096,   "counts", "12-bit ADC full-scale counts"),
        "I2C_speed_fast":             (400e3,  "Hz",     "I2C fast-mode clock"),
        "SPI_speed_typical":          (10e6,   "Hz",     "SPI clock, typical MCU peripheral"),
        "UART_baud_typical":          (115200, "bps",    "UART baud rate, embedded default"),
    },

    # ── 16. RF / ANTENNA / WIRELESS ────────────────────────────────────────
    "rf": {
        "c_light":                    (3e8,    "m/s",    "Speed of light in vacuum"),
        "free_space_path_loss_factor": (20.0,  "dB",     "FSPL exponent factor (20 log10)"),
        "noise_figure_LNA":           (2.0,    "dB",     "Low-noise amplifier noise figure"),
        "thermal_noise_kTB_1Hz":      (-174.0, "dBm/Hz", "Thermal noise floor, room temperature"),
        "antenna_gain_dipole":        (2.15,   "dBi",    "Half-wave dipole antenna gain"),
        "antenna_gain_patch":         (7.0,    "dBi",    "Microstrip patch antenna gain"),
        "antenna_gain_yagi_6el":      (10.0,   "dBi",    "6-element Yagi antenna gain"),
        "WiFi_2_4GHz_freq":           (2.4e9,  "Hz",     "WiFi 2.4 GHz band center"),
        "WiFi_5GHz_freq":             (5.8e9,  "Hz",     "WiFi 5 GHz band center"),
        "LoRa_sensitivity":           (-148.0, "dBm",    "LoRa SF12 receiver sensitivity"),
        "link_margin_typical":        (10.0,   "dB",     "Minimum acceptable link margin"),
        "coax_loss_RG58_per_m":       (0.15,   "dB/m",   "RG-58 coax loss at 100 MHz"),
        "impedance_coax":             (50.0,   "Ω",      "Standard coax characteristic impedance"),
    },

    # ── 17. OPTICS / PHOTONICS ─────────────────────────────────────────────
    "optics": {
        "n_glass":                    (1.52,   "",       "Refractive index, crown glass"),
        "n_water":                    (1.33,   "",       "Refractive index, water"),
        "n_air":                      (1.0003, "",       "Refractive index, air (STP)"),
        "wavelength_red":             (650e-9, "m",      "Red laser wavelength"),
        "wavelength_green":           (532e-9, "m",      "Green laser wavelength"),
        "wavelength_blue":            (450e-9, "m",      "Blue laser wavelength"),
        "diffraction_limit_factor":   (1.22,   "",       "Rayleigh criterion factor"),
        "camera_pixel_size_typical":  (3.5e-6, "m",      "Typical CMOS image sensor pixel size"),
        "f_number_typical":           (2.8,    "",       "Typical camera f-number"),
        "laser_beam_divergence_rad":  (1e-3,   "rad",    "Typical diode laser divergence"),
        "fiber_core_SMF":             (9e-6,   "m",      "Single-mode fiber core diameter"),
        "fiber_attenuation_1550":     (0.2,    "dB/km",  "SMF28 attenuation at 1550 nm"),
    },

    # ── 18. ACOUSTICS / NOISE ──────────────────────────────────────────────
    "acoustics": {
        "speed_sound_air":            (343.0,  "m/s",    "Speed of sound in air at 20°C"),
        "speed_sound_water":          (1480.0, "m/s",    "Speed of sound in water at 20°C"),
        "density_air_acoustic":       (1.204,  "kg/m³",  "Air density at 20°C"),
        "Zair":                       (413.0,  "Pa·s/m", "Acoustic impedance of air"),
        "p_ref":                      (20e-6,  "Pa",     "Reference pressure for dB SPL"),
        "threshold_hearing":          (0.0,    "dB SPL", "Threshold of hearing (1 kHz)"),
        "pain_threshold":             (130.0,  "dB SPL", "Threshold of pain"),
        "NR_insulation_stud_wall":    (45.0,   "dB",     "Noise reduction, stud+drywall partition"),
        "RT60_typical_office":        (0.5,    "s",      "Reverberation time, typical office"),
        "RT60_concert_hall":          (1.8,    "s",      "Reverberation time, concert hall"),
        "A_weighting_1kHz":           (0.0,    "dB",     "A-weighting at 1 kHz (reference)"),
        "hearing_protection_earmuff": (25.0,   "dB NRR", "Earmuff noise reduction rating"),
    },

    # ── 19. CHEMICAL ENGINEERING ───────────────────────────────────────────
    "chemical": {
        "R_gas":                      (8.31446,"J/(mol·K)","Universal gas constant"),
        "N_avogadro":                 (6.022e23,"mol⁻¹", "Avogadro's number"),
        "reactor_residence_time":     (60.0,   "s",      "Typical CSTR residence time"),
        "conversion_typical":         (0.85,   "",       "Typical single-pass conversion"),
        "selectivity_typical":        (0.90,   "",       "Typical reaction selectivity"),
        "heat_of_vaporization_water": (2.257e6,"J/kg",   "Water enthalpy of vaporization at 100°C"),
        "Cp_water":                   (4186.0, "J/(kg·K)","Specific heat capacity of water"),
        "Cp_air":                     (1005.0, "J/(kg·K)","Specific heat capacity of air"),
        "Antoine_water_A":            (8.07131,"",        "Antoine constant A (water, 1–100°C, mmHg)"),
        "Antoine_water_B":            (1730.63,"",        "Antoine constant B (water)"),
        "Antoine_water_C":            (233.426,"",        "Antoine constant C (water)"),
        "density_ethanol":            (789.0,  "kg/m³",  "Density of ethanol at 20°C"),
    },

    # ── 20. COMBUSTION ─────────────────────────────────────────────────────
    "combustion": {
        "LHV_gasoline":               (44.4e6, "J/kg",   "Lower heating value, gasoline"),
        "LHV_diesel":                 (42.8e6, "J/kg",   "Lower heating value, diesel"),
        "LHV_methane":                (50.0e6, "J/kg",   "Lower heating value, methane"),
        "LHV_hydrogen":               (120.0e6,"J/kg",   "Lower heating value, hydrogen"),
        "LHV_ethanol":                (26.8e6, "J/kg",   "Lower heating value, ethanol"),
        "stoich_AFR_gasoline":        (14.7,   "",       "Stoichiometric air-fuel ratio, gasoline"),
        "stoich_AFR_diesel":          (14.5,   "",       "Stoichiometric air-fuel ratio, diesel"),
        "stoich_AFR_methane":         (17.2,   "",       "Stoichiometric air-fuel ratio, methane"),
        "adiabatic_flame_T_methane":  (2230.0, "K",      "Adiabatic flame temperature, methane/air"),
        "adiabatic_flame_T_gasoline": (2275.0, "K",      "Adiabatic flame temperature, gasoline/air"),
        "combustion_efficiency":      (0.98,   "",       "Combustion efficiency, well-tuned engine"),
        "equivalence_ratio_rich":     (1.1,    "",       "Rich mixture equivalence ratio limit"),
        "equivalence_ratio_lean":     (0.6,    "",       "Lean blowout equivalence ratio limit"),
    },

    # ── 21. GAS TURBINE / JET ENGINE ───────────────────────────────────────
    "gas_turbine": {
        "OPR_modern_turbofan":        (45.0,   "",       "Overall pressure ratio, modern turbofan"),
        "TIT_modern":                 (1900.0, "K",      "Turbine inlet temperature, modern engine"),
        "isentropic_eff_compressor":  (0.88,   "",       "Polytropic efficiency, axial compressor"),
        "isentropic_eff_turbine":     (0.90,   "",       "Polytropic efficiency, turbine"),
        "combustor_pressure_loss":    (0.05,   "",       "Combustor total pressure loss fraction"),
        "bypass_ratio_turbofan":      (12.0,   "",       "Bypass ratio, modern high-BPR turbofan"),
        "SFC_turbofan_TO":            (8.5e-6, "kg/(N·s)","SFC at takeoff, high-BPR turbofan"),
        "gamma_air":                  (1.4,    "",       "Specific heat ratio, air (cold)"),
        "gamma_combustion":           (1.33,   "",       "Specific heat ratio, hot combustion gas"),
        "Cp_air_cold":                (1005.0, "J/(kg·K)","Cp of air (cold section)"),
    },

    # ── 22. STEAM TURBINE / RANKINE CYCLE ─────────────────────────────────
    "steam_turbine": {
        "T_steam_supercritical":      (600.0,  "°C",     "Supercritical steam temperature"),
        "P_steam_supercritical":      (25e6,   "Pa",     "Supercritical steam pressure"),
        "T_condenser":                (45.0,   "°C",     "Condenser temperature, typical"),
        "isentropic_eff_turbine":     (0.88,   "",       "Steam turbine isentropic efficiency"),
        "isentropic_eff_pump":        (0.80,   "",       "Feed pump isentropic efficiency"),
        "Rankine_cycle_eff":          (0.42,   "",       "Net efficiency, modern Rankine cycle"),
        "h_fg_water_100C":            (2257e3, "J/kg",   "Enthalpy of vaporization at 100°C, 1 atm"),
        "boiler_efficiency":          (0.90,   "",       "Boiler thermal efficiency"),
        "plant_capacity_factor":      (0.85,   "",       "Typical power plant capacity factor"),
    },

    # ── 23. REFRIGERATION / HVAC COOLING ──────────────────────────────────
    "refrigeration": {
        "COP_ref_typical":            (3.5,    "",       "COP, typical vapor-compression refrigerator"),
        "COP_AC_typical":             (4.0,    "",       "COP, typical room air conditioner"),
        "COP_heat_pump":              (4.5,    "",       "COP, heat pump (heating mode)"),
        "SEER_min":                   (14.0,   "",       "Minimum SEER, US regulation (2023)"),
        "T_evaporator_comfort":       (7.0,    "°C",     "Evaporator temperature, comfort cooling"),
        "T_condenser_air_cooled":     (45.0,   "°C",     "Condenser temperature, air-cooled"),
        "refrigerant_R134a_GWP":      (1430.0, "",       "GWP of R-134a refrigerant"),
        "refrigerant_R290_GWP":       (3.0,    "",       "GWP of R-290 (propane) refrigerant"),
        "compressor_efficiency":      (0.80,   "",       "Reciprocating compressor volumetric eff"),
        "dT_superheat":               (5.0,    "K",      "Evaporator outlet superheat"),
        "dT_subcooling":              (5.0,    "K",      "Condenser outlet subcooling"),
    },

    # ── 24. HEAT EXCHANGER ─────────────────────────────────────────────────
    "heat_exchanger": {
        "effectiveness_typical":      (0.80,   "",       "Typical HX effectiveness"),
        "U_shell_tube_liquid":        (1000.0, "W/(m²·K)","Overall HTC, shell-and-tube, liq/liq"),
        "U_shell_tube_condensing":    (2000.0, "W/(m²·K)","Overall HTC, condensing steam"),
        "U_plate_HX":                 (3000.0, "W/(m²·K)","Overall HTC, gasketed plate HX"),
        "fouling_factor_water":       (0.0002, "m²·K/W", "Fouling factor, cooling water"),
        "fouling_factor_steam":       (0.0001, "m²·K/W", "Fouling factor, steam"),
        "LMTD_correction_F":          (0.9,    "",       "LMTD correction factor F, cross-flow"),
        "pressure_drop_shell":        (50000.0,"Pa",     "Shell-side pressure drop, typical"),
        "NTU_max_compact":            (5.0,    "",       "Max NTU, compact HX (diminishing returns)"),
        "fin_efficiency":             (0.85,   "",       "Fin efficiency, aluminium finned HX"),
    },

    # ── 25. CRYOGENICS ─────────────────────────────────────────────────────
    "cryogenics": {
        "T_LN2":                      (77.0,   "K",      "Liquid nitrogen boiling point, 1 atm"),
        "T_LH2":                      (20.3,   "K",      "Liquid hydrogen boiling point, 1 atm"),
        "T_LHe":                      (4.2,    "K",      "Liquid helium boiling point, 1 atm"),
        "T_LOX":                      (90.2,   "K",      "Liquid oxygen boiling point, 1 atm"),
        "h_vap_LN2":                  (197e3,  "J/kg",   "Enthalpy of vaporization, LN2"),
        "h_vap_LH2":                  (446e3,  "J/kg",   "Enthalpy of vaporization, LH2"),
        "h_vap_LOX":                  (213e3,  "J/kg",   "Enthalpy of vaporization, LOX"),
        "density_LN2":                (808.0,  "kg/m³",  "Liquid nitrogen density"),
        "density_LH2":                (70.8,   "kg/m³",  "Liquid hydrogen density"),
        "density_LOX":                (1141.0, "kg/m³",  "Liquid oxygen density"),
        "boiloff_rate_LH2":           (0.003,  "1/day",  "LH2 boiloff, well-insulated tank (~0.3%/day)"),
        "MLI_layers":                 (30.0,   "",       "Multi-layer insulation, typical layer count"),
    },

    # ── 26. NUCLEAR (FISSION / SHIELDING) ─────────────────────────────────
    "nuclear": {
        "u235_fission_energy":        (3.2e-11,"J/fission","Energy per U-235 fission"),
        "u235_enrichment_LEU":        (0.035,  "",       "Low-enriched uranium enrichment fraction"),
        "burnup_typical":             (45e9,   "J/kg",   "Burnup, LWR fuel (45 GWd/tU)"),
        "neutron_flux_PWR":           (3e18,   "n/(m²·s)","Typical PWR thermal neutron flux"),
        "half_life_Cs137":            (9.5e8,  "s",      "Cs-137 half-life ~30 years"),
        "half_life_I131":             (6.9e5,  "s",      "I-131 half-life ~8 days"),
        "dose_limit_occupational":    (0.02,   "Sv/year","Occupational dose limit (ICRP 103)"),
        "dose_background":            (0.003,  "Sv/year","Typical background radiation dose"),
        "Pb_HVL_Co60":                (12e-3,  "m",      "Lead half-value layer for Co-60 gamma"),
        "concrete_HVL_Co60":          (60e-3,  "m",      "Concrete HVL for Co-60 gamma"),
    },

    # ── 27. SOLAR PV ───────────────────────────────────────────────────────
    "solar_pv": {
        "G_STC":                      (1000.0, "W/m²",   "Standard test conditions irradiance"),
        "T_STC":                      (25.0,   "°C",     "Standard test conditions cell temperature"),
        "efficiency_mono_Si":         (0.22,   "",       "Monocrystalline silicon panel efficiency"),
        "efficiency_poly_Si":         (0.17,   "",       "Polycrystalline silicon panel efficiency"),
        "efficiency_CdTe":            (0.18,   "",       "CdTe thin-film panel efficiency"),
        "temperature_coeff":          (-0.004, "1/°C",   "Power temperature coefficient, mono-Si"),
        "NOCT":                       (45.0,   "°C",     "Nominal operating cell temperature"),
        "PR_system":                  (0.80,   "",       "Performance ratio, PV system (losses)"),
        "inverter_efficiency":        (0.97,   "",       "String/central inverter efficiency"),
        "PSH_average_global":         (4.5,    "h/day",  "Peak sun hours, mid-latitude average"),
        "panel_degradation_per_year": (0.005,  "1/year", "Annual degradation rate, silicon panel"),
        "BOS_cost_fraction":          (0.40,   "",       "Balance-of-system cost fraction"),
    },

    # ── 28. WIND TURBINE ───────────────────────────────────────────────────
    "wind_turbine": {
        "Betz_limit":                 (0.593,  "",       "Betz limit, max theoretical power coefficient"),
        "Cp_typical":                 (0.45,   "",       "Power coefficient, modern 3-blade HAWT"),
        "cut_in_speed":               (3.0,    "m/s",    "Cut-in wind speed, typical utility turbine"),
        "rated_speed":                (12.0,   "m/s",    "Rated wind speed, utility-scale turbine"),
        "cut_out_speed":              (25.0,   "m/s",    "Cut-out wind speed"),
        "tip_speed_ratio":            (7.0,    "",       "Optimal tip speed ratio, 3-blade HAWT"),
        "capacity_factor_onshore":    (0.35,   "",       "Capacity factor, onshore wind"),
        "capacity_factor_offshore":   (0.45,   "",       "Capacity factor, offshore wind"),
        "hub_height_utility":         (100.0,  "m",      "Hub height, utility-scale turbine"),
        "rotor_diameter_utility":     (150.0,  "m",      "Rotor diameter, typical utility turbine"),
        "specific_power":             (300.0,  "W/m²",   "Rotor specific power, modern turbine"),
    },

    # ── 29. HVAC (BUILDING) ────────────────────────────────────────────────
    "hvac": {
        "fresh_air_per_person":       (10.0,   "L/s",    "Min fresh air supply per person (ASHRAE 62.1)"),
        "ACH_office":                 (6.0,    "1/h",    "Air changes per hour, office"),
        "ACH_clean_room_ISO7":        (60.0,   "1/h",    "Air changes, ISO 7 clean room"),
        "U_double_glazing":           (1.4,    "W/(m²·K)","U-value, double-glazed window"),
        "U_wall_insulated":           (0.3,    "W/(m²·K)","U-value, insulated wall"),
        "U_roof_insulated":           (0.2,    "W/(m²·K)","U-value, insulated roof"),
        "internal_gain_office":       (12.0,   "W/m²",   "Internal heat gain, office (equipment+people)"),
        "solar_gain_factor_SHGC":     (0.4,    "",       "Solar heat gain coefficient, window"),
        "duct_velocity_supply":       (7.5,    "m/s",    "Supply duct air velocity (noise limit)"),
        "duct_pressure_drop":         (1.0,    "Pa/m",   "Duct pressure drop per meter"),
        "fan_efficiency":             (0.70,   "",       "Centrifugal fan total efficiency"),
        "heating_design_delta_T":     (20.0,   "K",      "Design indoor-outdoor temp diff, heating"),
    },

    # ── 30. CIVIL / STRUCTURAL ─────────────────────────────────────────────
    "civil": {
        "concrete_fc":                (30.0,   "MPa",    "Characteristic compressive strength, C30"),
        "concrete_density":           (2400.0, "kg/m³",  "Reinforced concrete density"),
        "rebar_fy":                   (500.0,  "MPa",    "Rebar yield strength, B500B"),
        "E_concrete":                 (30e3,   "MPa",    "Young's modulus, C30 concrete"),
        "safety_factor_concrete":     (1.5,    "",       "Eurocode partial factor, concrete"),
        "safety_factor_steel_civil":  (1.15,   "",       "Eurocode partial factor, steel rebar"),
        "live_load_office":           (3.0,    "kPa",    "Imposed floor load, office (EN 1991)"),
        "live_load_residential":      (2.0,    "kPa",    "Imposed floor load, residential"),
        "wind_pressure_basic":        (0.5,    "kPa",    "Basic wind pressure, moderate exposure"),
        "deflection_limit_L_360":     (0.0028, "",       "Max deflection limit L/360"),
        "seismic_peak_ground_acc":    (0.3,    "g",      "Moderate seismicity PGA (zone 2B)"),
    },

    # ── 31. GEOTECHNICAL ───────────────────────────────────────────────────
    "geotechnical": {
        "bearing_capacity_clay":      (100.0,  "kPa",    "Allowable bearing pressure, stiff clay"),
        "bearing_capacity_sand":      (200.0,  "kPa",    "Allowable bearing pressure, dense sand"),
        "bearing_capacity_rock":      (1000.0, "kPa",    "Allowable bearing pressure, competent rock"),
        "friction_angle_sand":        (32.0,   "deg",    "Internal friction angle, medium-dense sand"),
        "friction_angle_gravel":      (38.0,   "deg",    "Internal friction angle, gravel"),
        "cohesion_clay":              (50.0,   "kPa",    "Cohesion, stiff clay"),
        "density_soil_dry":           (1600.0, "kg/m³",  "Bulk density, dry sand"),
        "density_soil_sat":           (2000.0, "kg/m³",  "Bulk density, saturated soil"),
        "permeability_sand":          (1e-4,   "m/s",    "Hydraulic conductivity, medium sand"),
        "permeability_clay":          (1e-9,   "m/s",    "Hydraulic conductivity, clay"),
        "slope_stability_FS_min":     (1.5,    "",       "Minimum factor of safety, slope stability"),
    },

    # ── 32. MARINE / NAVAL ─────────────────────────────────────────────────
    "marine": {
        "seawater_density":           (1025.0, "kg/m³",  "Seawater density at 15°C"),
        "seawater_viscosity":         (1.07e-3,"Pa·s",   "Seawater dynamic viscosity at 15°C"),
        "Froude_displacement_limit":  (0.4,    "",       "Froude number limit, displacement hull"),
        "Cb_typical_cargo":           (0.82,   "",       "Block coefficient, bulk carrier"),
        "Cb_typical_container":       (0.65,   "",       "Block coefficient, container ship"),
        "propeller_efficiency":       (0.65,   "",       "Open-water propeller efficiency"),
        "hull_efficiency":            (1.02,   "",       "Hull efficiency (wake + thrust deduction)"),
        "admiralty_coefficient":      (1.0,    "",       "Admiralty coefficient (relative, normalized)"),
        "sea_state_3_wave_height":    (1.25,   "m",      "Significant wave height, Sea State 3"),
        "sea_state_5_wave_height":    (3.5,    "m",      "Significant wave height, Sea State 5"),
        "mooring_safety_factor":      (3.0,    "",       "Safety factor, mooring line breaking load"),
    },

    # ── 33. MATERIALS (GENERAL PROPERTIES) ────────────────────────────────
    "materials": {
        "E_modulus_copper":           (110e3,  "MPa",    "Young's modulus, copper"),
        "E_modulus_GFRP":             (25e3,   "MPa",    "Young's modulus, GFRP (E-glass/epoxy)"),
        "yield_strength_copper":      (200.0,  "MPa",    "Yield strength, cold-worked copper"),
        "UTS_Al6061":                 (310.0,  "MPa",    "Ultimate tensile strength, 6061-T6 Al"),
        "UTS_4340_steel":             (1100.0, "MPa",    "UTS, 4340 steel, tempered 315°C"),
        "hardness_Al6061_HV":         (107.0,  "HV",     "Vickers hardness, 6061-T6 Al"),
        "fracture_toughness_Al6061":  (29.0,   "MPa√m",  "Fracture toughness, 6061-T6 Al"),
        "Poisson_ratio_steel":        (0.3,    "",       "Poisson's ratio, steel"),
        "Poisson_ratio_Al":           (0.33,   "",       "Poisson's ratio, aluminium"),
        "CTE_steel":                  (12e-6,  "1/K",    "Thermal expansion coefficient, steel"),
        "CTE_Al":                     (23e-6,  "1/K",    "Thermal expansion coefficient, aluminium"),
        "creep_exponent_steel":       (5.0,    "",       "Norton creep exponent, steel ~500°C"),
    },

    # ── 34. COMPOSITES ─────────────────────────────────────────────────────
    "composites": {
        "fiber_volume_fraction":      (0.60,   "",       "Fiber volume fraction, autoclave CFRP"),
        "E_fiber_T300":               (230e3,  "MPa",    "Longitudinal modulus, T300 carbon fiber"),
        "E_matrix_epoxy":             (3.5e3,  "MPa",    "Young's modulus, epoxy matrix"),
        "UTS_fiber_T300":             (3530.0, "MPa",    "UTS, T300 carbon fiber"),
        "UTS_CFRP_0deg":              (1500.0, "MPa",    "UTS, CFRP unidirectional, 0° (fiber dir)"),
        "UTS_CFRP_90deg":             (50.0,   "MPa",    "UTS, CFRP unidirectional, 90° (matrix)"),
        "ILSS_CFRP":                  (65.0,   "MPa",    "Interlaminar shear strength, CFRP"),
        "density_CFRP_prepreg":       (1550.0, "kg/m³",  "Density, CFRP prepreg laminate"),
        "density_GFRP":               (1900.0, "kg/m³",  "Density, GFRP E-glass/epoxy"),
        "cure_temp_standard":         (120.0,  "°C",     "Standard cure temperature, epoxy prepreg"),
        "cure_pressure_autoclave":    (600e3,  "Pa",     "Autoclave cure pressure"),
        "knockdown_impact":           (0.65,   "",       "CAI knockdown factor for CFRP"),
    },

    # ── 35. MANUFACTURING ──────────────────────────────────────────────────
    "manufacturing": {
        "surface_roughness_grinding": (0.4,    "µm Ra",  "Surface roughness, precision grinding"),
        "surface_roughness_milling":  (3.2,    "µm Ra",  "Surface roughness, end milling"),
        "surface_roughness_turning":  (1.6,    "µm Ra",  "Surface roughness, CNC turning"),
        "tolerance_IT6":              (0.016,  "mm",     "ISO IT6 tolerance, 50 mm nominal"),
        "tolerance_IT9":              (0.062,  "mm",     "ISO IT9 tolerance, 50 mm nominal"),
        "cutting_speed_Al":           (300.0,  "m/min",  "Cutting speed, aluminium CNC milling"),
        "cutting_speed_steel":        (80.0,   "m/min",  "Cutting speed, mild steel milling"),
        "feed_rate_typical":          (0.2,    "mm/tooth","Feed per tooth, carbide end mill"),
        "SLA_layer_thickness":        (0.1,    "mm",     "Layer thickness, SLA 3D printing"),
        "FDM_layer_thickness":        (0.2,    "mm",     "Layer thickness, FDM 3D printing"),
        "FDM_infill_structural":      (0.80,   "",       "FDM infill fraction for structural parts"),
        "press_fit_interference":     (0.025,  "mm",     "Typical press-fit interference, 50mm bore"),
    },

    # ── 36. WELDING ────────────────────────────────────────────────────────
    "welding": {
        "heat_input_GMAW":            (1.0,    "kJ/mm",  "Heat input, GMAW (MIG) typical"),
        "heat_input_GTAW":            (0.5,    "kJ/mm",  "Heat input, GTAW (TIG) typical"),
        "deposition_rate_GMAW":       (4.0,    "kg/h",   "Deposition rate, GMAW"),
        "preheat_carbon_steel_T1":    (50.0,   "°C",     "Preheat temp, medium-carbon steel (t<25mm)"),
        "HAZ_width_GMAW":             (5.0,    "mm",     "HAZ width, single-pass GMAW"),
        "joint_efficiency_butt":      (1.0,    "",       "Joint efficiency, full-penetration butt weld"),
        "joint_efficiency_fillet":    (0.7,    "",       "Joint efficiency, fillet weld (shear plane)"),
        "electrode_E7018_UTS":        (490.0,  "MPa",    "UTS, E7018 weld metal"),
        "distortion_factor":          (0.001,  "mm/mm",  "Transverse shrinkage per unit weld length"),
    },

    # ── 37. GEARBOX / POWER TRANSMISSION ──────────────────────────────────
    "gearbox": {
        "efficiency_spur":            (0.98,   "",       "Efficiency, single spur gear pair"),
        "efficiency_helical":         (0.98,   "",       "Efficiency, single helical gear pair"),
        "efficiency_bevel":           (0.96,   "",       "Efficiency, bevel gear pair"),
        "efficiency_worm_high_ratio": (0.5,    "",       "Efficiency, high-ratio worm gear"),
        "efficiency_planetary":       (0.97,   "",       "Efficiency, planetary stage"),
        "gear_ratio_max_spur":        (6.0,    "",       "Max gear ratio, single spur stage"),
        "gear_ratio_max_planetary":   (9.0,    "",       "Max gear ratio, single planetary stage"),
        "service_factor_uniform":     (1.0,    "",       "Service/application factor, uniform load"),
        "service_factor_heavy":       (2.0,    "",       "Service/application factor, heavy shock"),
        "Lewis_form_factor_20deg":    (0.32,   "",       "Lewis form factor Y, 20° pressure angle"),
        "backlash_standard":          (0.1,    "mm",     "Backlash, standard commercial gears"),
    },

    # ── 38. BEARING ────────────────────────────────────────────────────────
    "bearing": {
        "L10_life_hours":             (20000.0,"h",      "L10 bearing life, industrial machinery"),
        "L10_life_hours_automotive":  (3000.0, "h",      "L10 life, automotive wheel bearing"),
        "dynamic_load_factor_p":      (3.0,    "",       "Load exponent p, ball bearing"),
        "dynamic_load_factor_p_roller":(10.0/3,"",       "Load exponent p, roller bearing"),
        "viscosity_ISO_VG46":         (46.0,   "mm²/s",  "Kinematic viscosity, ISO VG 46 oil at 40°C"),
        "preload_angular_contact":    (0.05,   "",       "Preload as fraction of dynamic load rating"),
        "fatigue_limit_factor_Cu1":   (0.5,    "",       "Cu1 life modification factor (clean lube)"),
        "friction_coeff_ball":        (0.001,  "",       "Rolling friction coefficient, ball bearing"),
        "friction_coeff_roller":      (0.002,  "",       "Rolling friction coefficient, roller bearing"),
        "clearance_normal_C3":        (1.0,    "",       "C3 = normal clearance class (relative)"),
    },

    # ── 39. VIBRATION / DYNAMICS ───────────────────────────────────────────
    "vibration": {
        "damping_ratio_steel_structure": (0.01,"",       "Damping ratio, welded steel structure"),
        "damping_ratio_concrete":     (0.05,   "",       "Damping ratio, reinforced concrete"),
        "damping_ratio_rubber_mount": (0.15,   "",       "Damping ratio, rubber isolation mount"),
        "isolation_efficiency_1Hz_mount":(0.90,"",       "Vibration isolation eff at 10× nat. freq"),
        "frequency_ratio_isolation":  (2.5,    "",       "Min excitation/natural freq for isolation"),
        "TMD_mass_ratio":             (0.02,   "",       "Tuned mass damper mass ratio (typical)"),
        "Rayleigh_damping_alpha":     (0.05,   "1/s",    "Rayleigh alpha (mass-proportional damping)"),
        "Rayleigh_damping_beta":      (0.001,  "s",      "Rayleigh beta (stiffness-proportional)"),
        "floor_acceleration_limit":   (0.005,  "g",      "Human perception threshold, floor vibration"),
    },

    # ── 40. FATIGUE ────────────────────────────────────────────────────────
    "fatigue": {
        "fatigue_limit_steel_fraction": (0.5,  "",       "Endurance limit ≈ 0.5 UTS, steel (R=-1)"),
        "fatigue_limit_Al_fraction":  (0.35,   "",       "Fatigue strength @ 5×10⁸ cycles ≈ 0.35 UTS"),
        "SN_slope_k_steel":           (3.0,    "",       "S-N slope k, steel (stress-life)"),
        "SN_slope_k_Al":              (4.0,    "",       "S-N slope k, aluminium (stress-life)"),
        "stress_concentration_fillet":(2.5,    "",       "Stress concentration factor Kt, fillet"),
        "stress_concentration_hole":  (3.0,    "",       "Stress concentration factor Kt, circular hole"),
        "surface_factor_machined":    (0.9,    "",       "Surface finish factor, machined steel"),
        "reliability_factor_99pct":   (0.814,  "",       "Reliability factor ka, 99% reliability"),
        "Miner_rule_damage_limit":    (1.0,    "",       "Miner's rule cumulative damage limit"),
        "rain_flow_cycle_fraction":   (0.7,    "",       "Fraction of cycles counted by rainflow"),
    },

    # ── 41. IMPACT / CRASH ─────────────────────────────────────────────────
    "impact": {
        "coefficient_restitution_steel": (0.8, "",       "Coefficient of restitution, steel-steel"),
        "coefficient_restitution_rubber": (0.9,"",       "Coefficient of restitution, rubber ball"),
        "coefficient_restitution_clay": (0.1,  "",       "Coefficient of restitution, clay"),
        "HIC_threshold_head":         (1000.0, "",       "Head Injury Criterion (HIC) limit"),
        "g_limit_chest_airbag":       (60.0,   "g",      "Chest deceleration limit, airbag deployment"),
        "crush_distance_car_frontal": (0.5,    "m",      "Crumple zone crush distance, typical car"),
        "energy_absorb_foam":         (5e4,    "J/m³",   "Energy absorbed, EPS foam at 10% crush"),
        "strain_rate_crash":          (100.0,  "1/s",    "Typical strain rate, automotive crash"),
        "dynamic_amplification_factor": (2.0,  "",       "DAF for sudden loading (impulse)"),
    },

    # ── 42. PNEUMATICS ─────────────────────────────────────────────────────
    "pneumatics": {
        "P_shop_air":                 (7e5,    "Pa",     "Shop compressed air pressure (7 bar)"),
        "P_instrument_air":           (6e5,    "Pa",     "Instrument air supply pressure (6 bar)"),
        "air_consumption_cylinder":   (0.0005, "m³/stroke","Air consumption, 50mm bore × 100mm stroke"),
        "compressor_efficiency":      (0.75,   "",       "Reciprocating compressor efficiency"),
        "flow_coeff_Cv_valve":        (1.0,    "",       "Flow coefficient Cv, typical control valve"),
        "leakage_system":             (0.10,   "",       "Typical system leakage fraction of supply"),
        "pipe_velocity_compressed":   (10.0,   "m/s",    "Compressed air pipe velocity (velocity limit)"),
        "dew_point_instrument_air":   (-40.0,  "°C",     "Dew point, dried instrument air"),
    },

    # ── 43. OPEN-CHANNEL HYDRAULICS ────────────────────────────────────────
    "hydraulics": {
        "Manning_n_concrete":         (0.013,  "",       "Manning roughness, concrete channel"),
        "Manning_n_earthen":          (0.025,  "",       "Manning roughness, earthen channel"),
        "Manning_n_riprap":           (0.04,   "",       "Manning roughness, riprap lining"),
        "Froude_subcritical":         (0.9,    "",       "Max Froude number, subcritical design"),
        "weir_Cd":                    (0.62,   "",       "Discharge coefficient, sharp-crested weir"),
        "orifice_Cd":                 (0.61,   "",       "Discharge coefficient, sharp-edged orifice"),
        "dam_safety_factor_sliding":  (1.5,    "",       "Safety factor against sliding, dam"),
        "dam_safety_factor_overturning": (2.0, "",       "Safety factor against overturning, dam"),
        "flood_return_period_major":  (100.0,  "years",  "Design return period, major infrastructure"),
        "g_hydraulics":               (9.80665,"m/s²",   "Gravitational acceleration"),
    },

    # ── 44. VACUUM SYSTEMS ─────────────────────────────────────────────────
    "vacuum": {
        "P_rough":                    (1e4,    "Pa",     "Rough vacuum upper limit"),
        "P_medium":                   (0.1,    "Pa",     "Medium vacuum range"),
        "P_high":                     (1e-4,   "Pa",     "High vacuum lower limit"),
        "P_ultra_high":               (1e-7,   "Pa",     "Ultra-high vacuum lower limit"),
        "pumping_speed_turbo_200":    (0.20,   "m³/s",   "Pumping speed, 200 L/s turbomolecular pump"),
        "throughput_rotary_vane":     (10.0,   "Pa·m³/s","Throughput, rotary vane rough pump"),
        "outgassing_rate_steel":      (1e-9,   "Pa·m³/(s·m²)","Outgassing rate, clean stainless steel"),
        "outgassing_rate_Al":         (1e-8,   "Pa·m³/(s·m²)","Outgassing rate, clean aluminium"),
        "compression_ratio_turbo":    (1e8,    "",       "Compression ratio, turbomolecular pump, N2"),
        "mean_free_path_1e-4Pa":      (0.1,    "m",      "Mean free path, N2 at 10⁻⁴ Pa"),
    },

    # ── 45. SEMICONDUCTOR / DEVICE PHYSICS ────────────────────────────────
    "semiconductor": {
        "ni_Si_300K":                 (1.5e10, "cm⁻³",  "Intrinsic carrier density, Si at 300 K"),
        "Eg_Si":                      (1.12,   "eV",    "Bandgap energy, silicon at 300 K"),
        "Eg_GaAs":                    (1.42,   "eV",    "Bandgap energy, GaAs at 300 K"),
        "mobility_e_Si":              (1400.0, "cm²/(V·s)","Electron mobility, Si at 300 K"),
        "mobility_h_Si":              (450.0,  "cm²/(V·s)","Hole mobility, Si at 300 K"),
        "epsilon_Si":                 (11.7,   "",       "Relative permittivity, silicon"),
        "epsilon_SiO2":               (3.9,    "",       "Relative permittivity, SiO2"),
        "breakdown_Si":               (3e7,    "V/m",    "Breakdown field strength, silicon"),
        "MOSFET_channel_L_modern":    (7e-9,   "m",      "Channel length, leading-edge CMOS node"),
        "gate_oxide_SiO2_EOT":        (1.5e-9, "m",      "Equivalent gate oxide thickness, modern CMOS"),
    },

    # ── 46. BATTERY ELECTROCHEMISTRY ───────────────────────────────────────
    "battery_chem": {
        "cell_voltage_LiPo":          (3.7,    "V",      "Nominal cell voltage, LiPo"),
        "cell_voltage_LiFePO4":       (3.2,    "V",      "Nominal cell voltage, LiFePO4"),
        "cell_voltage_NiMH":          (1.2,    "V",      "Nominal cell voltage, NiMH"),
        "cell_voltage_lead_acid":     (2.0,    "V",      "Nominal cell voltage, lead-acid"),
        "C_rate_continuous_LiPo":     (1.0,    "C",      "Continuous discharge rate limit, LiPo"),
        "C_rate_peak_LiPo":           (5.0,    "C",      "Peak discharge rate, LiPo (short duration)"),
        "round_trip_efficiency_Li":   (0.96,   "",       "Coulombic efficiency, lithium-ion"),
        "self_discharge_Li":          (0.02,   "1/month","Self-discharge rate, lithium-ion"),
        "cycle_life_LiPo":            (500.0,  "cycles", "Cycle life to 80% SoH, LiPo"),
        "cycle_life_LiFePO4":         (2000.0, "cycles", "Cycle life to 80% SoH, LiFePO4"),
        "SOC_safe_min":               (0.20,   "",       "Minimum safe state-of-charge"),
        "SOC_safe_max":               (0.90,   "",       "Maximum safe state-of-charge (longevity)"),
        "temperature_limit_charge":   (45.0,   "°C",     "Max temperature during charging, Li-ion"),
    },

    # ── 47. NUCLEAR POWER ──────────────────────────────────────────────────
    "nuclear_power": {
        "thermal_efficiency_PWR":     (0.33,   "",       "Thermal efficiency, pressurized water reactor"),
        "thermal_efficiency_BWR":     (0.33,   "",       "Thermal efficiency, boiling water reactor"),
        "power_density_core_PWR":     (100e6,  "W/m³",   "Core power density, PWR"),
        "T_coolant_outlet_PWR":       (325.0,  "°C",     "Coolant outlet temperature, PWR primary"),
        "P_primary_coolant":          (15.5e6, "Pa",     "Primary coolant pressure, PWR (155 bar)"),
        "capacity_factor_nuclear":    (0.93,   "",       "Capacity factor, nuclear plant (US avg)"),
        "burnup_max_LWR":             (50000.0,"MWd/tU", "Max burnup, LWR fuel (regulatory limit)"),
        "enrichment_PWR":             (0.045,  "",       "U-235 enrichment, PWR fresh fuel"),
        "refueling_interval":         (1.5,    "years",  "Refueling interval, 18-month cycle"),
        "decay_heat_fraction_1s":     (0.065,  "",       "Decay heat as fraction of full power, t=1s"),
    },

    # ── 48. GEOPHYSICS / SEISMOLOGY ────────────────────────────────────────
    "geophysics": {
        "v_P_crust":                  (6000.0, "m/s",    "P-wave velocity, continental crust"),
        "v_S_crust":                  (3500.0, "m/s",    "S-wave velocity, continental crust"),
        "v_P_mantle":                 (8100.0, "m/s",    "P-wave velocity, upper mantle"),
        "density_crust":              (2800.0, "kg/m³",  "Average continental crust density"),
        "density_mantle":             (3300.0, "kg/m³",  "Upper mantle density"),
        "g_surface":                  (9.80665,"m/s²",   "Mean surface gravitational acceleration"),
        "geothermal_gradient":        (30.0,   "°C/km",  "Average geothermal gradient, continental"),
        "Richter_energy_factor":      (31.6,   "",       "Energy ratio per Richter magnitude step"),
        "seismic_attenuation_Q":      (200.0,  "",       "Quality factor Q, crust (attenuation)"),
        "GPS_accuracy_differential":  (0.01,   "m",      "Differential GPS position accuracy"),
    },

    # ── 49. BIOMEDICAL ENGINEERING ─────────────────────────────────────────
    "biomedical": {
        "E_cortical_bone":            (20e3,   "MPa",    "Young's modulus, cortical bone"),
        "E_cartilage":                (1.0,    "MPa",    "Young's modulus, articular cartilage"),
        "UTS_cortical_bone":          (130.0,  "MPa",    "UTS, cortical bone"),
        "yield_Ti6Al4V_implant":      (860.0,  "MPa",    "Yield strength, ASTM F136 Ti-6Al-4V"),
        "heart_rate_rest":            (70.0,   "bpm",    "Resting heart rate"),
        "cardiac_output_rest":        (5.0,    "L/min",  "Cardiac output at rest"),
        "blood_viscosity":            (3e-3,   "Pa·s",   "Blood dynamic viscosity"),
        "blood_density":              (1060.0, "kg/m³",  "Blood density"),
        "aortic_pressure_systolic":   (16000.0,"Pa",     "Systolic aortic pressure (120 mmHg)"),
        "ISO_biocompatibility_class": (1.0,    "",       "ISO 10993 risk class I (lowest risk)"),
        "sterilization_temp_autoclave": (121.0,"°C",     "Steam autoclave sterilization temp, 15 min"),
        "fatigue_limit_implant_Ti":   (550.0,  "MPa",    "Fatigue limit, Ti-6Al-4V implant (R=-1)"),
    },

    # ── 50. MINING / BLASTING ──────────────────────────────────────────────
    "mining": {
        "ANFO_energy_density":        (3.7e6,  "J/kg",   "Explosive energy, ANFO"),
        "TNT_energy_density":         (4.6e6,  "J/kg",   "Explosive energy, TNT"),
        "powder_factor_hard_rock":    (0.4,    "kg/m³",  "Explosives per m³ rock, hard rock"),
        "powder_factor_soft_rock":    (0.15,   "kg/m³",  "Explosives per m³ rock, soft rock"),
        "rock_density_granite":       (2700.0, "kg/m³",  "Density, granite"),
        "UCS_granite":                (150.0,  "MPa",    "Unconfined compressive strength, granite"),
        "detonation_velocity_ANFO":   (4500.0, "m/s",    "Detonation velocity, ANFO"),
        "bench_height_typical":       (10.0,   "m",      "Bench height, open-pit mining"),
        "stemming_ratio":             (0.7,    "",       "Stemming length / burden ratio"),
        "fragmentation_target_P80":   (0.3,    "m",      "Target P80 fragment size"),
        "shovel_cycle_time":          (30.0,   "s",      "Electric shovel dig cycle time"),
    },
}


# ─── DOMAIN ALIASES ───────────────────────────────────────────────────────────

DOMAIN_ALIASES: Dict[str, str] = {
    # rocket
    "rocket":                  "rocket",
    "launch vehicle":          "rocket",
    "launch_vehicle":          "rocket",
    "spacecraft":              "rocket",
    "propulsion":              "rocket",
    "launch":                  "rocket",
    # motor
    "motor":                   "motor",
    "electric motor":          "motor",
    "actuator":                "motor",
    "drive":                   "motor",
    "bldc":                    "motor",
    "servo":                   "motor",
    # structure
    "structure":               "structure",
    "structural":              "structure",
    "frame":                   "structure",
    "beam":                    "structure",
    "truss":                   "structure",
    # thermal
    "thermal":                 "thermal",
    "heat":                    "thermal",
    "cooling":                 "thermal",
    "conduction":              "thermal",
    "convection":              "thermal",
    # power
    "power":                   "power",
    "battery":                 "power",
    "electrical":              "power",
    "energy":                  "power",
    # fluid
    "fluid":                   "fluid",
    "pipe":                    "fluid",
    "pump":                    "fluid",
    "hydraulic":               "hydraulics",
    "pneumatic":               "pneumatics",
    # ballistics
    "ballistics":              "ballistics",
    "projectile":              "ballistics",
    "turret":                  "ballistics",
    "shooter":                 "ballistics",
    "shooting":                "ballistics",
    "trajectory":              "ballistics",
    "frc":                     "ballistics",
    "lead angle":              "ballistics",
    "shoot on the move":       "ballistics",
    "gun":                     "ballistics",
    "cannon":                  "ballistics",
    "mortar":                  "ballistics",
    # orbital
    "orbital":                 "orbital_mechanics",
    "orbital mechanics":       "orbital_mechanics",
    "orbit":                   "orbital_mechanics",
    "hohmann":                 "orbital_mechanics",
    "kepler":                  "orbital_mechanics",
    "satellite":               "orbital_mechanics",
    # aerodynamics
    "aerodynamics":            "aerodynamics",
    "aero":                    "aerodynamics",
    "lift":                    "aerodynamics",
    "drag":                    "aerodynamics",
    "airfoil":                 "aerodynamics",
    "wing":                    "aerodynamics",
    # aircraft
    "aircraft":                "aircraft",
    "airplane":                "aircraft",
    "plane":                   "aircraft",
    "fixed wing":              "aircraft",
    "fixed-wing":              "aircraft",
    "airliner":                "aircraft",
    "aviation":                "aircraft",
    # rotorcraft
    "rotorcraft":              "rotorcraft",
    "helicopter":              "rotorcraft",
    "drone":                   "rotorcraft",
    "multirotor":              "rotorcraft",
    "quadcopter":              "rotorcraft",
    "uav":                     "rotorcraft",
    # automotive
    "automotive":              "automotive",
    "vehicle":                 "automotive",
    "car":                     "automotive",
    "truck":                   "automotive",
    "ev":                      "automotive",
    "electric vehicle":        "automotive",
    # robotics
    "robotics":                "robotics",
    "robot":                   "robotics",
    "frc robot":               "robotics",
    "first robotics":          "robotics",
    "kinematics":              "robotics",
    "arm":                     "robotics",
    # controls
    "controls":                "controls",
    "control system":          "controls",
    "pid":                     "controls",
    "controller":              "controls",
    "feedback":                "controls",
    "stability":               "controls",
    # electronics
    "electronics":             "electronics",
    "circuit":                 "electronics",
    "pcb":                     "electronics",
    "microcontroller":         "electronics",
    "mcu":                     "electronics",
    "embedded":                "electronics",
    # rf
    "rf":                      "rf",
    "antenna":                 "rf",
    "wireless":                "rf",
    "radio":                   "rf",
    "microwave":               "rf",
    "signal":                  "rf",
    # optics
    "optics":                  "optics",
    "laser":                   "optics",
    "lens":                    "optics",
    "optical":                 "optics",
    "photonics":               "optics",
    # acoustics
    "acoustics":               "acoustics",
    "sound":                   "acoustics",
    "noise":                   "acoustics",
    "vibration acoustics":     "acoustics",
    # chemical
    "chemical":                "chemical",
    "chemistry":               "chemical",
    "reaction":                "chemical",
    "process":                 "chemical",
    # combustion
    "combustion":              "combustion",
    "flame":                   "combustion",
    "burning":                 "combustion",
    "fuel":                    "combustion",
    # gas turbine
    "gas turbine":             "gas_turbine",
    "gas_turbine":             "gas_turbine",
    "jet engine":              "gas_turbine",
    "turbofan":                "gas_turbine",
    "turbojet":                "gas_turbine",
    "brayton":                 "gas_turbine",
    # steam turbine
    "steam turbine":           "steam_turbine",
    "steam_turbine":           "steam_turbine",
    "rankine":                 "steam_turbine",
    "steam":                   "steam_turbine",
    # refrigeration
    "refrigeration":           "refrigeration",
    "refrigerant":             "refrigeration",
    "chiller":                 "refrigeration",
    "air conditioning":        "refrigeration",
    "heat pump":               "refrigeration",
    # heat exchanger
    "heat exchanger":          "heat_exchanger",
    "heat_exchanger":          "heat_exchanger",
    "hx":                      "heat_exchanger",
    "radiator":                "heat_exchanger",
    # cryogenics
    "cryogenics":              "cryogenics",
    "cryo":                    "cryogenics",
    "liquid nitrogen":         "cryogenics",
    "liquid hydrogen":         "cryogenics",
    "ln2":                     "cryogenics",
    "lh2":                     "cryogenics",
    "lox":                     "cryogenics",
    # nuclear
    "nuclear":                 "nuclear",
    "fission":                 "nuclear",
    "radiation":               "nuclear",
    "shielding":               "nuclear",
    # solar pv
    "solar":                   "solar_pv",
    "solar pv":                "solar_pv",
    "photovoltaic":            "solar_pv",
    "pv":                      "solar_pv",
    # wind
    "wind":                    "wind_turbine",
    "wind turbine":            "wind_turbine",
    "wind energy":             "wind_turbine",
    # hvac
    "hvac":                    "hvac",
    "ventilation":             "hvac",
    "air handling":            "hvac",
    "building hvac":           "hvac",
    # civil
    "civil":                   "civil",
    "concrete":                "civil",
    "column":                  "civil",
    "foundation":              "civil",
    "slab":                    "civil",
    # geotechnical
    "geotechnical":            "geotechnical",
    "soil":                    "geotechnical",
    "slope":                   "geotechnical",
    "embankment":              "geotechnical",
    # marine
    "marine":                  "marine",
    "ship":                    "marine",
    "boat":                    "marine",
    "vessel":                  "marine",
    "naval":                   "marine",
    "propeller":               "marine",
    # materials
    "materials":               "materials",
    "material":                "materials",
    "alloy":                   "materials",
    "metal":                   "materials",
    # composites
    "composites":              "composites",
    "composite":               "composites",
    "cfrp":                    "composites",
    "carbon fiber":            "composites",
    "fiberglass":              "composites",
    "gfrp":                    "composites",
    # manufacturing
    "manufacturing":           "manufacturing",
    "machining":               "manufacturing",
    "cnc":                     "manufacturing",
    "3d printing":             "manufacturing",
    "additive":                "manufacturing",
    "milling":                 "manufacturing",
    # welding
    "welding":                 "welding",
    "weld":                    "welding",
    "mig":                     "welding",
    "tig":                     "welding",
    # gearbox
    "gearbox":                 "gearbox",
    "gear":                    "gearbox",
    "transmission":            "gearbox",
    "reducer":                 "gearbox",
    "planetary":               "gearbox",
    # bearing
    "bearing":                 "bearing",
    "bushing":                 "bearing",
    "rolling element":         "bearing",
    # vibration
    "vibration":               "vibration",
    "dynamic":                 "vibration",
    "resonance":               "vibration",
    "modal":                   "vibration",
    # fatigue
    "fatigue":                 "fatigue",
    "cyclic":                  "fatigue",
    "s-n":                     "fatigue",
    "endurance":               "fatigue",
    # impact
    "impact":                  "impact",
    "crash":                   "impact",
    "collision":               "impact",
    "impulse":                 "impact",
    # pneumatics
    "pneumatics":              "pneumatics",
    "compressed air":          "pneumatics",
    "pneumatic cylinder":      "pneumatics",
    # hydraulics (open channel)
    "open channel":            "hydraulics",
    "dam":                     "hydraulics",
    "weir":                    "hydraulics",
    "channel":                 "hydraulics",
    # vacuum
    "vacuum":                  "vacuum",
    "turbomolecular":          "vacuum",
    "outgassing":              "vacuum",
    # semiconductor
    "semiconductor":           "semiconductor",
    "transistor":              "semiconductor",
    "mosfet":                  "semiconductor",
    "diode":                   "semiconductor",
    "cmos":                    "semiconductor",
    # battery chemistry
    "battery chemistry":       "battery_chem",
    "battery_chem":            "battery_chem",
    "electrochemistry":        "battery_chem",
    "cell":                    "battery_chem",
    "liion":                   "battery_chem",
    "lithium":                 "battery_chem",
    # nuclear power
    "nuclear power":           "nuclear_power",
    "nuclear_power":           "nuclear_power",
    "reactor":                 "nuclear_power",
    "pwr":                     "nuclear_power",
    "bwr":                     "nuclear_power",
    # geophysics
    "geophysics":              "geophysics",
    "seismic":                 "geophysics",
    "earthquake":              "geophysics",
    "seismology":              "geophysics",
    # biomedical
    "biomedical":              "biomedical",
    "medical":                 "biomedical",
    "implant":                 "biomedical",
    "biomechanics":            "biomedical",
    "prosthetic":              "biomedical",
    # mining
    "mining":                  "mining",
    "blasting":                "mining",
    "drilling":                "mining",
    "quarrying":               "mining",
}


def get_defaults_for_domain(domain: str) -> Dict[str, Tuple]:
    """
    Return defaults dict for the given domain (or empty dict if unknown).
    Resolves aliases so callers can pass 'launch vehicle', 'frc', 'jet engine', etc.
    """
    key = DOMAIN_ALIASES.get(domain.lower().strip(), domain.lower().strip())
    return ENGINEERING_DEFAULTS.get(key, {})


def format_defaults_for_prompt(domain: str) -> str:
    """
    Format all defaults for a domain as Python comment lines suitable for
    inclusion in an LLM code-generation prompt.

    Returns one line per default:
        # Isp_LOX_RP1_sl = 311.0 s  — Merlin sea-level Isp
    """
    defaults = get_defaults_for_domain(domain)
    if not defaults:
        return f"# No pre-loaded defaults for domain: {domain}"
    lines = [f"# Engineering defaults for: {domain}"]
    for key, (value, unit, rationale) in defaults.items():
        unit_str = f" {unit}" if unit else ""
        lines.append(f"#   {key:<40} = {value}{unit_str}  — {rationale}")
    return "\n".join(lines)


# ─── PHYSICAL BOUNDS ──────────────────────────────────────────────────────────
# Structure: variable_key → (lower, upper, unit, lower_is_fatal, upper_is_fatal)
# None means no bound in that direction.

PHYSICAL_BOUNDS: Dict[str, Tuple] = {
    "mass":                   (0.0,    None,    "kg",   True,  False),
    "GTOW":                   (0.0,    5e8,     "kg",   True,  True),
    "payload_mass":           (0.0,    None,    "kg",   True,  False),
    "Isp":                    (50.0,   5000.0,  "s",    True,  True),
    "structural_fraction":    (0.02,   0.35,    "",     True,  True),
    "eps":                    (0.02,   0.35,    "",     True,  True),
    "delta_v":                (0.0,    150000.0,"m/s",  True,  True),
    "mass_ratio":             (1.0,    None,    "",     True,  False),
    "efficiency":             (0.0,    1.0,     "",     True,  True),
    "velocity":               (None,   3e8,     "m/s",  False, True),
    "thrust":                 (0.0,    None,    "N",    True,  False),
    "TW_ratio":               (0.0,    200.0,   "",     True,  True),
    "propellant_mass":        (0.0,    None,    "kg",   True,  False),
    "structural_mass":        (0.0,    None,    "kg",   True,  False),
    "temperature":            (-273.16,1e8,     "°C",   True,  True),
    "temperature_K":          (0.0,    1e8,     "K",    True,  True),
    "pressure":               (0.0,    None,    "Pa",   True,  False),
    "power":                  (None,   None,    "W",    False, False),
    "energy":                 (0.0,    None,    "J",    True,  False),
    "eccentricity":           (0.0,    1.0,     "",     True,  True),
    "altitude":               (-6.4e6, None,    "m",    True,  False),
    "exhaust_velocity":       (500.0,  50000.0, "m/s",  True,  True),
    "chamber_pressure":       (1e4,    5e8,     "Pa",   True,  True),
    "COP":                    (0.0,    30.0,    "",     True,  True),
    "stress":                 (0.0,    1e12,    "Pa",   True,  True),
    "safety_factor":          (0.01,   None,    "",     True,  False),
    "mach":                   (0.0,    50.0,    "",     True,  True),
    "frequency":              (0.0,    None,    "Hz",   True,  False),
    "lead_angle":             (-90.0,  90.0,    "deg",  True,  True),
}


@dataclass
class BoundsViolation:
    """Records a physical bounds violation found in computed results."""
    variable: str
    value: float
    bound_type: str     # "lower" or "upper"
    message: str
    is_fatal: bool


def check_physical_bounds(computed_values: Dict[str, float]) -> List[BoundsViolation]:
    """
    Scan a dict of computed results for physical bounds violations.
    Returns a list of BoundsViolation objects; empty list means all values pass.
    """
    violations: List[BoundsViolation] = []

    for var_key, value in computed_values.items():
        if not isinstance(value, (int, float)):
            continue
        for bound_key, (lower, upper, unit, lower_fatal, upper_fatal) in PHYSICAL_BOUNDS.items():
            if bound_key.lower() not in var_key.lower():
                continue
            if lower is not None and value < lower:
                violations.append(BoundsViolation(
                    variable=var_key, value=value, bound_type="lower",
                    message=f"{var_key} = {value:.4g} {unit} is below physical lower bound {lower} {unit}",
                    is_fatal=lower_fatal,
                ))
            if upper is not None and value > upper:
                violations.append(BoundsViolation(
                    variable=var_key, value=value, bound_type="upper",
                    message=f"{var_key} = {value:.4g} {unit} exceeds physical upper bound {upper} {unit}",
                    is_fatal=upper_fatal,
                ))
            break

    return violations


if __name__ == "__main__":
    print("Engineering Defaults Library v2")
    print("=" * 70)
    print(f"\nTotal domains: {len(ENGINEERING_DEFAULTS)}")
    print(f"Total aliases: {len(DOMAIN_ALIASES)}")
    print("\nAll domains:", sorted(ENGINEERING_DEFAULTS.keys()))
    total_defaults = sum(len(v) for v in ENGINEERING_DEFAULTS.values())
    print(f"\nTotal default entries: {total_defaults}")
    print("\nSample — ballistics:")
    print(format_defaults_for_prompt("ballistics"))
    print("\nSample — robotics:")
    print(format_defaults_for_prompt("frc robot"))
    print("\nBounds check:")
    viol = check_physical_bounds({"Isp": 9999, "lead_angle": 78.0, "mass": -5})
    for v in viol:
        print(f"  {'FATAL' if v.is_fatal else 'WARN'}: {v.message}")
