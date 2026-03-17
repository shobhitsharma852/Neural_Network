"""
Hybrid Microgrid Simulator  (HOMER Pro–style)
=============================================
An interactive Streamlit application that designs, simulates, and optimises
hybrid renewable energy systems — mirroring the core workflow of HOMER Pro:

  • Configure components: Solar PV, Wind Turbine, Battery, Diesel Generator, Grid
  • Set monthly resource data (irradiance, wind speed) and a load profile
  • Run an hour-by-hour (8 760-hour) dispatch simulation
  • Evaluate economics: Net Present Cost, LCOE, cost breakdown
  • Explore a sensitivity analysis on key parameters
"""

import math

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Hybrid Microgrid Simulator",
    page_icon="⚡",
    layout="wide",
)

# ─── Constants ────────────────────────────────────────────────────────────────
HOURS_PER_YEAR = 8_760
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
DAYS_PER_MONTH = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

# ─── Financial helpers ────────────────────────────────────────────────────────

def capital_recovery_factor(r: float, n: int) -> float:
    """Annualise a present-value cost over *n* years at discount rate *r*."""
    if r == 0:
        return 1.0 / n
    return r * (1 + r) ** n / ((1 + r) ** n - 1)


def pv_annuity_factor(r: float, n: int) -> float:
    """Present-value factor for a uniform annual series."""
    if r == 0:
        return float(n)
    return (1 - (1 + r) ** (-n)) / r


def replacement_pv(unit_cost: float, comp_life: int, proj_life: int, r: float) -> float:
    """PV of all mid-project replacement costs."""
    total = 0.0
    t = comp_life
    while t < proj_life:
        total += unit_cost / (1 + r) ** t
        t += comp_life
    return total


def salvage_pv(unit_cost: float, comp_life: int, proj_life: int, r: float) -> float:
    """PV of end-of-project salvage (straight-line depreciation)."""
    remainder = comp_life - (proj_life % comp_life)
    if proj_life % comp_life == 0:
        remainder = 0
    salvage = unit_cost * remainder / comp_life
    return salvage / (1 + r) ** proj_life


def component_npc(unit_cost: float, om_per_unit: float,
                  comp_life: int, proj_life: int, r: float, quantity: float):
    """Total NPC for a single component type."""
    if quantity == 0:
        return 0.0, 0.0, 0.0, 0.0
    capital = unit_cost * quantity
    om_pv = om_per_unit * quantity * pv_annuity_factor(r, proj_life)
    repl_pv = replacement_pv(unit_cost * quantity, comp_life, proj_life, r)
    salv_pv = salvage_pv(unit_cost * quantity, comp_life, proj_life, r)
    npc = capital + om_pv + repl_pv - salv_pv
    return npc, capital, om_pv + repl_pv, salv_pv


# ─── Default resource data ────────────────────────────────────────────────────
# Monthly average solar irradiance (kWh/m²/day) — generic mid-latitude site
DEFAULT_SOLAR_IRR = np.array(
    [3.5, 4.2, 5.1, 5.8, 6.3, 6.8, 6.6, 6.2, 5.5, 4.6, 3.8, 3.2]
)
# Monthly average wind speed (m/s)
DEFAULT_WIND_SPD = np.array(
    [5.5, 5.8, 6.2, 5.9, 5.3, 4.8, 4.5, 4.6, 5.1, 5.7, 5.9, 5.6]
)

# 24-hour normalised load shapes (peak = 1.0)
LOAD_SHAPES = {
    "Residential": np.array([
        0.35, 0.30, 0.28, 0.28, 0.30, 0.40, 0.55, 0.70, 0.75, 0.72,
        0.70, 0.72, 0.68, 0.65, 0.65, 0.68, 0.72, 0.85, 0.95, 1.00,
        0.95, 0.85, 0.70, 0.50,
    ]),
    "Commercial": np.array([
        0.20, 0.18, 0.18, 0.18, 0.20, 0.30, 0.50, 0.75, 0.90, 0.95,
        1.00, 0.98, 0.95, 0.95, 0.95, 0.90, 0.85, 0.75, 0.60, 0.45,
        0.35, 0.28, 0.23, 0.21,
    ]),
    "Industrial": np.array([
        0.70, 0.70, 0.70, 0.70, 0.72, 0.80, 0.90, 0.95, 0.98, 1.00,
        1.00, 0.98, 0.98, 1.00, 1.00, 0.98, 0.95, 0.90, 0.85, 0.80,
        0.75, 0.72, 0.70, 0.70,
    ]),
}
MONTHLY_LOAD_SCALE = {
    "Residential": np.array(
        [0.90, 0.88, 0.85, 0.82, 0.85, 0.95, 1.00, 1.00, 0.92, 0.87, 0.90, 0.95]
    ),
    "Commercial": np.array(
        [0.92, 0.90, 0.92, 0.94, 0.95, 0.96, 0.98, 1.00, 0.98, 0.97, 0.95, 0.93]
    ),
    "Industrial": np.array(
        [0.95, 0.90, 0.95, 0.96, 0.98, 0.99, 0.97, 0.98, 0.99, 1.00, 0.97, 0.95]
    ),
}


# ─── Profile generators ───────────────────────────────────────────────────────

def gen_load(peak_kw: float, profile: str) -> np.ndarray:
    """8760-hour load array [kW]."""
    shape = LOAD_SHAPES[profile]
    mscale = MONTHLY_LOAD_SCALE[profile]
    out = []
    for m, days in enumerate(DAYS_PER_MONTH):
        for _ in range(days):
            for h in range(24):
                out.append(peak_kw * shape[h] * mscale[m])
    return np.array(out[:HOURS_PER_YEAR])


def gen_solar_irr(monthly_kwh_m2: np.ndarray) -> np.ndarray:
    """8760-hour array of irradiance [kW/m²] from monthly daily averages."""
    # Half-sine daylight profile; peak = daily_kWh * π/24
    out = []
    for m, days in enumerate(DAYS_PER_MONTH):
        i_peak = monthly_kwh_m2[m] * math.pi / 24.0
        for _ in range(days):
            for h in range(24):
                if 6 <= h <= 18:
                    out.append(i_peak * math.sin(math.pi * (h - 6) / 12))
                else:
                    out.append(0.0)
    return np.array(out[:HOURS_PER_YEAR])


def gen_wind_speed(monthly_ms: np.ndarray) -> np.ndarray:
    """8760-hour wind speed array [m/s] with ±15% diurnal variation."""
    out = []
    for m, days in enumerate(DAYS_PER_MONTH):
        for _ in range(days):
            for h in range(24):
                variation = 1.0 + 0.15 * math.sin(math.pi * (h - 3) / 12)
                out.append(monthly_ms[m] * variation)
    return np.array(out[:HOURS_PER_YEAR])


def wind_power(speed: np.ndarray, cap_kw: float,
               v_cut_in: float = 3.0, v_rated: float = 12.0,
               v_cut_out: float = 25.0) -> np.ndarray:
    """Wind turbine power curve → hourly output [kW]."""
    out = np.zeros_like(speed)
    mask_partial = (speed >= v_cut_in) & (speed < v_rated)
    mask_rated = (speed >= v_rated) & (speed < v_cut_out)
    out[mask_partial] = cap_kw * ((speed[mask_partial] - v_cut_in) /
                                  (v_rated - v_cut_in)) ** 3
    out[mask_rated] = cap_kw
    return out


# ─── Dispatch simulation ──────────────────────────────────────────────────────

def simulate(
    load_kw: np.ndarray,
    solar_kw: float, solar_irr: np.ndarray, derating: float,
    wind_out: np.ndarray,
    batt_cap_kwh: float, batt_eff: float, batt_min_soc: float,
    gen_cap_kw: float, fuel_slope: float, fuel_intercept: float, fuel_price: float,
    grid_enabled: bool, grid_buy_price: float, grid_sell_price: float,
) -> dict:
    """Hour-by-hour load-following dispatch; returns result arrays and totals."""

    solar_out = solar_irr * solar_kw * derating

    bat_max = batt_cap_kwh
    bat_min = batt_min_soc * batt_cap_kwh
    sqrt_eff = math.sqrt(batt_eff)

    soc = np.empty(HOURS_PER_YEAR + 1)
    soc[0] = 0.5 * batt_cap_kwh  # start at 50 %

    gen_output = np.zeros(HOURS_PER_YEAR)
    batt_delta = np.zeros(HOURS_PER_YEAR)   # + = charging, – = discharging
    unmet = np.zeros(HOURS_PER_YEAR)
    curtailed = np.zeros(HOURS_PER_YEAR)
    fuel_L = np.zeros(HOURS_PER_YEAR)
    grid_buy_arr = np.zeros(HOURS_PER_YEAR)
    grid_sell_arr = np.zeros(HOURS_PER_YEAR)

    for h in range(HOURS_PER_YEAR):
        renew = solar_out[h] + wind_out[h]
        demand = load_kw[h]
        net = renew - demand  # > 0 = surplus

        if net >= 0:
            # Surplus → charge battery
            space = (bat_max - soc[h]) / sqrt_eff
            charge = min(net, space)
            batt_delta[h] = charge
            soc[h + 1] = soc[h] + charge * sqrt_eff
            leftover = net - charge
            if grid_enabled and leftover > 0:
                grid_sell_arr[h] = leftover
            else:
                curtailed[h] = leftover
        else:
            deficit = -net
            # Discharge battery
            avail = max(0.0, (soc[h] - bat_min) * sqrt_eff)
            discharge = min(deficit, avail)
            batt_delta[h] = -discharge / sqrt_eff
            soc[h + 1] = soc[h] - discharge / sqrt_eff
            rem = deficit - discharge

            if rem > 0:
                if grid_enabled:
                    grid_buy_arr[h] = rem
                    rem = 0.0
                elif gen_cap_kw > 0:
                    gen_out = min(rem, gen_cap_kw)
                    fuel_L[h] = fuel_intercept * gen_cap_kw + fuel_slope * gen_out
                    gen_output[h] = gen_out
                    rem -= gen_out
            unmet[h] = rem

    solar_total = solar_out.sum()
    wind_total = wind_out.sum()
    gen_total = gen_output.sum()
    grid_buy_total = grid_buy_arr.sum()
    grid_sell_total = grid_sell_arr.sum()
    unmet_total = unmet.sum()
    curtailed_total = curtailed.sum()
    fuel_total = fuel_L.sum()
    load_total = load_kw.sum()
    renew_total = solar_total + wind_total
    rf = renew_total / max(load_total + curtailed_total, 1)

    return dict(
        # hourly arrays
        solar_out=solar_out,
        wind_out=wind_out,
        gen_output=gen_output,
        batt_delta=batt_delta,
        soc=soc[:HOURS_PER_YEAR],
        unmet=unmet,
        curtailed=curtailed,
        fuel_arr=fuel_L,
        grid_buy_arr=grid_buy_arr,
        grid_sell_arr=grid_sell_arr,
        load_arr=load_kw,
        # annual totals
        solar=solar_total,
        wind=wind_total,
        gen=gen_total,
        grid_buy=grid_buy_total,
        grid_sell=grid_sell_total,
        unmet_kwh=unmet_total,
        curtailed_kwh=curtailed_total,
        fuel_L=fuel_total,
        load=load_total,
        renew=renew_total,
        renewable_fraction=rf,
    )


# ─── Economics ────────────────────────────────────────────────────────────────

def economics(res: dict, cfg: dict) -> dict:
    r = cfg["discount_rate"]
    n = cfg["project_lifetime"]

    sol_npc, sol_cap, sol_om, sol_salv = component_npc(
        cfg["solar_cost"], cfg["solar_om"], cfg["solar_life"], n, r,
        cfg["solar_kw"])
    wnd_npc, wnd_cap, wnd_om, wnd_salv = component_npc(
        cfg["wind_cost"], cfg["wind_om"], cfg["wind_life"], n, r,
        cfg["wind_kw"])
    bat_npc, bat_cap, bat_om, bat_salv = component_npc(
        cfg["batt_cost"], cfg["batt_om"], cfg["batt_life"], n, r,
        cfg["batt_kwh"])
    gen_npc, gen_cap, gen_om, gen_salv = component_npc(
        cfg["gen_cost"], cfg["gen_om"], cfg["gen_life"], n, r,
        cfg["gen_kw"])

    fuel_annual = res["fuel_L"] * cfg["fuel_price"]
    fuel_pv = fuel_annual * pv_annuity_factor(r, n)

    if cfg["grid_enabled"]:
        grid_annual = (res["grid_buy"] * cfg["grid_buy_price"]
                       - res["grid_sell"] * cfg["grid_sell_price"])
        grid_pv = grid_annual * pv_annuity_factor(r, n)
    else:
        grid_annual = 0.0
        grid_pv = 0.0

    total_npc = sol_npc + wnd_npc + bat_npc + gen_npc + fuel_pv + grid_pv
    served_kwh = (res["load"] - res["unmet_kwh"]) * pv_annuity_factor(r, n)
    lcoe = total_npc / max(served_kwh, 1)

    return dict(
        npc=total_npc,
        lcoe=lcoe,
        sol_npc=sol_npc, wnd_npc=wnd_npc, bat_npc=bat_npc,
        gen_npc=gen_npc, fuel_pv=fuel_pv, grid_pv=grid_pv,
        capital={
            "Solar PV": sol_cap, "Wind Turbine": wnd_cap,
            "Battery": bat_cap, "Generator": gen_cap,
        },
        annual_fuel=fuel_annual,
        annual_grid=grid_annual,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  STREAMLIT UI
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("## ⚡ Hybrid Microgrid Simulator  *(HOMER Pro–style)*")
st.caption(
    "Design a hybrid power system, run an 8 760-hour simulation, and analyse "
    "costs — similar to HOMER Pro. Adjust components in the sidebar and click "
    "**▶ Run Simulation**."
)

# ═══ SIDEBAR ══════════════════════════════════════════════════════════════════
with st.sidebar:
    st.header("🔧 System Configuration")

    # ── Project ──────────────────────────────────────────────────────────────
    with st.expander("📋 Project Settings", expanded=True):
        proj_life = st.slider("Project Lifetime (years)", 5, 30, 25)
        discount = st.slider("Discount Rate (%)", 1.0, 20.0, 8.0, 0.5) / 100
        inflation = st.slider("Inflation Rate (%)", 0.0, 10.0, 2.5, 0.5) / 100

    # ── Load ─────────────────────────────────────────────────────────────────
    with st.expander("⚡ Load", expanded=True):
        peak_load = st.slider("Peak Load (kW)", 1.0, 500.0, 50.0, 1.0)
        load_type = st.selectbox("Load Profile", list(LOAD_SHAPES.keys()))
        annual_load_kwh = peak_load * sum(
            LOAD_SHAPES[load_type].mean() * MONTHLY_LOAD_SCALE[load_type][m] * DAYS_PER_MONTH[m] * 24
            for m in range(12)
        )
        st.info(f"Estimated annual load: **{annual_load_kwh:,.0f} kWh/yr**")

    # ── Solar PV ─────────────────────────────────────────────────────────────
    with st.expander("☀️ Solar PV"):
        sol_en = st.checkbox("Enable Solar PV", value=True)
        sol_kw = st.slider("PV Capacity (kW)", 0.0, 500.0, 30.0, 1.0) if sol_en else 0.0
        sol_cost = st.number_input("Capital Cost ($/kW)", 200, 5000, 1200, 50) if sol_en else 0
        sol_om = st.number_input("O&M Cost ($/kW/yr)", 0, 100, 20, 5) if sol_en else 0
        sol_life = st.slider("Lifetime (years)", 10, 30, 25) if sol_en else 25
        sol_derate = st.slider("Derating Factor (%)", 70, 100, 90) / 100 if sol_en else 0.9

    # ── Wind ─────────────────────────────────────────────────────────────────
    with st.expander("🌬️ Wind Turbine"):
        wnd_en = st.checkbox("Enable Wind Turbine", value=False)
        wnd_kw = st.slider("Wind Capacity (kW)", 0.0, 500.0, 20.0, 1.0) if wnd_en else 0.0
        wnd_cost = st.number_input("Capital Cost ($/kW)", 500, 8000, 2500, 100) if wnd_en else 0
        wnd_om = st.number_input("O&M Cost ($/kW/yr)", 0, 200, 50, 10) if wnd_en else 0
        wnd_life = st.slider("Lifetime (years)", 10, 30, 20) if wnd_en else 20
        wnd_v_cut_in = st.slider("Cut-in Speed (m/s)", 1.0, 5.0, 3.0, 0.5) if wnd_en else 3.0
        wnd_v_rated = st.slider("Rated Speed (m/s)", 8.0, 20.0, 12.0, 0.5) if wnd_en else 12.0

    # ── Battery ──────────────────────────────────────────────────────────────
    with st.expander("🔋 Battery Storage"):
        bat_en = st.checkbox("Enable Battery", value=True)
        bat_kwh = st.slider("Capacity (kWh)", 0.0, 2000.0, 100.0, 5.0) if bat_en else 0.0
        bat_cost = st.number_input("Capital Cost ($/kWh)", 50, 2000, 500, 25) if bat_en else 0
        bat_om = st.number_input("O&M Cost ($/kWh/yr)", 0, 50, 10, 2) if bat_en else 0
        bat_life = st.slider("Lifetime (years)", 3, 20, 10) if bat_en else 10
        bat_eff = st.slider("Round-trip Efficiency (%)", 60, 100, 90) / 100 if bat_en else 0.9
        bat_min_soc = st.slider("Min. State of Charge (%)", 5, 50, 20) / 100 if bat_en else 0.2

    # ── Generator ────────────────────────────────────────────────────────────
    with st.expander("⛽ Diesel Generator"):
        gen_en = st.checkbox("Enable Generator", value=True)
        gen_kw = st.slider("Capacity (kW)", 0.0, 500.0, 20.0, 1.0) if gen_en else 0.0
        gen_cost = st.number_input("Capital Cost ($/kW)", 100, 2000, 400, 25) if gen_en else 0
        gen_om = st.number_input("O&M Cost ($/kW/yr)", 0, 100, 30, 5) if gen_en else 0
        gen_life = st.slider("Lifetime (years)", 3, 20, 15) if gen_en else 15
        fuel_price = st.number_input("Fuel Price ($/L)", 0.3, 5.0, 1.20, 0.05) if gen_en else 1.2
        # Fuel curve coefficients (L/hr): F = F0*P_nom + F1*P_out
        fuel_f0 = 0.08   # intercept coefficient
        fuel_f1 = 0.25   # slope coefficient

    # ── Grid ─────────────────────────────────────────────────────────────────
    with st.expander("🔌 Utility Grid"):
        grid_en = st.checkbox("Enable Grid Connection", value=False)
        grid_buy = st.number_input("Buy Price ($/kWh)", 0.05, 2.0, 0.15, 0.01) if grid_en else 0.15
        grid_sell = st.number_input("Sell Price ($/kWh)", 0.0, 1.0, 0.08, 0.01) if grid_en else 0.08

    st.markdown("---")

    # ── Resource Data ────────────────────────────────────────────────────────
    with st.expander("🌞 Monthly Solar Irradiance (kWh/m²/day)"):
        sol_irr = np.array([
            st.slider(m, 1.0, 10.0, float(DEFAULT_SOLAR_IRR[i]), 0.1,
                      key=f"irr_{i}")
            for i, m in enumerate(MONTHS)
        ])

    with st.expander("💨 Monthly Wind Speed (m/s)"):
        wind_spd = np.array([
            st.slider(m, 0.5, 20.0, float(DEFAULT_WIND_SPD[i]), 0.1,
                      key=f"wnd_{i}")
            for i, m in enumerate(MONTHS)
        ])

    run_btn = st.button("▶ Run Simulation", type="primary", use_container_width=True)

# ═══ MAIN CONTENT ═════════════════════════════════════════════════════════════

if not run_btn and "sim_result" not in st.session_state:
    st.info(
        "👈 Configure your system in the sidebar and click **▶ Run Simulation** to begin."
    )
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/thumb/b/b6/Image_created_with_a_mobile_phone.png/1200px-Image_created_with_a_mobile_phone.png",
        caption="Hybrid power systems combine solar, wind, batteries, and generators.",
        use_container_width=True,
    ) if False else None  # placeholder — skip actual image fetch

    # Show sample system diagram using Graphviz
    import graphviz
    d = graphviz.Digraph(engine="dot")
    d.attr(rankdir="LR", size="14,5", nodesep="0.5", ranksep="0.8")
    d.attr("node", shape="box", style="filled", fontsize="11")
    d.node("Solar", "☀️ Solar PV", fillcolor="lightyellow")
    d.node("Wind", "🌬️ Wind", fillcolor="lightcyan")
    d.node("Gen", "⛽ Generator", fillcolor="lightsalmon")
    d.node("Grid", "🔌 Grid", fillcolor="lavender")
    d.node("Bus", "AC Bus", fillcolor="gold", shape="ellipse")
    d.node("Batt", "🔋 Battery", fillcolor="lightgreen")
    d.node("Load", "⚡ Load", fillcolor="white")
    for src in ("Solar", "Wind", "Gen", "Grid", "Batt"):
        d.edge(src, "Bus")
    d.edge("Bus", "Load")
    d.edge("Bus", "Batt", style="dashed", label="charge/discharge")
    st.graphviz_chart(d, use_container_width=True)
    st.stop()

# ── Generate profiles ─────────────────────────────────────────────────────────
if run_btn:
    with st.spinner("Running 8 760-hour simulation …"):
        load_arr = gen_load(peak_load, load_type)
        irr_arr = gen_solar_irr(sol_irr)
        spd_arr = gen_wind_speed(wind_spd)
        wind_arr = wind_power(spd_arr, wnd_kw,
                              v_cut_in=wnd_v_cut_in, v_rated=wnd_v_rated)

        cfg = dict(
            project_lifetime=proj_life, discount_rate=discount,
            solar_kw=sol_kw, solar_cost=sol_cost, solar_om=sol_om, solar_life=sol_life,
            wind_kw=wnd_kw, wind_cost=wnd_cost, wind_om=wnd_om, wind_life=wnd_life,
            batt_kwh=bat_kwh, batt_cost=bat_cost, batt_om=bat_om, batt_life=bat_life,
            gen_kw=gen_kw, gen_cost=gen_cost, gen_om=gen_om, gen_life=gen_life,
            fuel_price=fuel_price, grid_enabled=grid_en,
            grid_buy_price=grid_buy, grid_sell_price=grid_sell,
        )

        res = simulate(
            load_arr,
            sol_kw, irr_arr, sol_derate,
            wind_arr,
            bat_kwh, bat_eff, bat_min_soc,
            gen_kw, fuel_f1, fuel_f0, fuel_price,
            grid_en, grid_buy, grid_sell,
        )
        econ = economics(res, cfg)

        st.session_state["sim_result"] = res
        st.session_state["econ_result"] = econ
        st.session_state["cfg"] = cfg
        st.session_state["load_arr"] = load_arr
        st.session_state["irr_arr"] = irr_arr
        st.session_state["wind_arr"] = wind_arr

res = st.session_state["sim_result"]
econ = st.session_state["econ_result"]
cfg = st.session_state["cfg"]
load_arr = st.session_state["load_arr"]

# ── KPI strip ─────────────────────────────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Net Present Cost", f"${econ['npc']:,.0f}")
k2.metric("LCOE", f"${econ['lcoe']:.4f}/kWh")
k3.metric("Renewable Fraction", f"{res['renewable_fraction']*100:.1f}%")
k4.metric("Unmet Load", f"{res['unmet_kwh']:,.0f} kWh/yr")
k5.metric("Fuel Consumed", f"{res['fuel_L']:,.0f} L/yr")

st.markdown("---")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_energy, tab_monthly, tab_soc, tab_econ, tab_sens = st.tabs(
    ["⚡ Energy Mix", "📅 Monthly Balance", "🔋 Battery SOC",
     "💰 Economics", "📊 Sensitivity"]
)

# ─── Tab 1: Energy Mix ────────────────────────────────────────────────────────
with tab_energy:
    st.subheader("Annual Energy Production & Consumption")

    prod_labels, prod_vals = [], []
    for label, val in [
        ("Solar PV", res["solar"]), ("Wind", res["wind"]),
        ("Generator", res["gen"]), ("Grid Import", res["grid_buy"]),
    ]:
        if val > 0:
            prod_labels.append(label)
            prod_vals.append(val)

    col_a, col_b = st.columns(2)

    with col_a:
        fig_pie = go.Figure(go.Pie(
            labels=prod_labels, values=prod_vals, hole=0.4,
            marker_colors=px.colors.qualitative.Set2,
        ))
        fig_pie.update_layout(title="Energy Sources (kWh/yr)", height=380)
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_b:
        flow_labels = ["Total Generation", "Load Served", "Battery Charging",
                       "Curtailed", "Unmet Load"]
        flow_vals = [
            res["solar"] + res["wind"] + res["gen"] + res["grid_buy"],
            res["load"] - res["unmet_kwh"],
            max(0, res["batt_delta"].clip(min=0).sum()) if "batt_delta" in res else 0,
            res["curtailed_kwh"],
            res["unmet_kwh"],
        ]
        fig_bar = go.Figure(go.Bar(
            x=flow_vals, y=flow_labels, orientation="h",
            marker_color=px.colors.qualitative.Pastel,
        ))
        fig_bar.update_layout(title="Energy Flows (kWh/yr)", height=380,
                              xaxis_title="kWh/year")
        st.plotly_chart(fig_bar, use_container_width=True)

    # Hourly sample (first week)
    st.subheader("Hourly Dispatch — First 7 Days")
    hrs = np.arange(168)
    fig_week = go.Figure()
    if res["solar"] > 0:
        fig_week.add_trace(go.Scatter(x=hrs, y=res["solar_out"][:168],
                                      name="Solar", fill="tozeroy",
                                      line=dict(color="gold")))
    if res["wind"] > 0:
        fig_week.add_trace(go.Scatter(x=hrs, y=res["wind_out"][:168],
                                      name="Wind", fill="tozeroy",
                                      line=dict(color="skyblue")))
    if res["gen"] > 0:
        fig_week.add_trace(go.Scatter(x=hrs, y=res["gen_output"][:168],
                                      name="Generator", fill="tozeroy",
                                      line=dict(color="salmon")))
    fig_week.add_trace(go.Scatter(x=hrs, y=load_arr[:168],
                                  name="Load", line=dict(color="black", dash="dot")))
    fig_week.update_layout(xaxis_title="Hour", yaxis_title="Power (kW)", height=350)
    st.plotly_chart(fig_week, use_container_width=True)

# ─── Tab 2: Monthly Balance ───────────────────────────────────────────────────
with tab_monthly:
    st.subheader("Monthly Energy Balance")

    month_data = {"Month": MONTHS}
    solar_m, wind_m, gen_m, load_m, unmet_m, grid_buy_m = (
        [] for _ in range(6))

    h = 0
    for mi, days in enumerate(DAYS_PER_MONTH):
        hrs = days * 24
        sl = slice(h, h + hrs)
        solar_m.append(res["solar_out"][sl].sum())
        wind_m.append(res["wind_out"][sl].sum())
        gen_m.append(res["gen_output"][sl].sum())
        load_m.append(load_arr[sl].sum())
        unmet_m.append(res["unmet"][sl].sum())   # unmet is the hourly array
        grid_buy_m.append(res["grid_buy_arr"][sl].sum())
        h += hrs

    fig_month = go.Figure()
    colors = dict(Solar="gold", Wind="skyblue", Generator="salmon",
                  Grid="plum")
    for label, vals, col in [
        ("Solar", solar_m, "gold"), ("Wind", wind_m, "skyblue"),
        ("Generator", gen_m, "salmon"), ("Grid Import", grid_buy_m, "plum"),
    ]:
        if sum(vals) > 0:
            fig_month.add_trace(go.Bar(name=label, x=MONTHS, y=vals,
                                       marker_color=col))
    fig_month.add_trace(go.Scatter(x=MONTHS, y=load_m, name="Load",
                                   mode="lines+markers",
                                   line=dict(color="black", dash="dot")))
    fig_month.update_layout(barmode="stack", yaxis_title="Energy (kWh)",
                             height=400)
    st.plotly_chart(fig_month, use_container_width=True)

    df_month = pd.DataFrame({
        "Month": MONTHS,
        "Solar (kWh)": [f"{v:,.0f}" for v in solar_m],
        "Wind (kWh)": [f"{v:,.0f}" for v in wind_m],
        "Generator (kWh)": [f"{v:,.0f}" for v in gen_m],
        "Grid Buy (kWh)": [f"{v:,.0f}" for v in grid_buy_m],
        "Load (kWh)": [f"{v:,.0f}" for v in load_m],
        "Unmet (kWh)": [f"{v:,.0f}" for v in unmet_m],
    })
    st.dataframe(df_month, use_container_width=True, hide_index=True)

# ─── Tab 3: Battery SOC ───────────────────────────────────────────────────────
with tab_soc:
    st.subheader("Battery State of Charge")

    if bat_kwh > 0:
        soc_pct = res["soc"] / bat_kwh * 100

        # Full-year SOC heatmap (daily averages)
        daily_soc = soc_pct.reshape(-1, 24).mean(axis=1)[:365]
        fig_heat = px.imshow(
            daily_soc.reshape(52, 7)[:, :],
            labels=dict(x="Day of Week", y="Week", color="SOC (%)"),
            color_continuous_scale="RdYlGn",
            zmin=0, zmax=100,
            title="Daily Average Battery SOC (%) — Full Year",
        )
        fig_heat.update_layout(height=350)
        st.plotly_chart(fig_heat, use_container_width=True)

        # First 7-day detail
        fig_soc = go.Figure()
        fig_soc.add_trace(go.Scatter(
            x=np.arange(168), y=soc_pct[:168], name="SOC (%)",
            fill="tozeroy", line=dict(color="mediumseagreen"),
        ))
        fig_soc.add_hline(y=bat_min_soc * 100, line_dash="dash",
                          line_color="red", annotation_text="Min SOC")
        fig_soc.update_layout(xaxis_title="Hour",
                               yaxis_title="State of Charge (%)",
                               yaxis_range=[0, 105], height=300)
        st.subheader("First 7 Days — Hourly SOC")
        st.plotly_chart(fig_soc, use_container_width=True)

        m1, m2, m3 = st.columns(3)
        m1.metric("Mean SOC", f"{soc_pct.mean():.1f}%")
        m2.metric("Min SOC", f"{soc_pct.min():.1f}%")
        m3.metric("Max SOC", f"{soc_pct.max():.1f}%")
    else:
        st.info("No battery configured in this system.")

# ─── Tab 4: Economics ─────────────────────────────────────────────────────────
with tab_econ:
    st.subheader("Economic Analysis")

    ec1, ec2 = st.columns(2)

    with ec1:
        # NPC breakdown
        npc_items = {
            "Solar PV": econ["sol_npc"],
            "Wind Turbine": econ["wnd_npc"],
            "Battery": econ["bat_npc"],
            "Generator": econ["gen_npc"],
            "Fuel": econ["fuel_pv"],
            "Grid": econ["grid_pv"],
        }
        npc_items = {k: v for k, v in npc_items.items() if v > 0}
        fig_npc = go.Figure(go.Pie(
            labels=list(npc_items.keys()),
            values=list(npc_items.values()),
            hole=0.35,
            marker_colors=px.colors.qualitative.Pastel1,
        ))
        fig_npc.update_layout(title="Net Present Cost Breakdown", height=380)
        st.plotly_chart(fig_npc, use_container_width=True)

    with ec2:
        # Capital cost waterfall
        cap = {k: v for k, v in econ["capital"].items() if v > 0}
        fig_cap = go.Figure(go.Bar(
            x=list(cap.keys()), y=list(cap.values()),
            marker_color=px.colors.qualitative.Set2,
        ))
        fig_cap.update_layout(title="Initial Capital Costs ($)", height=380,
                               yaxis_title="USD")
        st.plotly_chart(fig_cap, use_container_width=True)

    # Summary table
    r = cfg["discount_rate"]
    n = cfg["project_lifetime"]
    st.subheader("Summary")
    summary_df = pd.DataFrame({
        "Metric": [
            "Net Present Cost (NPC)",
            "Levelised Cost of Energy (LCOE)",
            "Initial Capital Cost",
            "Annual O&M (approx.)",
            "Annual Fuel Cost",
            "Renewable Fraction",
            "Annual Unmet Load",
            "Capacity Factor — Solar",
            "Capacity Factor — Wind",
        ],
        "Value": [
            f"${econ['npc']:,.0f}",
            f"${econ['lcoe']:.4f} / kWh",
            f"${sum(econ['capital'].values()):,.0f}",
            f"${(sol_om*sol_kw + wnd_om*wnd_kw + bat_om*bat_kwh + gen_om*gen_kw):,.0f} / yr",
            f"${econ['annual_fuel']:,.0f} / yr",
            f"{res['renewable_fraction']*100:.1f}%",
            f"{res['unmet_kwh']:,.0f} kWh/yr",
            f"{res['solar']/(max(sol_kw,1)*HOURS_PER_YEAR)*100:.1f}%" if sol_kw > 0 else "N/A",
            f"{res['wind']/(max(wnd_kw,1)*HOURS_PER_YEAR)*100:.1f}%" if wnd_kw > 0 else "N/A",
        ],
    })
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

# ─── Tab 5: Sensitivity ───────────────────────────────────────────────────────
with tab_sens:
    st.subheader("Sensitivity Analysis")
    st.caption("See how LCOE changes as you vary one parameter at a time "
               "(all other values held at their configured levels).")

    sens_param = st.selectbox(
        "Parameter to vary",
        ["Fuel Price ($/L)", "Solar Capacity (kW)", "Battery Capacity (kWh)",
         "Discount Rate (%)", "Peak Load (kW)"],
    )

    @st.cache_data(show_spinner=False)
    def run_sens(param_name: str, base_cfg: dict, base_res: dict) -> pd.DataFrame:
        """Sweep one parameter and collect LCOE."""
        ranges = {
            "Fuel Price ($/L)": np.linspace(0.5, 4.0, 20),
            "Solar Capacity (kW)": np.linspace(0, 200, 20),
            "Battery Capacity (kWh)": np.linspace(0, 500, 20),
            "Discount Rate (%)": np.linspace(2, 20, 20),
            "Peak Load (kW)": np.linspace(10, 200, 20),
        }
        vals = ranges[param_name]
        lcoes = []

        _load = gen_load(base_cfg["peak_load"], base_cfg["load_type"])
        _irr = gen_solar_irr(base_cfg["sol_irr"])
        _spd = gen_wind_speed(base_cfg["wind_spd"])
        _wind = wind_power(_spd, base_cfg["wind_kw"])

        for v in vals:
            _cfg = dict(base_cfg)
            _load_local = _load
            _wind_local = _wind
            _irr_local = _irr

            if param_name == "Fuel Price ($/L)":
                _cfg["fuel_price"] = v
            elif param_name == "Solar Capacity (kW)":
                _cfg["solar_kw"] = v
            elif param_name == "Battery Capacity (kWh)":
                _cfg["batt_kwh"] = v
            elif param_name == "Discount Rate (%)":
                _cfg["discount_rate"] = v / 100
            elif param_name == "Peak Load (kW)":
                _load_local = gen_load(v, base_cfg["load_type"])

            _r = simulate(
                _load_local,
                _cfg["solar_kw"], _irr_local, _cfg["sol_derate"],
                _wind_local,
                _cfg["batt_kwh"], _cfg["batt_eff"], _cfg["batt_min_soc"],
                _cfg["gen_kw"], _cfg["fuel_f1"], _cfg["fuel_f0"], _cfg["fuel_price"],
                _cfg["grid_en"], _cfg["grid_buy"], _cfg["grid_sell"],
            )
            _e = economics(_r, _cfg)
            lcoes.append(_e["lcoe"])

        return pd.DataFrame({param_name: vals, "LCOE ($/kWh)": lcoes})

    base_cfg_sens = dict(
        project_lifetime=proj_life, discount_rate=discount,
        solar_kw=sol_kw, solar_cost=sol_cost, solar_om=sol_om, solar_life=sol_life,
        wind_kw=wnd_kw, wind_cost=wnd_cost, wind_om=wnd_om, wind_life=wnd_life,
        batt_kwh=bat_kwh, batt_cost=bat_cost, batt_om=bat_om, batt_life=bat_life,
        gen_kw=gen_kw, gen_cost=gen_cost, gen_om=gen_om, gen_life=gen_life,
        fuel_price=fuel_price, grid_en=grid_en, grid_buy=grid_buy, grid_sell=grid_sell,
        peak_load=peak_load, load_type=load_type,
        sol_irr=tuple(sol_irr), wind_spd=tuple(wind_spd),
        sol_derate=sol_derate, batt_eff=bat_eff, batt_min_soc=bat_min_soc,
        fuel_f0=fuel_f0, fuel_f1=fuel_f1,
    )

    df_sens = run_sens(sens_param, base_cfg_sens, res)
    fig_sens = px.line(df_sens, x=sens_param, y="LCOE ($/kWh)",
                       markers=True, title=f"LCOE vs {sens_param}")
    fig_sens.update_layout(height=400)
    st.plotly_chart(fig_sens, use_container_width=True)

    st.dataframe(df_sens.style.format({
        sens_param: "{:.2f}", "LCOE ($/kWh)": "{:.4f}"
    }), use_container_width=True, hide_index=True)

# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "Hybrid Microgrid Simulator — educational tool inspired by HOMER Pro. "
    "Results are indicative; consult a professional engineer for real projects."
)
