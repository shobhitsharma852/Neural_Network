# Neural_Network & Hybrid Microgrid Simulator

This repository contains two interactive Streamlit applications.

---

## 1. ⚡ Hybrid Microgrid Simulator *(HOMER Pro–style)* — `homer_simulator.py`

An educational tool that mirrors the core workflow of **HOMER Pro** software for designing and analysing hybrid renewable power systems (microgrids).

### Features

| Feature | Description |
|---|---|
| **Component configuration** | Solar PV, Wind Turbine, Battery Storage, Diesel Generator, Grid connection |
| **Resource data** | Monthly solar irradiance (kWh/m²/day) and wind speed (m/s) inputs |
| **Load profiles** | Residential, Commercial, and Industrial diurnal load shapes |
| **Simulation** | Hour-by-hour (8 760-hour) load-following dispatch over a full year |
| **Economic analysis** | Net Present Cost (NPC), Levelised Cost of Energy (LCOE), capital & O&M costs, fuel cost, replacement & salvage values |
| **Visualisations** | Energy mix pie chart, monthly stacked bar chart, battery SOC heatmap, weekly dispatch plot, cost breakdown |
| **Sensitivity analysis** | Sweep any key parameter (fuel price, PV capacity, battery size, discount rate, load) and see its effect on LCOE |

### How to run

```bash
pip install -r requirements.txt
streamlit run homer_simulator.py
```

### Workflow

1. Configure system components and project settings in the **sidebar**.
2. Adjust monthly **resource data** (solar irradiance, wind speed).
3. Click **Run Simulation** to run the 8 760-hour dispatch engine.
4. Explore results across five tabs:
   - **Energy Mix** — annual production pie chart + first-week dispatch
   - **Monthly Balance** — stacked monthly energy bar chart + table
   - **Battery SOC** — full-year SOC heatmap + 7-day detail
   - **Economics** — NPC breakdown, capital costs, summary table
   - **Sensitivity** — LCOE vs any parameter sweep

---

## 2. Trip Neural Network — `Neural.py`

An interactive neural network simulator that decides "Should Shobhit go on a Goa trip?" using weighted inputs (Weather, Budget, Free Time, Mood) and a configurable decision threshold.

### How to run

```bash
streamlit run Neural.py
```

---

## Dependencies

```
streamlit
graphviz
numpy
pandas
plotly
```
