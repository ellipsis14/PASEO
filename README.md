# PASEO
IQM-Quantum-Hackathon
# Wind + Battery Revenue Maximization (24h) — Problem & Models

This repository models a **wind farm + battery** portfolio over a **one-day horizon (24 hours)** to maximize revenue under **uncertain wind production** (13 equally likely scenarios) and known **hourly market prices**.

---

## 1) Problem statement

### Goal
Choose an **hour-by-hour battery operation plan** (charge/discharge/idle) to maximize the **expected** joint revenue of:
- selling wind generation to the market, and
- battery energy arbitrage (buy low / sell high),

subject to battery physics and operating limits.

### Data (`input_data.csv`)
Columns:
- `hour`: 1..24
- `scenario_1` … `scenario_13`: wind production forecasts (MWh) for each scenario (equally probable)
- `price`: market price (€/MWh), same for all scenarios

### Battery parameters
- Energy capacity: **16 MWh**
- Max charge power: **5 MW** (≡ 5 MWh/h)
- Max discharge power: **4 MW** (≡ 4 MWh/h)
- Charge efficiency: **ηᶜʰ = 0.8**
- Discharge efficiency: **ηᵈⁱˢ = 1.0**
- Max cycles/day: **2** (modeled via a discharge-throughput proxy)
- Initial SOC: **0 MWh**; Final SOC: **0 MWh**
- Each hour: **charge OR discharge OR idle** (no simultaneous charge/discharge)

---

## 2) Mathematical formulation — Version A (No Netting)

### Interpretation
The battery **buys and sells all energy directly from/to the grid at market price**.  
Wind is sold to the market independently. There is **no physical coupling** between wind and battery.

### Indices / Sets
- Hours:  \( t \in \{1,\dots,24\} \)
- Scenarios: \( s \in \{1,\dots,13\} \), with \( \pi_s = 1/13 \)

### Parameters
- \( p_t \) : price (€/MWh)
- \( w_{t,s} \) : wind energy (MWh) in scenario \(s\)

### Decision variables (day-ahead, scenario-independent)
For each hour \(t\):
- \( P^{ch}_t \ge 0 \): charge energy (MWh)
- \( P^{dis}_t \ge 0 \): discharge energy (MWh)
- \( e_t \): SOC at end of hour (MWh)
- \( y_t \in \{0,1\} \): mode (1=charge, 0=discharge)

### Objective (maximize expected revenue)
\[
\max \left[
\sum_{s} \pi_s \sum_{t} p_t w_{t,s}
\; + \;
\sum_{t} p_t \left(P^{dis}_t - P^{ch}_t\right)
\right]
\]

> Under “no netting”, the wind term is **constant** w.r.t. battery decisions; the battery schedule is driven by prices + battery constraints.

### Constraints
SOC dynamics (1h steps):
\[
e_t = e_{t-1} + \eta^{ch} P^{ch}_t - \frac{1}{\eta^{dis}} P^{dis}_t \quad \forall t
\]

Capacity:
\[
0 \le e_t \le 16 \quad \forall t
\]

Power limits:
\[
0 \le P^{ch}_t \le 5, \quad 0 \le P^{dis}_t \le 4 \quad \forall t
\]

Mutual exclusivity:
\[
P^{ch}_t \le 5 y_t, \quad P^{dis}_t \le 4(1-y_t), \quad y_t\in\{0,1\} \quad \forall t
\]

Initial/final SOC:
\[
e_0 = 0, \quad e_{24} = 0
\]

Cycle proxy (optional but commonly used):
\[
\sum_t P^{dis}_t \le 2\cdot 16 = 32
\]

---

## 3) Mathematical formulation — Version B (Physically Coupled Wind + Battery)

### Interpretation
Wind can be **sold**, **used to charge the battery**, or **curtailed** (scenario-dependent).  
The battery’s **charge/discharge schedule** is decided **day-ahead** (same across scenarios).  
After uncertainty realizes, wind/grid allocations are adjusted (recourse).

### Additional recourse variables (scenario-dependent)
For each \((t,s)\):
- \( x_{t,s} \ge 0 \): wind sold (MWh)
- \( z_{t,s} \ge 0 \): wind sent to charger (MWh)
- \( g_{t,s} \ge 0 \): grid energy sent to charger (MWh)
- \( c_{t,s} \ge 0 \): curtailed wind (MWh)

### First-stage variables (day-ahead)
For each \(t\):
- \( U_t \ge 0 \): total charging input energy to charger (MWh)
- \( D_t \ge 0 \): discharge energy sold (MWh)
- \( e_t \): SOC (MWh)
- \( y_t \in \{0,1\} \): mode

### Objective (maximize expected revenue)
\[
\max \sum_s \pi_s \sum_t p_t\,\big(x_{t,s} + D_t - g_{t,s}\big)
\]

### Constraints
Wind balance:
\[
x_{t,s} + z_{t,s} + c_{t,s} = w_{t,s} \quad \forall t,s
\]

Charger supply coupling:
\[
z_{t,s} + g_{t,s} = U_t \quad \forall t,s
\]

(Optional) grid charging cap (makes wind uncertainty impactful):
\[
0 \le g_{t,s} \le G^{max}_t \quad \forall t,s
\]
- **Wind-only charging:** set \(G^{max}_t = 0\)

Battery SOC dynamics (scenario-independent because \(U_t, D_t\) are day-ahead):
\[
e_t = e_{t-1} + \eta^{ch} U_t - \frac{1}{\eta^{dis}} D_t \quad \forall t
\]

Battery bounds:
\[
0\le e_t\le 16,\quad 0\le U_t\le 5,\quad 0\le D_t\le 4 \quad \forall t
\]

Mutual exclusivity:
\[
U_t \le 5 y_t, \quad D_t \le 4(1-y_t), \quad y_t\in\{0,1\} \quad \forall t
\]

Initial/final SOC:
\[
e_0=0,\quad e_{24}=0
\]

Cycle proxy:
\[
\sum_t D_t \le 32
\]

---

## 4) Notes
- **Version A** is the simplest and often sufficient for pure arbitrage.
- **Version B** is needed when physical coupling matters (e.g., wind-only charging, limited grid charging, or export/import constraints), so wind uncertainty affects feasibility and optimal strategy.

