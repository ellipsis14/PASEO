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




