"""Generate synthetic B2B customer (retail outlet) data for segmentation and churn analysis.

Context: an FMCG manufacturer/distributor in Kenya selling to ~4,000 outlets:
dukas, supermarkets, wholesalers, hotels/restaurants (HoReCa) and petrol-station shops.

How the data is built (so there is real structure to discover):
  * Each outlet has a channel, region, size (order rate), price sensitivity,
    and a service experience (late deliveries, stock-outs, complaints).
  * Monthly orders follow a Poisson process from Jan 2025 to Sep 2026.
  * Each outlet has a monthly churn hazard driven by service failures, price
    sensitivity, low engagement, competition by region and channel. Once an outlet
    "decides" to leave, its order rate decays over 2-3 months, then stops. This
    mimics real B2B churn: customers fade before they go silent.

Outputs:
  data/customers.csv   static attributes, one row per outlet
  data/orders.csv      one row per order
  data/complaints.csv  one row per logged complaint

All data is randomly generated. No real company or customer data is included.
Usage: python data/generate_data.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

RNG = np.random.default_rng(2026)
OUT = Path(__file__).parent
N = 4000
MONTHS = pd.period_range("2025-01", "2026-09", freq="M")

CHANNELS = {  # share, base orders/month, base order value KES, base monthly churn hazard
    "Duka":          (.48, 2.2,  9_000, .020),
    "Supermarket":   (.12, 5.5, 85_000, .008),
    "Wholesaler":    (.10, 4.0, 160_000, .012),
    "HoReCa":        (.18, 3.0, 28_000, .018),
    "Petrol Station":(.12, 2.5, 15_000, .015),
}
REGIONS = {"Nairobi": (.32, 1.25), "Central": (.16, .9), "Rift Valley": (.17, .95),
           "Western": (.12, 1.0), "Nyanza": (.11, 1.05), "Coast": (.12, 1.15)}  # share, competition

ch_names = list(CHANNELS)
rg_names = list(REGIONS)
cust = pd.DataFrame({
    "customer_id": [f"C{i:05d}" for i in range(1, N + 1)],
    "channel": RNG.choice(ch_names, N, p=[CHANNELS[c][0] for c in ch_names]),
    "region": RNG.choice(rg_names, N, p=[REGIONS[r][0] for r in rg_names]),
})
# Onboarding: most outlets exist before 2025, some join during the window
onboard = np.where(RNG.random(N) < .75, RNG.integers(-36, 0, N), RNG.integers(0, 15, N))
cust["onboard_month"] = [str(MONTHS[0] + int(m)) for m in onboard]
cust["size_factor"] = RNG.lognormal(0, .45, N)
cust["price_sensitivity"] = RNG.beta(2, 4, N)                 # 0..1
cust["service_quality"] = RNG.beta(6, 2, N)                    # 0..1, higher = better served
cust["has_merchandiser"] = RNG.random(N) < np.where(cust.channel.isin(["Supermarket", "Wholesaler"]), .85, .35)
cust["credit_terms_days"] = np.select([cust.channel.isin(["Supermarket", "Wholesaler"]), cust.channel == "HoReCa"],
                                      [30, 14], 0)

orders, complaints = [], []
churn_month = {}
for r in cust.itertuples():
    base_rate, base_val, base_h = CHANNELS[r.channel][1:]
    comp = REGIONS[r.region][1]
    start = max(0, onboard[r.Index])
    gone, fade = False, None
    for t in range(start, len(MONTHS)):
        m = MONTHS[t]
        late_p = .05 + .35 * (1 - r.service_quality)
        stockout_p = .03 + .25 * (1 - r.service_quality)
        hazard = (2.2 * base_h * comp * (1 + 2.2 * r.price_sensitivity) * (1.9 - 1.6 * r.service_quality)
                  * (0.6 if r.has_merchandiser else 1.15) * (1.6 if t - onboard[r.Index] < 6 else 1))
        if fade is None and RNG.random() < hazard:
            fade = t
        rate = base_rate * r.size_factor
        if fade is not None:
            k = t - fade
            rate *= [.6, .3, .1][k] if k < 3 else 0
        if fade is not None and t - fade >= 3:
            gone = True
        if gone:
            continue
        for _ in range(RNG.poisson(rate)):
            day = int(RNG.integers(1, 28))
            n_sku = max(1, int(RNG.poisson(6 if r.channel in ("Supermarket", "Wholesaler") else 3)))
            disc = round(float(np.clip(RNG.normal(.02 + .08 * r.price_sensitivity, .015), 0, .2)), 3)
            gross = base_val * r.size_factor * RNG.lognormal(0, .35)
            late = RNG.random() < late_p
            stockout = RNG.random() < stockout_p
            returned = gross * RNG.uniform(.02, .15) if RNG.random() < .06 + .1 * (1 - r.service_quality) else 0
            orders.append((r.customer_id, f"{m.year}-{m.month:02d}-{day:02d}", round(gross * (1 - disc), 0),
                           n_sku, disc, int(late), int(stockout), round(returned, 0)))
            if (late or stockout) and RNG.random() < .25 + .4 * (1 - r.service_quality):
                complaints.append((r.customer_id, f"{m.year}-{m.month:02d}-{min(day + 2, 28):02d}",
                                   "Late delivery" if late else "Stock-out"))
        if RNG.random() < .01:
            complaints.append((r.customer_id, f"{m.year}-{m.month:02d}-15", RNG.choice(["Pricing", "Product quality"])))
    churn_month[r.customer_id] = None if fade is None else str(MONTHS[fade])

orders = pd.DataFrame(orders, columns=["customer_id", "order_date", "net_value_kes", "n_skus",
                                       "discount_pct", "delivered_late", "had_stockout", "returned_value_kes"])
orders.insert(0, "order_id", [f"O{i:07d}" for i in range(1, len(orders) + 1)])
complaints = pd.DataFrame(complaints, columns=["customer_id", "complaint_date", "complaint_type"])

# Hidden drivers are NOT exported (as in real life) - only observable attributes are
cust[["customer_id", "channel", "region", "onboard_month", "has_merchandiser", "credit_terms_days"]] \
    .to_csv(OUT / "customers.csv", index=False)
orders.sort_values(["customer_id", "order_date"]).to_csv(OUT / "orders.csv", index=False)
complaints.sort_values(["customer_id", "complaint_date"]).to_csv(OUT / "complaints.csv", index=False)
print(f"{len(cust):,} customers | {len(orders):,} orders | {len(complaints):,} complaints")
