"""Feature engineering shared by the notebook (kept in one module so it is testable).

Everything is computed using ONLY data up to the snapshot date (CUTOFF). The churn
label looks at the following 90 days. This separation prevents data leakage.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CUTOFF = pd.Timestamp("2026-06-30")
HORIZON_DAYS = 90


def load():
    cust = pd.read_csv(ROOT / "data" / "customers.csv")
    orders = pd.read_csv(ROOT / "data" / "orders.csv", parse_dates=["order_date"])
    comp = pd.read_csv(ROOT / "data" / "complaints.csv", parse_dates=["complaint_date"])
    return cust, orders, comp


def build_features(cust, orders, comp, cutoff=CUTOFF):
    past = orders[orders.order_date <= cutoff]
    d = lambda days: past[past.order_date > cutoff - pd.Timedelta(days=days)]
    last365, last90, prior270 = d(365), d(90), past[(past.order_date > cutoff - pd.Timedelta(days=365))
                                                    & (past.order_date <= cutoff - pd.Timedelta(days=90))]
    g = past.groupby("customer_id")
    f = pd.DataFrame({
        "recency_days": (cutoff - g.order_date.max()).dt.days,
        "first_order": g.order_date.min(),
    })
    f["frequency_365d"] = last365.groupby("customer_id").size()
    f["monetary_365d"] = last365.groupby("customer_id").net_value_kes.sum()
    f["orders_90d"] = last90.groupby("customer_id").size()
    f["orders_prior_270d"] = prior270.groupby("customer_id").size()
    f = f.fillna({"frequency_365d": 0, "monetary_365d": 0, "orders_90d": 0, "orders_prior_270d": 0})
    # trend: last-90-day order rate vs the average 90-day rate in the prior 270 days (1 = steady)
    f["order_trend"] = (f.orders_90d + 1) / (f.orders_prior_270d / 3 + 1)
    l = last365.groupby("customer_id")
    f["avg_order_value"] = l.net_value_kes.mean()
    f["avg_skus_per_order"] = l.n_skus.mean()
    f["avg_discount_pct"] = l.discount_pct.mean()
    f["late_delivery_rate"] = l.delivered_late.mean()
    f["stockout_rate"] = l.had_stockout.mean()
    f["return_rate"] = l.returned_value_kes.sum() / l.net_value_kes.sum()
    c = comp[(comp.complaint_date <= cutoff) & (comp.complaint_date > cutoff - pd.Timedelta(days=180))]
    f["complaints_180d"] = c.groupby("customer_id").size()
    f["complaints_180d"] = f.complaints_180d.fillna(0)
    f["tenure_months"] = ((cutoff - f.first_order).dt.days / 30.44).round(1)
    f = f.drop(columns="first_order").reset_index().merge(cust, on="customer_id", how="left")
    f["has_merchandiser"] = f.has_merchandiser.astype(int)
    return f


def churn_label(orders, cutoff=CUTOFF, horizon=HORIZON_DAYS):
    fut = orders[(orders.order_date > cutoff) & (orders.order_date <= cutoff + pd.Timedelta(days=horizon))]
    return lambda ids: (~pd.Series(ids).isin(set(fut.customer_id))).astype(int).values


def is_test(customer_id: pd.Series) -> pd.Series:
    """Deterministic 25% test split (customer number divisible by 4) - identical in Python and R."""
    return customer_id.str[1:].astype(int) % 4 == 0
