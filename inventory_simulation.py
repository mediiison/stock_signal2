import argparse
import math

import numpy as np
import pandas as pd


def draw_demand(rng, mean, std, size):
    var = std ** 2
    if var <= mean:
        return rng.poisson(mean, size)
    n = mean ** 2 / (var - mean)
    p = n / (n + mean)
    return rng.negative_binomial(n, p, size)


def draw_lead(rng, mean, std, size):
    lead = np.rint(rng.normal(mean, std, size)).astype(int)
    return np.clip(lead, 1, None)


def simulate(reorder_days, cycle_days, args, rng):
    runs = args.runs
    horizon = args.horizon
    max_lead = int(args.lead_mean + 6 * args.lead_std + 2)

    reorder_point = reorder_days * args.mean_demand
    order_up_to = (reorder_days + cycle_days) * args.mean_demand

    arrivals = np.zeros((runs, horizon + max_lead + 1))
    on_hand = np.full(runs, order_up_to, dtype=float)
    on_order = np.zeros(runs)
    idx = np.arange(runs)

    total_demand = np.zeros(runs)
    total_lost = np.zeros(runs)
    stockout_days = np.zeros(runs)
    inventory_sum = np.zeros(runs)
    orders = np.zeros(runs)

    for t in range(horizon):
        arrived = arrivals[:, t]
        on_hand += arrived
        on_order -= arrived

        demand = draw_demand(rng, args.mean_demand, args.demand_std, runs)
        sold = np.minimum(on_hand, demand)
        lost = demand - sold
        on_hand -= sold

        total_demand += demand
        total_lost += lost
        stockout_days += lost > 0
        inventory_sum += on_hand

        position = on_hand + on_order
        trigger = position <= reorder_point
        qty = np.where(trigger, order_up_to - position, 0.0)
        lead = np.minimum(draw_lead(rng, args.lead_mean, args.lead_std, runs), max_lead)

        rows = idx[trigger]
        arrivals[rows, t + lead[trigger]] += qty[trigger]
        on_order += qty
        orders += trigger

    avg_inventory = inventory_sum / horizon
    cost = (
        args.holding_cost * inventory_sum
        + args.stockout_cost * total_lost
        + args.order_cost * orders
    )
    fill_rate = 1 - total_lost.sum() / total_demand.sum()

    return {
        "reorder_days": reorder_days,
        "reorder_point": round(reorder_point, 1),
        "order_up_to": round(order_up_to, 1),
        "fill_rate": round(fill_rate, 4),
        "stockout_day_share": round(stockout_days.mean() / horizon, 4),
        "avg_inventory": round(avg_inventory.mean(), 1),
        "orders_per_run": round(orders.mean(), 2),
        "avg_cost": round(cost.mean(), 1),
        "cost_p95": round(np.percentile(cost, 95), 1),
    }


def analytic_reorder_point(args, z):
    mu = args.mean_demand
    lead = args.lead_mean
    sigma_d = args.demand_std
    sigma_l = args.lead_std
    variance = lead * sigma_d ** 2 + (mu ** 2) * sigma_l ** 2
    return mu * lead + z * math.sqrt(variance)


def z_for_service(level):
    from statistics import NormalDist
    return NormalDist().inv_cdf(level)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mean-demand", type=float, default=20.0)
    p.add_argument("--demand-std", type=float, default=12.0)
    p.add_argument("--lead-mean", type=float, default=10.0)
    p.add_argument("--lead-std", type=float, default=3.0)
    p.add_argument("--horizon", type=int, default=180)
    p.add_argument("--runs", type=int, default=2000)
    p.add_argument("--cycle-days", type=float, default=14.0)
    p.add_argument("--holding-cost", type=float, default=0.5)
    p.add_argument("--stockout-cost", type=float, default=30.0)
    p.add_argument("--order-cost", type=float, default=200.0)
    p.add_argument("--target-service", type=float, default=0.95)
    p.add_argument("--min-days", type=int, default=3)
    p.add_argument("--max-days", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="inventory_simulation_results.csv")
    p.add_argument("--plot", default="")
    return p.parse_args()


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    rows = [
        simulate(days, args.cycle_days, args, rng)
        for days in range(args.min_days, args.max_days + 1)
    ]
    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)

    feasible = df[df["fill_rate"] >= args.target_service]
    best_cost = df.loc[df["avg_cost"].idxmin()]
    print(df.to_string(index=False))
    print()
    print(
        f"Min cost: reorder at {best_cost['reorder_days']:.0f} days of cover, "
        f"fill rate {best_cost['fill_rate']:.3f}, cost {best_cost['avg_cost']:.0f}"
    )
    if feasible.empty:
        print(f"No policy reaches fill rate {args.target_service}")
    else:
        best = feasible.loc[feasible["avg_cost"].idxmin()]
        print(
            f"Cheapest with fill rate >= {args.target_service}: reorder at "
            f"{best['reorder_days']:.0f} days of cover ({best['reorder_point']:.0f} units), "
            f"fill rate {best['fill_rate']:.3f}, cost {best['avg_cost']:.0f}"
        )

    z = z_for_service(args.target_service)
    analytic = analytic_reorder_point(args, z)
    print(
        f"Analytic reorder point for cycle service level {args.target_service}: "
        f"{analytic:.0f} units ({analytic / args.mean_demand:.1f} days of cover)"
    )

    if args.plot:
        import matplotlib.pyplot as plt

        fig, ax1 = plt.subplots(figsize=(9, 5))
        ax1.plot(df["reorder_days"], df["avg_cost"], color="tab:blue", label="Average cost")
        ax1.set_xlabel("Reorder point, days of cover")
        ax1.set_ylabel("Average cost")
        ax2 = ax1.twinx()
        ax2.plot(df["reorder_days"], df["fill_rate"], color="tab:orange", label="Fill rate")
        ax2.axhline(args.target_service, color="gray", linestyle="--")
        ax2.set_ylabel("Fill rate")
        fig.tight_layout()
        fig.savefig(args.plot, dpi=150)
        print(f"Saved plot: {args.plot}")


if __name__ == "__main__":
    main()
