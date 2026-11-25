# -*- coding: utf-8 -*-
"""Дослідження моделі: вплив окремих змінних і повний факторний
експеримент 2^4 зі статистичною обробкою результатів.

Запуск:  python experiment.py
"""
import itertools
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from smo_model import simulate
from verification import analytic, run_series

RESULTS_DIR = "results"
ALPHA = 0.05            # рівень значущості

# ---------- повний факторний експеримент ----------
FACTORS = ["x1", "x2", "x3", "x4"]
LEVELS = {              # нижній і верхній рівні факторів
    "x1": (0.2, 0.3),   # lam1 - інтенсивність потоку 1
    "x2": (0.1, 0.2),   # lam2 - інтенсивність потоку 2
    "x3": (0.3, 0.4),   # mu   - інтенсивність обслуговування
    "x4": (3, 7),       # m    - місткість черги
}
T_END = 480             # тривалість одного досліду, хв (робочий день)
P_RUNS = 5              # кількість паралельних дослідів у кожній точці плану


def one_factor_study():
    """Зміна вхідних змінних по одній (тривалі прогони)."""
    variants = [(0.2, 0.1, 0.4, 5),
                (0.1, 0.1, 0.4, 5), (0.3, 0.1, 0.4, 5),
                (0.2, 0.05, 0.4, 5), (0.2, 0.2, 0.4, 5),
                (0.2, 0.1, 0.3, 5), (0.2, 0.1, 0.5, 5),
                (0.2, 0.1, 0.4, 3), (0.2, 0.1, 0.4, 8)]
    rows = []
    print("1. Зміна вхідних змінних по одній (прогони по 100000 хв)")
    print("-" * 78)
    print(f"{'№':>2}{'lam1':>7}{'lam2':>7}{'mu':>6}{'m':>4}"
          f"{'P відм.':>10}{'P відм.':>10}{'Черга':>10}{'Черга':>10}")
    print(f"{'':>26}{'модель':>10}{'аналіт.':>10}{'модель':>10}{'аналіт.':>10}")
    print("-" * 78)
    for i, (l1, l2, mu, m) in enumerate(variants, 1):
        d = run_series(l1, l2, mu, m, 100_000, 10)
        th = analytic(l1 + l2, mu, m)
        row = (i, l1, l2, mu, m, d["p_reject"].mean(), th["p_reject"],
               d["mean_queue"].mean(), th["mean_queue"])
        rows.append(row)
        print(f"{i:>2}{l1:>7}{l2:>7}{mu:>6}{m:>4}{row[5]:>10.4f}"
              f"{row[6]:>10.4f}{row[7]:>10.4f}{row[8]:>10.4f}")
    print("-" * 78)
    pd.DataFrame(rows, columns=["№", "lam1", "lam2", "mu", "m",
                                "P відмови (модель)", "P відмови (аналітична)",
                                "Черга (модель)", "Черга (аналітична)"]
                 ).to_csv(os.path.join(RESULTS_DIR, "one_factor.csv"),
                          index=False)


def cochran_critical(n, f, alpha):
    """Критичне значення критерію Кохрена для n дисперсій з f ступенями."""
    fq = stats.f.ppf(1 - alpha / n, f, (n - 1) * f)
    return fq / (fq + n - 1)


def factorial_experiment():
    n = 2 ** len(FACTORS)                               # кількість точок плану
    plan = np.array(list(itertools.product((-1, 1), repeat=4)))[:, ::-1]

    # ---------- проведення дослідів ----------
    y = np.zeros((n, P_RUNS))
    for j in range(n):
        values = [LEVELS[f][0] if plan[j, k] < 0 else LEVELS[f][1]
                  for k, f in enumerate(FACTORS)]
        for i in range(P_RUNS):
            res = simulate(values[0], values[1], values[2], values[3],
                           T_END, seed=100 * (j + 1) + i + 1)
            y[j, i] = res["p_served"]
    y_mean = y.mean(axis=1)
    y_var = y.var(axis=1, ddof=1)

    print()
    print(f"2. Повний факторний експеримент 2^4 "
          f"({P_RUNS} дослідів у кожній точці, по {T_END} хв)")
    print("-" * 78)
    head = "".join(f"{'y' + str(i + 1):>8}" for i in range(P_RUNS))
    print(f"{'№':>2}{'x1':>4}{'x2':>4}{'x3':>4}{'x4':>4}{head}"
          f"{'Середнє':>10}{'Дисперсія':>11}")
    print("-" * 78)
    for j in range(n):
        signs = "".join(f"{'+' if v > 0 else '-':>4}" for v in plan[j])
        vals = "".join(f"{v:>8.4f}" for v in y[j])
        print(f"{j + 1:>2}{signs}{vals}{y_mean[j]:>10.4f}{y_var[j]:>11.6f}")
    print("-" * 78)

    # ---------- критерій Кохрена ----------
    g = y_var.max() / y_var.sum()
    g_kr = cochran_critical(n, P_RUNS - 1, ALPHA)
    d_repr = y_var.mean()                       # дисперсія відтворюваності
    f_repr = n * (P_RUNS - 1)
    print(f"Критерій Кохрена: G = {g:.4f}, G кр = {g_kr:.4f} -> "
          f"{'дисперсії однорідні' if g < g_kr else 'дисперсії неоднорідні'}")
    print(f"Дисперсія відтворюваності D = {d_repr:.6f}")

    # ---------- коефіцієнти рівняння регресії ----------
    terms = [()]
    for r in range(1, 5):
        terms += list(itertools.combinations(range(4), r))
    names = ["b0"] + ["b" + "".join(str(k + 1) for k in t) for t in terms[1:]]
    columns = np.array([np.prod(plan[:, list(t)], axis=1) if t
                        else np.ones(n) for t in terms]).T
    b = columns.T @ y_mean / n
    s_b = np.sqrt(d_repr / (n * P_RUNS))
    t_val = np.abs(b) / s_b
    t_kr = stats.t.ppf(1 - ALPHA / 2, f_repr)
    significant = t_val > t_kr

    print()
    print(f"Коефіцієнти рівняння регресії (t кр = {t_kr:.3f})")
    print("-" * 46)
    print(f"{'Коефіцієнт':<12}{'Значення':>10}{'t':>10}   Значущий")
    print("-" * 46)
    for name, bk, tk, sk in zip(names, b, t_val, significant):
        print(f"{name:<12}{bk:>10.4f}{tk:>10.2f}   {'так' if sk else 'ні'}")
    print("-" * 46)

    # ---------- перевірка адекватності (критерій Фішера) ----------
    d = int(significant.sum())
    y_hat = columns[:, significant] @ b[significant]
    print(f"Значущих коефіцієнтів: {d}")
    if d < n:
        d_ad = P_RUNS * ((y_mean - y_hat) ** 2).sum() / (n - d)
        f_val = d_ad / d_repr
        f_kr = stats.f.ppf(1 - ALPHA, n - d, f_repr)
        print(f"Критерій Фішера: F = {f_val:.3f}, F кр = {f_kr:.3f} -> "
              f"{'рівняння адекватне' if f_val < f_kr else 'рівняння неадекватне'}")
    else:
        d_ad = f_val = f_kr = float("nan")
    print("-" * 78)

    # ---------- збереження результатів ----------
    table = pd.DataFrame(plan, columns=FACTORS)
    table.insert(0, "№", range(1, n + 1))
    for i in range(P_RUNS):
        table[f"y{i + 1}"] = y[:, i]
    table["y_mean"], table["D"], table["y_hat"] = y_mean, y_var, y_hat
    table.to_csv(os.path.join(RESULTS_DIR, "experiment_plan.csv"), index=False)
    pd.DataFrame({"coef": names, "b": b, "t": t_val,
                  "significant": significant}).to_csv(
        os.path.join(RESULTS_DIR, "experiment_coef.csv"), index=False)
    pd.DataFrame([{"G": g, "G_kr": g_kr, "D": d_repr, "t_kr": t_kr,
                   "d": d, "D_ad": d_ad, "F": f_val, "F_kr": f_kr}]).to_csv(
        os.path.join(RESULTS_DIR, "experiment_stats.csv"), index=False)

    # ---------- діаграма коефіцієнтів ----------
    plt.figure(figsize=(9, 3.8))
    colors = ["black" if s else "lightgray" for s in significant[1:]]
    plt.bar(names[1:], b[1:], color=colors, edgecolor="black", linewidth=0.6)
    plt.axhline(0, color="black", linewidth=0.8)
    limit = t_kr * s_b
    plt.axhline(limit, color="gray", linestyle="--", linewidth=0.9,
                label="межа значущості")
    plt.axhline(-limit, color="gray", linestyle="--", linewidth=0.9)
    plt.ylabel("Значення коефіцієнта")
    plt.grid(True, axis="y", linewidth=0.3)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "effects.png"), dpi=200)


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print("=" * 78)
    print("ДОСЛІДЖЕННЯ МОДЕЛІ")
    print("=" * 78)
    one_factor_study()
    factorial_experiment()
    print(f"Таблиці та діаграму збережено в папці {RESULTS_DIR}")


if __name__ == "__main__":
    main()
