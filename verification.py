# -*- coding: utf-8 -*-
"""Верифікація моделі: порівняння результатів імітації з аналітичними
формулами одноканальної СМО з обмеженою чергою.

Запуск:  python verification.py
"""
import os

import numpy as np
import pandas as pd
from scipy import stats

from smo_model import simulate

# ---------- вихідні дані ----------
LAM1, LAM2, MU, M = 0.2, 0.1, 0.4, 5
T_VER = 100_000     # час одного прогону для верифікації, хв
N_RUNS = 20         # кількість незалежних прогонів
RESULTS_DIR = "results"


def analytic(lam, mu, m):
    """Аналітичні характеристики СМО з одним каналом і чергою на m місць."""
    rho = lam / mu
    n = m + 1                                   # найбільша кількість клієнтів
    if abs(rho - 1.0) < 1e-12:
        p = np.full(n + 1, 1.0 / (n + 1))
    else:
        p0 = (1 - rho) / (1 - rho ** (n + 1))
        p = p0 * rho ** np.arange(n + 1)        # імовірності станів
    p_reject = p[n]
    q = 1 - p_reject                            # відносна пропускна здатність
    a = lam * q                                 # абсолютна пропускна здатність
    l_queue = sum((k - 1) * p[k] for k in range(1, n + 1))
    return {
        "p_reject": p_reject, "p_served": q, "throughput": a,
        "mean_queue": l_queue, "mean_wait": l_queue / a,
        "utilization": 1 - p[0],
    }


def run_series(lam1, lam2, mu, m, t_end, n_runs, first_seed=1):
    """Серія незалежних прогонів; повертає таблицю результатів."""
    rows = [simulate(lam1, lam2, mu, m, t_end, seed=first_seed + i)
            for i in range(n_runs)]
    return pd.DataFrame(rows).drop(columns=["log", "history"])


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print("=" * 78)
    print("ВЕРИФІКАЦІЯ МОДЕЛІ")
    print("=" * 78)
    print(f"lam1 = {LAM1}, lam2 = {LAM2}, mu = {MU}, m = {M}; "
          f"час прогону {T_VER} хв, прогонів {N_RUNS}")
    print()

    # ---------- 1. Порівняння з аналітичними формулами ----------
    theory = analytic(LAM1 + LAM2, MU, M)
    df = run_series(LAM1, LAM2, MU, M, T_VER, N_RUNS)
    names = [("p_reject", "Імовірність відмови"),
             ("p_served", "Відносна пропускна здатність"),
             ("throughput", "Абсолютна пропускна здатність"),
             ("mean_queue", "Середня довжина черги"),
             ("mean_wait", "Середній час очікування, хв"),
             ("utilization", "Коефіцієнт завантаження")]
    t_crit = stats.t.ppf(0.975, N_RUNS - 1)
    table1 = []
    print("1. Порівняння з аналітичними формулами")
    print("-" * 78)
    print(f"{'Характеристика':<31}{'Аналіт.':>9}{'Модель':>9}"
          f"{'Дов. інт.':>11}{'Відхил., %':>12}")
    print("-" * 78)
    for key, title in names:
        mean = df[key].mean()
        half = t_crit * df[key].std(ddof=1) / np.sqrt(N_RUNS)
        dev = abs(mean - theory[key]) / theory[key] * 100
        table1.append((title, theory[key], mean, half, dev))
        print(f"{title:<31}{theory[key]:>9.4f}{mean:>9.4f}"
              f"{'±' + format(half, '.4f'):>11}{dev:>12.2f}")
    print("-" * 78)
    pd.DataFrame(table1, columns=["Характеристика", "Аналітична модель",
                                  "Імітаційна модель", "Довірчий інтервал",
                                  "Відхилення, %"]).to_csv(
        os.path.join(RESULTS_DIR, "verification_1.csv"), index=False)

    # ---------- 2. Рівноправність потоків ----------
    arr1, arr2 = df["arrived_1"].sum(), df["arrived_2"].sum()
    rej1, rej2 = df["rejected_1"].sum(), df["rejected_2"].sum()
    print()
    print("2. Рівноправність потоків")
    print("-" * 78)
    print(f"{'Показник':<46}{'Потік 1':>16}{'Потік 2':>16}")
    print("-" * 78)
    print(f"{'Прийшло клієнтів за хвилину':<46}"
          f"{arr1 / (T_VER * N_RUNS):>16.4f}{arr2 / (T_VER * N_RUNS):>16.4f}")
    print(f"{'Частка клієнтів, які отримали відмову':<46}"
          f"{rej1 / arr1:>16.4f}{rej2 / arr2:>16.4f}")
    print("-" * 78)

    print(f"Таблицю збережено в папці {RESULTS_DIR}")


if __name__ == "__main__":
    main()
