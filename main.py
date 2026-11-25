# -*- coding: utf-8 -*-
"""Базовий прогін моделі поштового відділення.

Запуск:  python main.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from smo_model import simulate

# ---------- вихідні дані ----------
LAM1 = 0.2      # потік 1: клієнти, які отримують посилки, клієнтів/хв
LAM2 = 0.1      # потік 2: клієнти, які відправляють посилки, клієнтів/хв
MU = 0.4        # інтенсивність обслуговування, клієнтів/хв
M = 5           # місткість черги, клієнтів
T_END = 480     # час моделювання, хв (робочий день 8 годин)
SEED = 12       # початкове значення генератора випадкових чисел
TRACE = 15      # кількість подій у протоколі

RESULTS_DIR = "results"


def main():
    res = simulate(LAM1, LAM2, MU, M, T_END, seed=SEED, trace=TRACE,
                   history=True)

    print("=" * 78)
    print("ІМІТАЦІЙНА МОДЕЛЬ ДВОХПОТОКОВОЇ СМО БЕЗ ПРІОРИТЕТІВ")
    print("Поштове відділення з одним оператором")
    print("=" * 78)
    print("Вихідні дані:")
    print(f"  інтенсивність потоку 1 (отримання посилок)   lam1 = {LAM1}")
    print(f"  інтенсивність потоку 2 (відправлення)        lam2 = {LAM2}")
    print(f"  інтенсивність обслуговування                 mu   = {MU}")
    print(f"  місткість черги                              m    = {M}")
    print(f"  час моделювання, хв                          T    = {T_END}")
    print()
    print(f"Протокол перших {TRACE} подій:")
    print("-" * 78)
    print(f"{'Час, хв':>8}  {'Подія':<26}{'Черга':>5}  {'Оператор':<9} Дія")
    print("-" * 78)
    for t, name, q, op, action in res["log"]:
        print(f"{t:8.2f}  {name:<26}{q:>5}  {op:<9} {action}")
    print("-" * 78)
    print()
    print("РЕЗУЛЬТАТИ МОДЕЛЮВАННЯ")
    print("-" * 78)
    print(f"{'Показник':<44}{'Потік 1':>10}{'Потік 2':>10}{'Разом':>10}")
    print("-" * 78)
    for title, key in (("Прийшло клієнтів", "arrived"),
                       ("Обслужено клієнтів", "served"),
                       ("Отримали відмову", "rejected")):
        print(f"{title:<44}{res[key + '_1']:>10}{res[key + '_2']:>10}"
              f"{res[key]:>10}")
    print("-" * 78)
    print(f"{'Залишилося у відділенні на кінець дня':<64}"
          f"{res['in_system']:>10}")
    print(f"{'Частка обслужених клієнтів':<64}{res['p_served']:>10.4f}")
    print(f"{'Імовірність відмови':<64}{res['p_reject']:>10.4f}")
    print(f"{'Обслужено клієнтів за хвилину':<64}{res['throughput']:>10.4f}")
    print(f"{'Середня довжина черги, клієнтів':<64}{res['mean_queue']:>10.4f}")
    print(f"{'Середній час очікування в черзі, хв':<64}"
          f"{res['mean_wait']:>10.4f}")
    print(f"{'Коефіцієнт завантаження оператора':<64}"
          f"{res['utilization']:>10.4f}")
    print("=" * 78)

    # ---------- графік довжини черги ----------
    os.makedirs(RESULTS_DIR, exist_ok=True)
    times = [p[0] for p in res["history"]]
    lengths = [p[1] for p in res["history"]]
    plt.figure(figsize=(9, 3.6))
    plt.step(times, lengths, where="post", color="black", linewidth=0.9)
    plt.axhline(M, color="gray", linestyle="--", linewidth=0.9,
                label=f"місткість черги m = {M}")
    plt.xlabel("Модельний час, хв")
    plt.ylabel("Довжина черги, клієнтів")
    plt.xlim(0, T_END)
    plt.ylim(0, M + 0.5)
    plt.grid(True, linewidth=0.3)
    plt.legend(loc="upper left")
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "queue_length.png")
    plt.savefig(path, dpi=200)
    print(f"Графік довжини черги збережено у файлі {path}")


if __name__ == "__main__":
    main()
