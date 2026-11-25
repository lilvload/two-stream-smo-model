# -*- coding: utf-8 -*-
"""Імітаційна модель двохпотокової СМО без пріоритетів.

Об'єкт моделювання: поштове відділення з одним оператором.
Потік 1 - клієнти, які отримують посилки; потік 2 - клієнти, які
відправляють посилки. Черга спільна, обмеженої місткості, дисципліна FIFO.
Модельний час просувається від події до події (календар подій на heapq).
"""
import heapq
from collections import deque

import numpy as np

# Типи подій
ARRIVAL_1 = 1      # надходження клієнта потоку 1
ARRIVAL_2 = 2      # надходження клієнта потоку 2
DEPARTURE = 3      # завершення обслуговування

EVENT_NAMES = {
    ARRIVAL_1: "надходження (потік 1)",
    ARRIVAL_2: "надходження (потік 2)",
    DEPARTURE: "завершення обслуговування",
}


def simulate(lam1, lam2, mu, m, t_end, seed=None, trace=0, history=False):
    """Виконує один прогін моделі.

    lam1, lam2 - інтенсивності вхідних потоків, клієнтів за хвилину;
    mu         - інтенсивність обслуговування, клієнтів за хвилину;
    m          - місткість черги, клієнтів;
    t_end      - час моделювання, хвилин;
    seed       - початкове значення генератора випадкових чисел;
    trace      - скільки перших подій записати до протоколу;
    history    - чи зберігати зміну довжини черги в часі (для графіка).

    Повертає словник з результатами прогону.
    """
    rng = np.random.default_rng(seed)
    lam = {1: lam1, 2: lam2}

    # ---------- стан моделі ----------
    t = 0.0                 # поточний модельний час
    calendar = []           # календар подій: (час, номер, тип події)
    counter = 0             # порядковий номер події в календарі
    queue = deque()         # спільна черга: (час надходження, номер потоку)
    busy = False            # чи зайнятий оператор
    current = None          # номер потоку клієнта, якого обслуговують

    # ---------- статистика ----------
    arrived = {1: 0, 2: 0}      # прийшло клієнтів
    served = {1: 0, 2: 0}       # обслужено клієнтів
    rejected = {1: 0, 2: 0}     # отримали відмову
    area_queue = 0.0            # сума (довжина черги * тривалість)
    busy_time = 0.0             # сумарний час зайнятості оператора
    wait_total = 0.0            # сумарний час очікування в черзі
    started = 0                 # скільки клієнтів почали обслуговуватися
    log = []                    # протокол перших подій
    hist = [(0.0, 0)]           # зміна довжини черги в часі

    def schedule(time, kind):
        """Додає подію до календаря."""
        nonlocal counter
        counter += 1
        heapq.heappush(calendar, (time, counter, kind))

    def start_service(arrival_time, stream):
        """Оператор починає обслуговувати клієнта."""
        nonlocal busy, current, wait_total, started
        busy = True
        current = stream
        wait_total += t - arrival_time
        started += 1
        schedule(t + rng.exponential(1.0 / mu), DEPARTURE)

    # Початкові події: перше надходження клієнта кожного потоку
    schedule(rng.exponential(1.0 / lam1), ARRIVAL_1)
    schedule(rng.exponential(1.0 / lam2), ARRIVAL_2)

    # ---------- основний цикл ----------
    while calendar:
        time, _, kind = heapq.heappop(calendar)    # найближча подія
        if time > t_end:
            break

        # накопичення статистики за проміжок між подіями
        area_queue += len(queue) * (time - t)
        if busy:
            busy_time += time - t
        t = time                                    # просування часу

        if kind in (ARRIVAL_1, ARRIVAL_2):          # надходження клієнта
            stream = kind
            arrived[stream] += 1
            schedule(t + rng.exponential(1.0 / lam[stream]), kind)
            if not busy:
                start_service(t, stream)
                action = "одразу до оператора"
            elif len(queue) < m:
                queue.append((t, stream))
                action = "стає в чергу"
            else:
                rejected[stream] += 1
                action = "відмова"
        else:                                       # завершення обслуговування
            served[current] += 1
            if queue:
                arrival_time, stream = queue.popleft()
                start_service(arrival_time, stream)
                action = "оператор бере наступного з черги"
            else:
                busy = False
                current = None
                action = "оператор вільний"

        if len(log) < trace:
            log.append((t, EVENT_NAMES[kind], len(queue),
                        "зайнятий" if busy else "вільний", action))
        if history:
            hist.append((t, len(queue)))

    # статистика за проміжок від останньої події до кінця моделювання
    area_queue += len(queue) * (t_end - t)
    if busy:
        busy_time += t_end - t
    if history:
        hist.append((t_end, len(queue)))

    # ---------- показники роботи ----------
    n_arrived = arrived[1] + arrived[2]
    n_served = served[1] + served[2]
    n_rejected = rejected[1] + rejected[2]
    return {
        "arrived_1": arrived[1], "arrived_2": arrived[2], "arrived": n_arrived,
        "served_1": served[1], "served_2": served[2], "served": n_served,
        "rejected_1": rejected[1], "rejected_2": rejected[2],
        "rejected": n_rejected,
        "in_system": len(queue) + (1 if busy else 0),
        "p_reject": n_rejected / n_arrived if n_arrived else 0.0,
        "p_served": n_served / n_arrived if n_arrived else 0.0,
        "throughput": n_served / t_end,
        "mean_queue": area_queue / t_end,
        "mean_wait": wait_total / started if started else 0.0,
        "utilization": busy_time / t_end,
        "log": log,
        "history": hist,
    }
