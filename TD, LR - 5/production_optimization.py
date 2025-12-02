# Задача 1: оптимизация производства (полный код)
import numpy as np
from scipy.optimize import linprog
import matplotlib.pyplot as plt

# ========== 1. РЕШЕНИЕ ЗАДАЧИ ==========
print("="*50)
print("ЗАДАЧА 1: ОПТИМИЗАЦИЯ ПРОИЗВОДСТВА ЭЛЕКТРОНИКИ")
print("="*50)

# Параметры задачи
profit = np.array([8000, 12000])  # прибыль с x1 и x2
c = -profit  # для максимизации используем минус

# Ограничения A_ub @ x <= b_ub
A_ub = np.array([
    [2, 3],   # процессорное время
    [4, 6],   # оперативная память
    [1, 2]    # аккумуляторы
], dtype=float)

b_ub = np.array([240, 480, 150], dtype=float)
bounds = [(0, None), (0, None)]  # x1 ≥ 0, x2 ≥ 0

# Решение
res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')

print('\nРЕЗУЛЬТАТЫ:')
print('Статус:', res.message)
x1_opt, x2_opt = res.x
print(f'Оптимальное количество:')
print(f'  Смартфонов (x1): {x1_opt:.0f} шт.')
print(f'  Планшетов (x2): {x2_opt:.0f} шт.')
print(f'Максимальная прибыль: {-res.fun:,.0f} руб.')

# ========== 2. АНАЛИЗ РЕСУРСОВ ==========
print('\nАНАЛИЗ ИСПОЛЬЗОВАНИЯ РЕСУРСОВ:')
resources = ['Процессорное время', 'Оперативная память', 'Аккумуляторы']
for i in range(3):
    used = A_ub[i][0]*x1_opt + A_ub[i][1]*x2_opt
    available = b_ub[i]
    percent = (used / available) * 100
    slack = available - used
    print(f'\n{resources[i]}:')
    print(f'  Использовано: {used:.1f} из {available} ({percent:.1f}%)')
    if slack < 0.1:  # если остаток очень маленький
        print(f'  → Ресурс используется полностью!')
        print(f'  → Теневая цена (множитель Лагранжа): {res.slack[i]:.2f}')
    else:
        print(f'  → Остаток ресурса: {slack:.1f}')

# ========== 3. ВИЗУАЛИЗАЦИЯ ==========
print('\nСтроим график...')
x1 = np.linspace(0, 150, 400)

# Границы ограничений
x2_cpu = (240 - 2*x1) / 3      # 2x1 + 3x2 ≤ 240
x2_ram = (480 - 4*x1) / 6      # 4x1 + 6x2 ≤ 480
x2_batt = (150 - x1) / 2       # x1 + 2x2 ≤ 150

# Линия нулевой прибыли для сравнения
profit_line = (0 - 8000*x1) / 12000

# Оптимальная линия прибыли
optimal_profit = -res.fun
x2_opt_line = (optimal_profit - 8000*x1) / 12000

plt.figure(figsize=(10, 7))

# Заполняем допустимую область
# Находим минимальное значение из всех ограничений для каждого x1
x2_min = np.minimum.reduce([x2_cpu, x2_ram, x2_batt])
x2_min = np.maximum(x2_min, 0)  # учитываем x2 ≥ 0

plt.fill_between(x1, 0, x2_min, where=(x2_min>0),
                 alpha=0.2, color='lightblue', label='Допустимая область')

# Рисуем линии ограничений
plt.plot(x1, x2_cpu, 'r-', linewidth=2, label='Процессорное время: 2x₁ + 3x₂ ≤ 240')
plt.plot(x1, x2_ram, 'g-', linewidth=2, label='Оперативная память: 4x₁ + 6x₂ ≤ 480')
plt.plot(x1, x2_batt, 'b-', linewidth=2, label='Аккумуляторы: x₁ + 2x₂ ≤ 150')

# Оптимальная точка
plt.plot(x1_opt, x2_opt, 'ro', markersize=10,
         label=f'Оптимум: ({x1_opt:.0f}, {x2_opt:.0f})')

# Линии уровня прибыли
plt.plot(x1, profit_line, 'k--', alpha=0.5, label='Прибыль = 0')
plt.plot(x1, x2_opt_line, 'r--', alpha=0.7,
         label=f'Оптимальная прибыль: {optimal_profit:,.0f} руб.')

# Настройки графика
plt.xlabel('x₁ - Количество смартфонов (шт.)', fontsize=12)
plt.ylabel('x₂ - Количество планшетов (шт.)', fontsize=12)
plt.title('Задача оптимизации производства', fontsize=14, fontweight='bold')
plt.xlim(0, 160)
plt.ylim(0, 100)
plt.legend(loc='upper right', fontsize=9)
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig('production_solution.png', dpi=150)
plt.show()

print('\n' + "="*50)
print("ЗАДАЧА 1 ВЫПОЛНЕНА!")
print("="*50)