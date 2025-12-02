# Задача 2: транспортная задача (снабжение военных баз)
import numpy as np
from scipy.optimize import linprog
import matplotlib.pyplot as plt

# ========== 1. РЕШЕНИЕ ЗАДАЧИ ==========
print("=" * 50)
print("ЗАДАЧА 2: ОПТИМИЗАЦИЯ СНАБЖЕНИЯ ВОЕННЫХ БАЗ")
print("=" * 50)

# Проверка сбалансированности
print("\nПРОВЕРКА СБАЛАНСИРОВАННОСТИ:")
total_supply = 150 + 250
total_demand = 120 + 180 + 100
print(f"Общий запас складов: {total_supply} тонн")
print(f"Общая потребность баз: {total_demand} тонн")
if total_supply == total_demand:
    print("✓ Задача сбалансирована")
else:
    print("✗ Задача не сбалансирована")

# Стоимости по маршрутам (в порядке: x11, x12, x13, x21, x22, x23)
c = [8, 6, 10, 9, 7, 5]

# Ограничения-равенства A_eq @ x = b_eq
A_eq = [
    [1, 1, 1, 0, 0, 0],  # склад 1: x11 + x12 + x13 = 150
    [0, 0, 0, 1, 1, 1],  # склад 2: x21 + x22 + x23 = 250
    [1, 0, 0, 1, 0, 0],  # база Альфа: x11 + x21 = 120
    [0, 1, 0, 0, 1, 0],  # база Бета: x12 + x22 = 180
    [0, 0, 1, 0, 0, 1]  # база Гамма: x13 + x23 = 100
]

b_eq = [150, 250, 120, 180, 100]
bounds = [(0, None)] * 6  # все переменные ≥ 0

# Решение
res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')

print('\nРЕЗУЛЬТАТЫ:')
print('Статус:', res.message)
print(f'Минимальная стоимость: {res.fun:,.0f} усл. ед.')

# Извлекаем решение
x = res.x
flows = x.reshape(2, 3)  # преобразуем в матрицу 2×3

print('\nОПТИМАЛЬНЫЙ ПЛАН ПЕРЕВОЗОК:')
warehouses = ['Склад 1', 'Склад 2']
bases = ['Альфа', 'Бета', 'Гамма']

for i in range(2):  # склады
    for j in range(3):  # базы
        amount = flows[i, j]
        if amount > 0.001:  # показываем только используемые маршруты
            cost = c[i * 3 + j]
            total_cost = amount * cost
            print(f'  {warehouses[i]} → База {bases[j]}:')
            print(f'    Количество: {amount:.1f} т')
            print(f'    Стоимость: {cost} × {amount:.1f} = {total_cost:.1f} усл. ед.')

print('\nСВОДНАЯ ТАБЛИЦА:')
print(" " * 10 + "Альфа  Бета  Гамма  | Итого")
print("-" * 40)
for i in range(2):
    row_sum = np.sum(flows[i, :])
    print(f"{warehouses[i]:8} | {flows[i, 0]:5.1f}  {flows[i, 1]:5.1f}  {flows[i, 2]:5.1f}  | {row_sum:5.1f}")
print("-" * 40)
col_sums = np.sum(flows, axis=0)
print(f"Потребность | {col_sums[0]:5.1f}  {col_sums[1]:5.1f}  {col_sums[2]:5.1f}")

# ========== 2. ВИЗУАЛИЗАЦИЯ ==========
print('\nСтроим сетевую диаграмму...')

fig, ax = plt.subplots(figsize=(12, 8))

# Координаты
warehouse_pos = {'Склад 1': (2, 8), 'Склад 2': (2, 3)}
base_pos = {'Альфа': (10, 10), 'Бета': (10, 6), 'Гамма': (10, 2)}

# Рисуем склады
for name, (x, y) in warehouse_pos.items():
    rect = plt.Rectangle((x - 1, y - 0.4), 2, 0.8,
                         facecolor='lightblue', edgecolor='black',
                         linewidth=2, alpha=0.8, zorder=2)
    ax.add_patch(rect)
    ax.text(x, y, f"{name}", ha='center', va='center',
            fontsize=11, fontweight='bold', zorder=3)

# Рисуем базы
for name, (x, y) in base_pos.items():
    rect = plt.Rectangle((x - 1, y - 0.4), 2, 0.8,
                         facecolor='lightgreen', edgecolor='black',
                         linewidth=2, alpha=0.8, zorder=2)
    ax.add_patch(rect)
    ax.text(x, y, f"База {name}", ha='center', va='center',
            fontsize=11, fontweight='bold', zorder=3)

# Рисуем потоки (стрелки)
warehouse_names = list(warehouse_pos.keys())
base_names = list(base_pos.keys())

arrow_style = dict(arrowstyle='->', connectionstyle='arc3',
                   color='red', linewidth=1, alpha=0.7)

for i in range(2):  # склады
    for j in range(3):  # базы
        amount = flows[i, j]
        if amount > 0.001:  # рисуем только ненулевые потоки
            # Толщина стрелки пропорциональна объёму
            linewidth = max(1, amount / 30)

            # Координаты начала и конца
            start = warehouse_pos[warehouse_names[i]]
            end = base_pos[base_names[j]]

            # Рисуем стрелку
            ax.annotate('', xy=end, xytext=start,
                        arrowprops=dict(arrowstyle='->',
                                        linewidth=linewidth,
                                        color='red',
                                        alpha=0.6,
                                        shrinkA=5, shrinkB=5))

            # Подпись с объёмом
            mid_x = (start[0] + end[0]) / 2
            mid_y = (start[1] + end[1]) / 2
            ax.text(mid_x, mid_y + 0.3, f'{amount:.1f} т',
                    ha='center', va='center', fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.2',
                              facecolor='white', alpha=0.8))

# Настройки графика
ax.set_xlim(0, 12)
ax.set_ylim(0, 12)
ax.set_aspect('equal')
ax.axis('off')
ax.set_title('Оптимальный план снабжения военных баз',
             fontsize=14, fontweight='bold', pad=20)

# Легенда
legend_elements = [
    plt.Line2D([0], [0], color='lightblue', lw=4, label='Склады'),
    plt.Line2D([0], [0], color='lightgreen', lw=4, label='Базы'),
    plt.Line2D([0], [0], color='red', lw=2, label='Потоки перевозок')
]
ax.legend(handles=legend_elements, loc='upper left', fontsize=10)

plt.tight_layout()
plt.savefig('transportation_solution.png', dpi=150)
plt.show()

# ========== 3. ДОПОЛНИТЕЛЬНЫЙ АНАЛИЗ ==========
print('\n' + "=" * 50)
print("ДОПОЛНИТЕЛЬНЫЙ АНАЛИЗ:")
print("=" * 50)

# Проверяем, все ли ограничения выполнены
print("\nПРОВЕРКА ВЫПОЛНЕНИЯ ОГРАНИЧЕНИЙ:")
print("1. Склады:")
print(f"   Склад 1: {np.sum(flows[0, :]):.1f} т (должно быть 150 т)")
print(f"   Склад 2: {np.sum(flows[1, :]):.1f} т (должно быть 250 т)")

print("\n2. Базы:")
print(f"   База Альфа: {np.sum(flows[:, 0]):.1f} т (должно быть 120 т)")
print(f"   База Бета: {np.sum(flows[:, 1]):.1f} т (должно быть 180 т)")
print(f"   База Гамма: {np.sum(flows[:, 2]):.1f} т (должно быть 100 т)")

# Анализ использования маршрутов
print("\nАНАЛИЗ МАРШРУТОВ:")
print("Используемые маршруты:")
for i in range(2):
    for j in range(3):
        if flows[i, j] > 0:
            print(f"  ✓ {warehouses[i]} → База {bases[j]}: {flows[i, j]:.1f} т")

print("\nНеиспользуемые маршруты:")
unused_routes = []
for i in range(2):
    for j in range(3):
        if flows[i, j] < 0.001:
            unused_routes.append(f"{warehouses[i]} → База {bases[j]}")
            if len(unused_routes) % 3 == 0:
                print("  " + ", ".join(unused_routes[-3:]))
if len(unused_routes) % 3 != 0:
    print("  " + ", ".join(unused_routes[- (len(unused_routes) % 3):]))

print('\n' + "=" * 50)
print("ЗАДАЧА 2 ВЫПОЛНЕНА!")
print("=" * 50)