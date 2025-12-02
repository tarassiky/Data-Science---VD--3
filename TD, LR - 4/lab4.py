"""
Лабораторная работа №6: Прогнозирование временных рядов (AR-модель)
Выполнение всех шагов задания
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from statsmodels.tsa.stattools import adfuller, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.ar_model import AutoReg
import warnings

warnings.filterwarnings('ignore')

# ==================== 1. ЗАГРУЗКА ДАННЫХ ====================

print("=" * 60)
print("ЛАБОРАТОРНАЯ РАБОТА №6: АНАЛИЗ ВРЕМЕННЫХ РЯДОВ")
print("=" * 60)

# Загрузка данных (предполагаем, что файл уже распакован)
try:
    df = pd.read_csv('tovar_moving.csv', parse_dates=['date'])
    print("✓ Данные успешно загружены")
except:
    # Если файл не найден, создадим демо-данные для примера
    print("⚠ Файл 'tovar_moving.csv' не найден. Используются демо-данные.")
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    np.random.seed(42)
    trend = np.linspace(50, 150, 100)
    seasonal = 20 * np.sin(2 * np.pi * np.arange(100) / 30)
    noise = np.random.normal(0, 10, 100)
    values = trend + seasonal + noise
    values = np.maximum(values, 10)  # Отрицательных заказов нет
    df = pd.DataFrame({'date': dates, 'qty': values})

df.set_index('date', inplace=True)
df.sort_index(inplace=True)

# ==================== 2. РАЗДЕЛЕНИЕ НА TRAIN/TEST ====================

train = df.iloc[:-1].copy()
test = df.iloc[-1:].copy()

print(f"\n1. РАЗДЕЛЕНИЕ ДАННЫХ:")
print(f"   - Общий размер ряда: {len(df)}")
print(f"   - Обучающая выборка: {len(train)}")
print(f"   - Тестовая выборка: {len(test)}")
print(f"   - Тестовое значение: {test['qty'].values[0]:.2f}")

# ==================== 3. АНАЛИЗ ТРЕНДА И СЕЗОННОСТИ ====================

print(f"\n2. АНАЛИЗ ТРЕНДА И СЕЗОННОСТИ:")

# Создаём графики
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Анализ временного ряда', fontsize=16)

# 3.1. Исходный ряд
axes[0, 0].plot(train.index, train['qty'], label='Обучающая выборка', linewidth=1)
axes[0, 0].plot(test.index, test['qty'], 'ro', label='Тестовое значение')
axes[0, 0].set_title('Исходный временной ряд')
axes[0, 0].set_xlabel('Дата')
axes[0, 0].set_ylabel('Количество заказов')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 3.2. Скользящее среднее (тренд)
window = min(30, len(train) // 4)
train['rolling_mean'] = train['qty'].rolling(window=window).mean()
train['rolling_std'] = train['qty'].rolling(window=window).std()

axes[0, 1].plot(train.index, train['qty'], alpha=0.5, label='Исходный ряд', linewidth=0.8)
axes[0, 1].plot(train.index, train['rolling_mean'], 'r-', label=f'Скользящее среднее ({window} дней)', linewidth=2)
axes[0, 1].fill_between(train.index,
                        train['rolling_mean'] - train['rolling_std'],
                        train['rolling_mean'] + train['rolling_std'],
                        alpha=0.2, color='red')
axes[0, 1].set_title('Анализ тренда')
axes[0, 1].set_xlabel('Дата')
axes[0, 1].set_ylabel('Количество заказов')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# 3.3. Сезонная декомпозиция (упрощённая)
if len(train) > 30:
    # Разбиваем на сезоны по 30 дней
    train['month'] = train.index.month
    seasonal_pattern = train.groupby('month')['qty'].mean()

    axes[1, 0].bar(seasonal_pattern.index, seasonal_pattern.values)
    axes[1, 0].set_title('Сезонность по месяцам (средние значения)')
    axes[1, 0].set_xlabel('Месяц')
    axes[1, 0].set_ylabel('Среднее количество заказов')
    axes[1, 0].grid(True, alpha=0.3, axis='y')
else:
    axes[1, 0].text(0.5, 0.5, 'Недостаточно данных\nдля анализа сезонности',
                    ha='center', va='center', transform=axes[1, 0].transAxes)
    axes[1, 0].set_title('Анализ сезонности')

# 3.4. График автокорреляции (ACF)
plot_acf(train['qty'].dropna(), lags=min(40, len(train) // 2), ax=axes[1, 1])
axes[1, 1].set_title('Функция автокорреляции (ACF)')

plt.tight_layout()
plt.savefig('time_series_analysis.png', dpi=150, bbox_inches='tight')
print("   ✓ Графики сохранены в файл 'time_series_analysis.png'")

# Выводы о тренде и сезонности
trend_strength = abs(train['qty'].corr(pd.Series(range(len(train)))))
print(f"   - Наличие тренда: {'Да' if trend_strength > 0.3 else 'Слабый/Нет'}")
print(f"   - Сила тренда (корреляция с временем): {trend_strength:.3f}")

# ==================== 4. ЭКСПОНЕНЦИАЛЬНОЕ СГЛАЖИВАНИЕ ====================

print(f"\n3. ЭКСПОНЕНЦИАЛЬНОЕ СГЛАЖИВАНИЕ (α=0.7):")

model_ses = SimpleExpSmoothing(train['qty']).fit(smoothing_level=0.7, optimized=False)
forecast_ses = model_ses.forecast(1)
forecast_ses_value = forecast_ses.values[0]
actual_value = test['qty'].values[0]
error_ses = abs(actual_value - forecast_ses_value)

print(f"   - Прогноз: {forecast_ses_value:.2f}")
print(f"   - Фактическое значение: {actual_value:.2f}")
print(f"   - Абсолютная ошибка: {error_ses:.2f}")
print(f"   - Относительная ошибка: {error_ses / actual_value * 100:.1f}%")

# ==================== 5. ПРОВЕРКА НА СТАЦИОНАРНОСТЬ ====================

print(f"\n4. ПРОВЕРКА НА СТАЦИОНАРНОСТЬ:")


def adf_test(series, name=""):
    result = adfuller(series.dropna())
    is_stationary = result[1] < 0.05
    return {
        'name': name,
        'adf_statistic': result[0],
        'p_value': result[1],
        'is_stationary': is_stationary,
        'd_order': 0
    }


# Проверяем исходный ряд
result_original = adf_test(train['qty'], "Исходный ряд")
print(f"   - Тест Дики-Фуллера для исходного ряда:")
print(f"     ADF статистика: {result_original['adf_statistic']:.3f}")
print(f"     p-value: {result_original['p_value']:.4f}")
print(f"     Стационарен: {'Да' if result_original['is_stationary'] else 'Нет'}")

# Определяем порядок интегрирования
d = 0
current_series = train['qty'].copy()

if not result_original['is_stationary']:
    print(f"\n   Поиск порядка интегрирования d:")
    for i in range(1, 3):
        diff_series = current_series.diff(i).dropna()
        if len(diff_series) > 10:
            result_diff = adfuller(diff_series)
            print(f"     d={i}: p-value = {result_diff[1]:.4f}", end="")
            if result_diff[1] < 0.05:
                d = i
                print(" → стационарен")
                break
            else:
                print(" → нестационарен")
        else:
            print(f"     d={i}: недостаточно данных после дифференцирования")
            break

print(f"   - Порядок интегрирования d: {d}")

# ==================== 6. ОПРЕДЕЛЕНИЕ ПОРЯДКА AR МОДЕЛИ (PACF) ====================

print(f"\n5. ОПРЕДЕЛЕНИЕ ПОРЯДКА AR МОДЕЛИ:")

# Строим PACF для определения p
plt.figure(figsize=(10, 4))
pacf_values, confint = pacf(train['qty'].dropna(), nlags=20, alpha=0.05, method='ywm')

# Определяем значимые лаги (выходят за доверительный интервал)
significant_lags = []
for i in range(1, len(pacf_values)):
    if abs(pacf_values[i]) > confint[i, 1] - pacf_values[i]:
        significant_lags.append(i)

# Выбираем порядок p (обычно первый значимый лаг или несколько первых)
if significant_lags:
    p = min(significant_lags[-1], 10)  # Ограничиваем максимальный порядок
    if p > 5:
        # Если много значимых лагов, смотрим на "обрыв" PACF
        for i in range(1, len(significant_lags) - 1):
            if significant_lags[i + 1] - significant_lags[i] > 2:
                p = significant_lags[i]
                break
else:
    p = 1  # По умолчанию

print(f"   - Значимые лаги по PACF: {significant_lags}")
print(f"   - Выбранный порядок модели AR(p): {p}")

# Визуализация PACF
plot_pacf(train['qty'].dropna(), lags=20, method='ywm')
plt.title('Частичная автокорреляционная функция (PACF)')
plt.grid(True, alpha=0.3)
plt.savefig('pacf_plot.png', dpi=150, bbox_inches='tight')
print("   ✓ График PACF сохранён в файл 'pacf_plot.png'")

# ==================== 7. ПОСТРОЕНИЕ AR МОДЕЛИ ====================

print(f"\n6. ПОСТРОЕНИЕ AR({p}) МОДЕЛИ:")

# Если ряд нестационарен, дифференцируем
if d > 0:
    train_ar = train['qty'].diff(d).dropna()
else:
    train_ar = train['qty'].dropna()

# Строим AR модель
try:
    model_ar = AutoReg(train_ar, lags=p).fit()
    print(f"   - Модель успешно обучена")
    print(f"   - Коэффициенты модели:")
    for i, coef in enumerate(model_ar.params):
        print(f"     Lag {i}: {coef:.4f}" if i > 0 else f"     Const: {coef:.4f}")

    # Прогноз
    forecast_ar_diff = model_ar.predict(start=len(train_ar), end=len(train_ar))

    # Если дифференцировали, возвращаем к исходному масштабу
    if d > 0:
        last_value = train['qty'].iloc[-d]
        forecast_ar_value = last_value + forecast_ar_diff.values[0]
        for _ in range(1, d):
            forecast_ar_value += forecast_ar_diff.values[0]
    else:
        forecast_ar_value = forecast_ar_diff.values[0]

    error_ar = abs(actual_value - forecast_ar_value)

    print(f"   - Прогноз AR({p}): {forecast_ar_value:.2f}")
    print(f"   - Фактическое значение: {actual_value:.2f}")
    print(f"   - Абсолютная ошибка: {error_ar:.2f}")
    print(f"   - Относительная ошибка: {error_ar / actual_value * 100:.1f}%")

except Exception as e:
    print(f"   ⚠ Ошибка при построении AR модели: {e}")
    forecast_ar_value = np.nan
    error_ar = np.nan

# ==================== 8. СРАВНЕНИЕ РЕЗУЛЬТАТОВ ====================

print(f"\n7. СРАВНЕНИЕ РЕЗУЛЬТАТОВ:")

results_df = pd.DataFrame({
    'Метод': ['Экспоненциальное сглаживание', f'AR({p})'],
    'Прогноз': [forecast_ses_value, forecast_ar_value],
    'Фактическое_значение': [actual_value, actual_value],
    'Абсолютная_ошибка': [error_ses, error_ar],
    'Относительная_ошибка_%': [error_ses / actual_value * 100, error_ar / actual_value * 100]
})

print("\n" + results_df.to_string(index=False))

# Сохраняем результаты в CSV
results_df.to_csv('results.csv', index=False, encoding='utf-8-sig')
print(f"\n✓ Результаты сохранены в файл 'results.csv'")

# Определяем лучший метод
if not np.isnan(error_ar):
    if error_ses < error_ar:
        best_method = "Экспоненциальное сглаживание"
    else:
        best_method = f"AR({p}) модель"
    print(f"✓ Лучший метод: {best_method}")

# ==================== 9. ВИЗУАЛИЗАЦИЯ ПРОГНОЗОВ ====================

plt.figure(figsize=(12, 6))
plt.plot(train.index[-30:], train['qty'].values[-30:], 'b-', label='Исторические данные', linewidth=2)
plt.plot(test.index, test['qty'].values, 'ko', markersize=10, label='Фактическое значение', markerfacecolor='none')

if not np.isnan(forecast_ses_value):
    plt.plot(test.index, forecast_ses_value, 'ro', markersize=10, label='Прогноз (Exp. Smoothing)')
if not np.isnan(forecast_ar_value):
    plt.plot(test.index, forecast_ar_value, 'gs', markersize=10, label=f'Прогноз (AR({p}))')

plt.title('Сравнение прогнозов', fontsize=14)
plt.xlabel('Дата')
plt.ylabel('Количество заказов')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('forecast_comparison.png', dpi=150, bbox_inches='tight')
print(f"✓ График сравнения прогнозов сохранён в 'forecast_comparison.png'")

# ==================== 10. ОБЩИЕ ВЫВОДЫ ====================

print(f"\n" + "=" * 60)
print("ОБЩИЕ ВЫВОДЫ:")
print("=" * 60)

print(f"1. Характеристики ряда:")
print(f"   - Длина ряда: {len(df)} дней")
print(f"   - Среднее значение: {df['qty'].mean():.2f}")
print(f"   - Стандартное отклонение: {df['qty'].std():.2f}")
print(f"   - Наличие тренда: {'Да' if trend_strength > 0.3 else 'Слабый/Нет'}")

print(f"\n2. Стационарность:")
print(f"   - Исходный ряд стационарен: {'Да' if result_original['is_stationary'] else 'Нет'}")
print(f"   - Порядок интегрирования d: {d}")

print(f"\n3. Прогнозные модели:")
print(f"   - AR порядок (p): {p}")
print(f"   - Лучшая модель: {best_method if 'best_method' in locals() else 'Не определена'}")

print(f"\n4. Точность прогнозов:")
print(f"   - Эксп. сглаживание: ошибка {error_ses:.2f} ({error_ses / actual_value * 100:.1f}%)")
if not np.isnan(error_ar):
    print(f"   - AR модель: ошибка {error_ar:.2f} ({error_ar / actual_value * 100:.1f}%)")

print(f"\n✓ Все файлы успешно сохранены:")
print(f"   - results.csv - таблица с результатами")
print(f"   - time_series_analysis.png - анализ ряда")
print(f"   - pacf_plot.png - график PACF")
print(f"   - forecast_comparison.png - сравнение прогнозов")

plt.show()