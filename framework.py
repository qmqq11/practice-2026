import json
import csv
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from dataclasses import dataclass, asdict, field
from pathlib import Path
from datetime import datetime
from typing import Optional
import pandas as pd
import sympy as sp


# ══════════════════════════════════════════════════════════════════════
# РАЗДЕЛ 1: Конфигурация эксперимента
# ══════════════════════════════════════════════════════════════════════
@dataclass
class ModelParams:
    """Параметры математической модели Лотки–Вольтерра."""
    # α — скорость роста популяции жертв
    alpha: float = 1.0
    # β — интенсивность взаимодействия жертвы и хищника
    beta: float = 0.1
    # δ — эффективность превращения добычи в прирост хищников
    delta: float = 0.075
    # γ — естественная смертность хищников
    gamma: float = 1.5

    def equilibrium(self):
        """
        Вычисляет ненулевую точку равновесия системы.

        Для классической модели:
            x* = γ / δ
            y* = α / β

        Именно относительно этой точки строится управление
        и вычисляются метрики качества.
        """
        return self.gamma / self.delta, self.alpha / self.beta


@dataclass
class ControlParams:
    """Параметры АКАР-регулятора."""
    # Постоянная времени T.
    # Чем меньше T, тем быстрее система стремится к равновесию, но тем больше величина управляющего воздействия.
    T: float = 1.0
    control_type: str = "additive"


@dataclass
class SimParams:
    """Параметры численного моделирования."""
    # Время моделирования
    t_max: float = 100.0
    # Количество точек, в которых сохраняется решение
    n_eval: int = 10000
    # Начальное значение x задается как доля от равновесия:
    # x0 = x* · x0_frac
    x0_frac: float = 1.5
    # Аналогично для y:
    # y0 = y* · y0_frac
    y0_frac: float = 0.5
    # Относительная точность метода solve_ivp
    rtol: float = 1e-8


@dataclass
class ExperimentConfig:
    """
    Полная конфигурация одного вычислительного эксперимента.

    Объединяет:
      • параметры модели;
      • параметры регулятора;
      • параметры моделирования.

    Благодаря этому эксперимент можно запустить одной командой,
    сохранить в JSON и затем полностью воспроизвести.
    """
    name: str = "experiment"
    model: ModelParams = field(default_factory=ModelParams)
    control: ControlParams = field(default_factory=ControlParams)
    sim: SimParams = field(default_factory=SimParams)



# ══════════════════════════════════════════════════════════════════════
# РАЗДЕЛ 2: Ядро — модель и управление
# ══════════════════════════════════════════════════════════════════════

def compute_control(x, y, x_star, p: ModelParams, c: ControlParams) -> float:
    """
    Вычисляет управление u(x, y).

    Аддитивное (u входит в ẋ = αx − βxy + u):
        u = −(x − x*)/T − αx + βxy

    Мультипликативное (u заменяет β в ẋ = αx − u·xy):
        u = (αx + (x − x*)/T) / (x·y)
    """

    # Подсказка: используйте c.control_type для выбора типа
    if c.control_type == "additive":
        return -(x - x_star) / c.T - p.alpha * x + p.beta * x * y

    elif c.control_type == "multiplicative":
        return (p.alpha * x + (x - x_star) / c.T) / (x * y)

    else:
        raise ValueError(f"Неизвестный тип управления: {c.control_type}")

# ══════════════════════════════════════════════════════════════════════
# ДОПОЛНИТЕЛЬНАЯ МОДЕЛЬ (вариант Б)
# Лотка–Вольтерра с внутривидовой конкуренцией
# ══════════════════════════════════════════════════════════════════════

@dataclass
class CompetitionModelParams(ModelParams):
    """
    Модель Лотки–Вольтерра с внутривидовой конкуренцией.

        x' = αx − βxy − εx²
        y' = δxy − γy − μy²
    """

    epsilon: float = 0.02
    mu: float = 0.01

    def equilibrium(self):
        """
        Возвращает ненулевое равновесие модели
        с внутривидовой конкуренцией.
        """

        denom = self.beta * self.delta + self.mu * self.epsilon

        x_star = (
            self.mu * self.alpha +
            self.beta * self.gamma
        ) / denom

        y_star = (
            self.delta * self.alpha -
            self.gamma * self.epsilon
        ) / denom

        return x_star, y_star


# ══════════════════════════════════════════════════════════════════════
# Формирование правой части системы дифференциальных уравнений
# ══════════════════════════════════════════════════════════════════════

def make_rhs(p: ModelParams, c: ControlParams,
             x_star: float, y_star: float):
    """
    Создает функцию rhs(t, state), описывающую динамику системы.

    Именно эту функцию затем вызывает solve_ivp на каждом шаге
    численного интегрирования.

    p       — параметры математической модели;
    c       — параметры регулятора;
    x_star,
    y_star  — координаты точки равновесия.
    """
    # Вложенная функция rhs вычисляет производные dx/dt и dy/dt для текущего состояния системы.
    def rhs(t, state):
        x, y = state
        u = compute_control(x, y, x_star, p, c)

        # Проверяем тип управления.
        if c.control_type == "additive":
            # Классическая модель
            if not isinstance(p, CompetitionModelParams):
                dx = (
                    p.alpha * x
                    - p.beta * x * y
                    + u
                )
                dy = (
                    p.delta * x * y
                    - p.gamma * y
                )

            # Модель с внутривидовой конкуренцией
            else:
                dx = (
                    p.alpha * x
                    - p.beta * x * y
                    - p.epsilon * x**2
                    + u
                )
                dy = (
                    p.delta * x * y
                    - p.gamma * y
                    - p.mu * y**2
                )

        # Проверяем второй тип управления.
        elif c.control_type == "multiplicative":
            # Классическая модель
            if not isinstance(p, CompetitionModelParams):
                dx = (
                    p.alpha * x
                    - u * x * y
                )

                dy = (
                    p.delta * x * y
                    - p.gamma * y
                )

            # Модель с внутривидовой конкуренцией
            else:
                dx = (
                    p.alpha * x
                    - u * x * y
                    - p.epsilon * x**2
                )

                dy = (
                    p.delta * x * y
                    - p.gamma * y
                    - p.mu * y**2
                )

        else:
            raise ValueError(
                f"Неизвестный тип управления: {c.control_type}"
            )

        # Возвращаем производные.
        return [dx, dy]
    return rhs


# ══════════════════════════════════════════════════════════════════════
# РАЗДЕЛ 3: Метрики качества
# ══════════════════════════════════════════════════════════════════════

@dataclass
class Metrics:
    """Метрики качества управления."""
    tau:     Optional[float] = None  # время сходимости
    max_psi: float = 0.0             # max|ψ(t)|
    max_u:   float = 0.0             # max|u(t)|
    e_final: float = 0.0             # остаточная ошибка


def compute_metrics(sol, x_star: float,
                    p: ModelParams,
                    c: ControlParams) -> Metrics:
    """
    Вычисляет основные метрики качества управления
    по результатам численного моделирования.

    sol     — результат работы solve_ivp;
    x_star  — координата равновесия;
    p       — параметры модели;
    c       — параметры регулятора.
    """

    # Если численный метод завершился с ошибкой, дальнейшие вычисления выполнять бессмысленно.
    # Возвращаем бесконечные значения всех метрик.
    if not sol.success:
        return Metrics(
            tau=float("inf"),
            max_psi=float("inf"),
            max_u=float("inf"),
            e_final=float("inf")
        )

    # Получаем решение системы.
    # sol.y содержит массив решений:
    # sol.y[0] — значения x(t),
    # sol.y[1] — значения y(t).
    x_sol = sol.y[0]
    y_sol = sol.y[1]

    # ψ = x − x*
    psi = x_sol - x_star

    # Для каждого момента времени вычисляем управление u.
    # compute_control вызывается для каждой пары (x, y).
    # zip объединяет соответствующие элементы x_sol и y_sol.
    u_sol = np.array([
        compute_control(x, y, x_star, p, c)
        for x, y in zip(x_sol, y_sol)
    ])

    # Максимальное отклонение системы от равновесия.
    max_psi = np.max(np.abs(psi))

    # Максимальная величина управляющего воздействия.
    max_u = np.max(np.abs(u_sol))

    # Остаточная ошибка в конце моделирования.
    # Берем последнее значение ψ.
    e_final = abs(psi[-1])

    # Порог окончания переходного процесса. |ψ| станет меньше 1% от начального отклонения.
    threshold = 0.01 * abs(psi[0])

    idx = np.where(np.abs(psi) < threshold)[0]

    # Если такие моменты существуют,
    # временем сходимости считаем первый из них.
    if len(idx) > 0:
        tau = sol.t[idx[0]]

    # Иначе считаем, что за время моделирования система не сошлась.
    else:
        tau = None

    return Metrics(
        tau=tau,
        max_psi=max_psi,
        max_u=max_u,
        e_final=e_final
    )

# ══════════════════════════════════════════════════════════════════════
# РАЗДЕЛ 4: Запуск эксперимента
# ══════════════════════════════════════════════════════════════════════

@dataclass
class ExperimentResult:
    """Результат эксперимента."""
    config:  ExperimentConfig
    metrics: Metrics
    t:       np.ndarray
    x:       np.ndarray
    y:       np.ndarray
    u:       np.ndarray
    psi:     np.ndarray
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


def run_experiment(config: ExperimentConfig) -> ExperimentResult:
    """
    Запускает эксперимент и возвращает ExperimentResult.
    Это основная функция фреймворка.

    Пример использования:
        cfg = ExperimentConfig(name="test", control=ControlParams(T=0.5))
        result = run_experiment(cfg)
    """
    p = config.model
    c = config.control
    s = config.sim


    # Находим точку равновесия модели.
    x_star, y_star = p.equilibrium()

    # Вычисляем начальные условия.
    # Они задаются как доля от точки равновесия.
    x0 = x_star * s.x0_frac
    y0 = y_star * s.y0_frac

    # Формируем массив времени,в которых будет сохраняться решение.
    t_eval = np.linspace(0, s.t_max, s.n_eval)

    #Правая часть
    rhs = make_rhs(p, c, x_star, y_star)
    #решение дифф уравнений
    sol = solve_ivp(rhs, (0, s.t_max), [x0, y0],
                    t_eval=t_eval, method='RK45', rtol=s.rtol)
    # Из результата извлекаем найденные траектории и вычисляем управление, метрики
    x_sol = sol.y[0]
    y_sol = sol.y[1]
    u_sol = np.array([compute_control(x, y, x_star, p, c)
                      for x, y in zip(x_sol, y_sol)])
    psi_sol = x_sol - x_star

    metrics = compute_metrics(sol, x_star, p, c)

    return ExperimentResult(
        config=config, metrics=metrics,
        t=sol.t, x=x_sol, y=y_sol, u=u_sol, psi=psi_sol
    )


# ══════════════════════════════════════════════════════════════════════
# РАЗДЕЛ 5: Сохранение и загрузка
# ══════════════════════════════════════════════════════════════════════

def save_experiment(result: ExperimentResult, path: str) -> None:
    """
    Сохраняет конфигурацию и метрики в JSON-файл.
    Массивы t, x, y, u, psi НЕ сохраняются (слишком большие).
    Для воспроизведения достаточно конфигурации.

    Формат файла:
    {
      "name": "...",
      "timestamp": "...",
      "config": { ... },
      "metrics": { ... }
    }
    """
    data = {
        "name": result.config.name,
        "timestamp": result.timestamp,
        "config": asdict(result.config),
        "metrics": asdict(result.metrics)
    }
    # Если папка для сохранения не существует, создаем ее автоматически.
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def load_and_reproduce(path: str) -> ExperimentResult:
    """
    Загружает конфигурацию из JSON и воспроизводит эксперимент.
    Возвращает ExperimentResult как при обычном запуске.
    """
    # TODO: реализуйте загрузку и воспроизведение
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    cfg = data["config"]

    config = ExperimentConfig(
        name=cfg["name"],
        model=ModelParams(**cfg["model"]),
        control=ControlParams(**cfg["control"]),
        sim=SimParams(**cfg["sim"])
    )

    return run_experiment(config)

def save_metrics_csv(results: list, path: str) -> None:
    """
    Сохраняет метрики нескольких экспериментов в CSV.
    Каждая строка: name, T, alpha, beta, delta, gamma,
                   tau, max_psi, max_u, e_final
    """
    # TODO: реализуйте сохранение в CSV
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path,
              "w",
              newline="",
              encoding="utf-8") as f:

        writer = csv.writer(f)

        writer.writerow([
            "name",
            "T",
            "alpha",
            "beta",
            "delta",
            "gamma",
            "tau",
            "max_psi",
            "max_u",
            "e_final"
        ])

        for result in results:

            cfg = result.config
            m = result.metrics

            writer.writerow([
                cfg.name,
                cfg.control.T,
                cfg.model.alpha,
                cfg.model.beta,
                cfg.model.delta,
                cfg.model.gamma,
                m.tau,
                m.max_psi,
                m.max_u,
                m.e_final
            ])


# ══════════════════════════════════════════════════════════════════════
# РАЗДЕЛ 6: Визуализация
# ══════════════════════════════════════════════════════════════════════

def plot_experiment(result: ExperimentResult,
                    save_path: Optional[str] = None) -> None:
    """
    Строит стандартные 4 графика для одного эксперимента:
      - x(t) и y(t) с линиями равновесия
      - u(t)
      - ψ(t)
      - фазовый портрет (x, y)

    Если save_path задан — сохраняет PNG, иначе показывает plt.show().
    """
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))

    x_star, y_star = result.config.model.equilibrium()

    axs[0, 0].plot(result.t, result.x, label="x")
    axs[0, 0].plot(result.t, result.y, label="y")
    axs[0, 0].axhline(x_star, linestyle="--")
    axs[0, 0].axhline(y_star, linestyle="--")
    axs[0, 0].set_xlabel("t")
    axs[0, 0].set_ylabel("Population")
    axs[0, 0].legend()

    axs[0, 1].plot(result.t, result.u)
    axs[0, 1].set_title("u(t)")
    axs[0, 1].set_xlabel("t")
    axs[0, 1].set_ylabel("u")

    axs[1, 0].plot(result.t, result.psi)
    axs[1, 0].set_title("psi(t)")
    axs[1, 0].set_xlabel("t")
    axs[1, 0].set_ylabel("psi")

    axs[1, 1].plot(result.x, result.y)
    axs[1, 1].set_title("Phase portrait")
    axs[1, 1].set_xlabel("x")
    axs[1, 1].set_ylabel("y")

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
    else:
        plt.show()

    plt.close()


def plot_compare(results: list, metric: str = "psi",
                 save_path: Optional[str] = None) -> None:

    """
    Сравнивает несколько экспериментов на одном графике.

    metric: "psi" — график ψ(t), "x" — график x(t), "u" — график u(t)
    Каждый эксперимент подписывается именем из config.name.

    Пример использования:
        r1 = run_experiment(ExperimentConfig("T=0.5", control=ControlParams(T=0.5)))
        r2 = run_experiment(ExperimentConfig("T=1.0", control=ControlParams(T=1.0)))
        r3 = run_experiment(ExperimentConfig("T=2.0", control=ControlParams(T=2.0)))
        plot_compare([r1, r2, r3], metric="psi")
    """
    plt.figure(figsize=(8, 6))
    for result in results:

        if metric == "psi":
            y = result.psi
        elif metric == "x":
            y = result.x
        elif metric == "u":
            y = result.u
        else:
            raise ValueError("Неизвестная метрика")

        plt.plot(result.t, y, label=result.config.name)

    plt.xlabel("t")
    plt.ylabel(metric)
    plt.legend()
    plt.grid(True)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
    else:
        plt.show()

    plt.close()


def summary_table(results: list) -> None:
    """
    Выводит сводную таблицу метрик для списка экспериментов.
    Формат вывода — pandas DataFrame.

    Колонки: name, T, tau, max_psi, max_u, e_final
    """
    # TODO: реализуйте сводную таблицу
    rows = []

    for result in results:
        rows.append({
            "name": result.config.name,
            "T": result.config.control.T,
            "tau": result.metrics.tau,
            "max_psi": result.metrics.max_psi,
            "max_u": result.metrics.max_u,
            "e_final": result.metrics.e_final
        })

    df = pd.DataFrame(rows)

    print(df)


# ══════════════════════════════════════════════════════════════════════
# УНИВЕРСАЛЬНЫЙ СИМВОЛЬНЫЙ ВЫВОД АКАР
# ══════════════════════════════════════════════════════════════════════

def derive_control_symbolic(x, y, f, g, psi, u, T):
    """
    Автоматически выводит закон управления АКАР
    средствами библиотеки SymPy.

    На вход передаются:
      x, y  — символьные переменные состояния;
      f     — правая часть первого уравнения x' = f(...);
      g     — правая часть второго уравнения y' = g(...);
      psi   — выбранная макропеременная ψ(x,y);
      u     — символ управления;
      T     — постоянная времени АКАР.

    Возвращает:
      u_expr — символьную формулу управления;
      u_func — функцию для численного вычисления управления.
    """

    # Вычисляем производную макропеременной ψ
    # по правилу полной производной: ψ̇ = ∂ψ/∂x · ẋ + ∂ψ/∂y · ẏ
    # Вместо ẋ и ẏ используются заданные функции f(x,y,u) и g(x,y,u).
    psi_dot = (
        sp.diff(psi, x) * f +
        sp.diff(psi, y) * g
    )

    # Упрощаем полученное выражение.
    psi_dot = sp.simplify(psi_dot)

    # Формируем условие АКАР: ψ̇ = −ψ / T
    # Eq() создает символьное уравнение.
    equation = sp.Eq(
        psi_dot,
        -psi / T
    )

    # Решаем полученное уравнение относительно управления u.
    # solve() возвращает список решений, поэтому берем первое.
    u_expr = sp.solve(
        equation,
        u
    )[0]

    u_expr = sp.simplify(u_expr)

    # Определяем, какие параметры присутствуют в формуле.
    # Исключаем x и y, так как они являются переменными состояния.
    parameters = sorted(
        list(
            u_expr.free_symbols - {x, y}
        ),
        key=lambda s: s.name
    )

    u_func = sp.lambdify(
        (x, y, *parameters),
        u_expr,
        "numpy"
    )

    # Возвращаем: символьную формулу; функцию для вычислений.
    return u_expr, u_func








# ══════════════════════════════════════════════════════════════════════
# РАЗДЕЛ 7: Тесты — НЕ МЕНЯТЬ
# ══════════════════════════════════════════════════════════════════════
# Все тесты должны пройти без ошибок. Запуск: python framework_skeleton.py

def run_tests():
    print("Запуск тестов...")
    errors = []

    # Тест 1: равновесие
    p = ModelParams()
    xs, ys = p.equilibrium()
    assert abs(xs - 20.0) < 1e-6 and abs(ys - 10.0) < 1e-6, "Тест 1 провален: equilibrium()"
    print("  [OK] Тест 1: equilibrium()")

    # Тест 2: управление не падает
    try:
        c = ControlParams(T=1.0, control_type="additive")
        u = compute_control(xs*1.5, ys*0.5, xs, p, c)
        assert isinstance(u, (int, float)), "Тест 2: compute_control должен возвращать число"
        print("  [OK] Тест 2: compute_control (additive)")
    except Exception as e:
        print(f"  [FAIL] Тест 2: {e}")
        errors.append(2)

    # Тест 3: мультипликативное управление
    try:
        c2 = ControlParams(T=1.0, control_type="multiplicative")
        u2 = compute_control(xs*1.5, ys*0.5, xs, p, c2)
        assert isinstance(u2, (int, float))
        print("  [OK] Тест 3: compute_control (multiplicative)")
    except Exception as e:
        print(f"  [FAIL] Тест 3: {e}")
        errors.append(3)

    # Тест 4: run_experiment
    try:
        cfg = ExperimentConfig(name="test")
        r = run_experiment(cfg)
        assert len(r.t) > 0 and len(r.x) == len(r.t)
        print("  [OK] Тест 4: run_experiment()")
    except Exception as e:
        print(f"  [FAIL] Тест 4: {e}")
        errors.append(4)

    # Тест 5: метрики
    try:
        assert r.metrics.max_psi > 0
        assert r.metrics.max_u > 0
        assert r.metrics.e_final < 1.0
        print("  [OK] Тест 5: метрики разумные")
    except Exception as e:
        print(f"  [FAIL] Тест 5: {e}")
        errors.append(5)

    # Тест 6: сохранение и воспроизведение
    try:
        save_experiment(r, "/tmp/test_exp.json")
        r2 = load_and_reproduce("/tmp/test_exp.json")
        assert abs(r2.metrics.tau - r.metrics.tau) < 1e-3
        print("  [OK] Тест 6: save/load/reproduce")
    except Exception as e:
        print(f"  [FAIL] Тест 6: {e}")
        errors.append(6)

    # Тест 7: CSV
    try:
        cfg2 = ExperimentConfig(name="test2", control=ControlParams(T=2.0))
        r3 = run_experiment(cfg2)
        save_metrics_csv([r, r3], "/tmp/test_metrics.csv")
        with open("/tmp/test_metrics.csv") as f:
            lines = f.readlines()
        assert len(lines) == 3  # заголовок + 2 строки
        print("  [OK] Тест 7: save_metrics_csv()")
    except Exception as e:
        print(f"  [FAIL] Тест 7: {e}")
        errors.append(7)

    # Тест 8: plot_experiment не падает
    try:
        plot_experiment(r, save_path="/tmp/test_plot.png")
        assert Path("/tmp/test_plot.png").exists()
        print("  [OK] Тест 8: plot_experiment()")
    except Exception as e:
        print(f"  [FAIL] Тест 8: {e}")
        errors.append(8)

    # Тест 9: plot_compare не падает
    try:
        plot_compare([r, r3], metric="psi", save_path="/tmp/test_compare.png")
        assert Path("/tmp/test_compare.png").exists()
        print("  [OK] Тест 9: plot_compare()")
    except Exception as e:
        print(f"  [FAIL] Тест 9: {e}")
        errors.append(9)

    # Тест 10: summary_table не падает
    try:
        summary_table([r, r3])
        print("  [OK] Тест 10: summary_table()")
    except Exception as e:
        print(f"  [FAIL] Тест 10: {e}")
        errors.append(10)

    print()
    if not errors:
        print("Все тесты пройдены!")
    else:
        print(f"Провалено тестов: {len(errors)} (тесты {errors})")
    return len(errors) == 0


if __name__ == "__main__":
    run_tests()
