import mlflow
import mlflow.sklearn
import clearml
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from datetime import datetime
from jinja2 import Template
from clearml import Task

task = Task.init(
    project_name="Obesity Prediction",
    task_name="Model Training",
    task_type=Task.TaskTypes.optimizer,
)

# Загрузка данных
data = pd.read_csv("data/clean_data.csv")

# Разделение признаков и целевой переменной
X = data.drop(columns=["Obesity_Binary"])
y = data["Obesity_Binary"]

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Настройка MLflow
mlflow.set_tracking_uri("file:./mlruns")  # Локальное хранилище для логов
experiment_name = "obesity_prediction_experiment"
mlflow.set_experiment(experiment_name)

# Создание директории для сохранения отчетов и диаграмм
output_dir = "mlflow_output"
os.makedirs(output_dir, exist_ok=True)


def save_confusion_matrix(y_true, y_pred, model_name):
    """Создание и сохранение матрицы ошибок"""
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")

    filename = f"confusion_matrix_{model_name.lower().replace(' ', '_')}.png"
    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path)
    plt.close()
    return filename


def log_model_with_metrics(model, model_name, X_train, X_test, y_train, y_test):
    """Логирование модели и метрик в MLflow"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with mlflow.start_run(run_name=f"{model_name}_{timestamp}") as run:

        # Обучение модели
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Метрики
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        # Логирование параметров модели
        mlflow.log_param("model", model_name)
        for param_name, param_value in model.get_params().items():
            mlflow.log_param(param_name, param_value)

        # Логирование метрик
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        # Сохранение матрицы ошибок
        cm_filename = save_confusion_matrix(y_test, y_pred, model_name)
        mlflow.log_artifact(os.path.join(output_dir, cm_filename))

        # Сохранение отчета о классификации
        class_report = classification_report(y_test, y_pred, zero_division=0)
        report_filename = (
            f"classification_report_{model_name.lower().replace(' ', '_')}.txt"
        )
        report_save_path = os.path.join(output_dir, report_filename)

        with open(report_save_path, "w") as f:
            f.write(class_report)
        mlflow.log_artifact(report_save_path)

        # Сохранение модели в MLflow с примером входных данных для автоматического определения сигнатуры модели
        mlflow.sklearn.log_model(
            model,
            "model",
            input_example=X_test.iloc[:1],
            registered_model_name=model_name,
        )

        return {
            "run_id": run.info.run_id,
            "confusion_matrix_path": cm_filename,
            "classification_report_path": report_filename,
        }

        task.upload_artifact("model", artifact_object=model)


def run_experiments():
    """Запуск нескольких экспериментов с разными параметрами"""
    experiments = [
        {
            "model": LogisticRegression(max_iter=1000, random_state=42),
            "name": "Logistic Regression (C=1.0)",
        },
        {
            "model": LogisticRegression(max_iter=2000, random_state=42, C=0.5),
            "name": "Logistic Regression (C=0.5)",
        },
        {
            "model": SVC(kernel="linear", random_state=42),
            "name": "SVM (Linear Kernel)",
        },
        {
            "model": DecisionTreeClassifier(max_depth=10, random_state=42),
            "name": "Decision Tree (max_depth=10)",
        },
        {
            "model": DecisionTreeClassifier(max_depth=15, random_state=42),
            "name": "Decision Tree (max_depth=15)",
        },
    ]

    results = []
    for exp in experiments:
        print(f"Starting experiment with {exp['name']}...")
        run_results = log_model_with_metrics(
            exp["model"], exp["name"], X_train, X_test, y_train, y_test
        )
        run_id = run_results["run_id"]
        # Сохраняем данные эксперимента
        results.append(
            {
                "RunID": run_id,
                "Model": exp["name"],
                "accuracy": mlflow.get_run(run_id).data.metrics["accuracy"],
                "precision": mlflow.get_run(run_id).data.metrics["precision"],
                "recall": mlflow.get_run(run_id).data.metrics["recall"],
                "f1_score": mlflow.get_run(run_id).data.metrics["f1_score"],
                "confusion_matrix_path": run_results["confusion_matrix_path"],
                "classification_report_path": run_results[
                    "classification_report_path"
                ],  # noqa:E501
            }
        )
        task.connect(
            {
                "accuracy": mlflow.get_run(run_id).data.metrics["accuracy"],
                "precision": mlflow.get_run(run_id).data.metrics["precision"],
                "recall": mlflow.get_run(run_id).data.metrics["recall"],
                "f1_score": mlflow.get_run(run_id).data.metrics["f1_score"],
                "confusion_matrix_path": run_results["confusion_matrix_path"],
                "classification_report_path": run_results["classification_report_path"],
            }
        )

        print(f"Completed experiment with {exp['name']}")

    # Преобразуем результаты в DataFrame
    results_df = pd.DataFrame(results)
    return results_df


# Генерация HTML-отчета
def generate_report(filtered_runs):
    # Сохраняем только последние 4 эксперимента
    # filtered_runs = filtered_runs[-4:]

    # Округляем значения до 4 знаков
    for metric in ["accuracy", "precision", "recall", "f1_score"]:
        filtered_runs[metric] = filtered_runs[metric].round(4)

    # Создаем копию для отображения в таблице метрик
    filtered_runs_display = filtered_runs.drop(
        columns=[
            "RunID",
            "confusion_matrix_path",
            "classification_report_path",
        ]  # noqa:E501
    )

    # Подготовка графика метрик
    plt.figure(figsize=(12, 6))  # Увеличиваем ширину
    filtered_runs_display.plot(
        x="Model",
        y=["accuracy", "precision", "recall", "f1_score"],
        kind="bar",
        figsize=(16, 8),
        colormap="viridis",
    )

    plt.title("Сравнение производительности моделей", fontsize=16)
    plt.ylabel("Score")
    plt.xticks(rotation=45, ha="right", fontsize=12)
    plt.legend(title="Metrics", fontsize=12)
    plt.tight_layout()

    # Сохранение графика
    graph_filename = "performance_comparison.png"
    graph_save_path = os.path.join(output_dir, graph_filename)
    plt.savefig(graph_save_path)
    plt.close()

    # Формирование выводов
    best_model = filtered_runs.loc[filtered_runs["f1_score"].idxmax()]["Model"]
    insights = [
        f"Наилучшая модель: {best_model}, так как она показала лучшие метрики.",  # noqa:E501
        "Результаты других моделей проанализированы в таблицах выше.",
        'Модели логистической регрессии также продемонстрировали хорошие результаты, но с некоторыми недостатками в полноте для класса "негатив".',
        "Модели деревьев решений оказались менее эффективными по сравнению с линейными моделями, что может быть связано с их чувствительностью к переобучению, особенно при увеличении глубины дерева.",
    ]

    # Рендеринг HTML-отчета через Jinja2
    template_path = "report_template.html"
    output_path = os.path.join(output_dir, "index.html")

    # Генерация идентификатора эксперимента
    experiment_id = datetime.now().strftime("%Y%m%d%H%M%S")
    current_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Лучшая модель и ее метрики
    best_model = filtered_runs.loc[filtered_runs["accuracy"].idxmax()]
    best_accuracy = best_model["accuracy"]
    best_precision = best_model["precision"]
    best_recall = best_model["recall"]
    best_f1_score = best_model["f1_score"]

    # Загружаем отчеты о классификации
    model_reports = {}
    for _, row in filtered_runs.iterrows():
        model_name = row["Model"]
        report_path = os.path.join(output_dir, row["classification_report_path"])
        with open(report_path, "r") as f:
            model_reports[model_name] = f.read()

    # Загрузка шаблона
    with open(template_path, "r", encoding="utf-8") as file:
        template = Template(file.read())

    # Получаем дополнительную информацию из MLflow
    mlflow_context = enhance_report_generation(filtered_runs)

    # Контекст шаблона
    template_context = {
        "experiment_name": experiment_name,
        "runs": filtered_runs_display.to_dict(orient="records"),
        "full_runs": filtered_runs.to_dict(orient="records"),
        "model_reports": model_reports,
        "insights": insights,
        "graph_path": graph_filename,
        "current_date": current_date,
        "experiment_id": experiment_id,
        "best_accuracy": best_accuracy,
        "best_precision": best_precision,
        "best_recall": best_recall,
        "best_f1_score": best_f1_score,
        **mlflow_context,
    }

    # Рендеринг HTML
    html_content = template.render(**template_context)

    # Сохранение отчета
    with open(output_path, "w", encoding="utf-8") as file:
        file.write(html_content)

    print(f"Report generated and saved to {output_path}")


def get_mlflow_model_performance(run_ids):
    """
    Получение детальной информации о производительности моделей из MLflow
    """
    performance_data = {}

    for run_id in run_ids:
        run = mlflow.get_run(run_id)
        metrics = run.data.metrics
        params = run.data.params

        model_name = params.get("model", "Unknown Model")
        performance_data[run_id] = {
            "model_name": model_name,
            "metrics": metrics,
            "parameters": params,
            "start_time": run.info.start_time,
            "end_time": run.info.end_time,
            # в секундах
            "duration": (run.info.end_time - run.info.start_time) / 1000,
            "status": run.info.status,
        }

    return performance_data


def generate_mlflow_performance_plot(performance_data):
    """
    Генерация дополнительных графиков производительности моделей
    """
    plots = {}

    # График времени выполнения
    plt.figure(figsize=(10, 6))
    durations = [data["duration"] for data in performance_data.values()]
    model_names = [data["model_name"] for data in performance_data.values()]

    plt.bar(model_names, durations)
    plt.title("Model Training Duration")
    plt.xlabel("Model")
    plt.ylabel("Duration (seconds)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    duration_plot_path = os.path.join(output_dir, "model_durations.png")
    plt.savefig(duration_plot_path)
    plt.close()
    plots["duration_plot"] = "model_durations.png"

    return plots


def enhance_report_generation(filtered_runs):
    """
    Дополнительная информацией из MLflow
    """
    # Получаем данные о производительности моделей из MLflow
    run_ids = filtered_runs["RunID"].tolist()
    performance_data = get_mlflow_model_performance(run_ids)

    # Генерируем дополнительные графики
    performance_plot = generate_mlflow_performance_plot(performance_data)

    # Добавляем новые данные в контекст шаблона
    template_context = {
        "mlflow_plot": performance_plot,
        "performance_data": performance_data,
    }

    return template_context


if __name__ == "__main__":
    print("Starting experiments...")
    experiment_results = run_experiments()
    print("Completed experiments")

    print("Generating report...")
    generate_report(experiment_results)
