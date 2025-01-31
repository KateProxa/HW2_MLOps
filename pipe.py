from clearml import PipelineDecorator, Task
import subprocess


@PipelineDecorator.component(cache=False, execution_queue="default")
def process_data():
    subprocess.run(["python", "data_preprocessing.py"], check=True)


@PipelineDecorator.component(execution_queue="default")
def run_experiment():
    try:
        current_task = Task.current_task()
        if current_task:
            task_id = current_task.id
            print(f"Current task ID: {task_id}")

            # Создание новой задачи для обучения модели
            new_task = Task.init(project_name="mlops_project", task_name="train_model")
            new_task.execute_remotely(queue_name="default")  # Запускаем задачу

            # Логирование
            print("Starting training for model 1...")

            # Запускаем скрипт с помощью subprocess
            subprocess.run(["python", "experiments.py"], check=True)

        else:
            print("Current task is None in run_experiment")
            raise ValueError("Current task is None in run_experiment")
    except Exception as e:
        print(f"Error in run_experiment component: {e}")
        raise


@PipelineDecorator.pipeline(
    name="mlops_pipeline", project="mlops_project", version="0.1"
)
def mlops_pipeline_logic():
    process_data()
    run_experiment()


if __name__ == "__main__":
    PipelineDecorator.run_locally()
    mlops_pipeline_logic()
