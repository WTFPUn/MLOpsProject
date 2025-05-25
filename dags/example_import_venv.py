from datetime import datetime
from airflow import DAG
from airflow.operators.python import PythonVirtualenvOperator

def print_sklearn_version():
    import logging
    logger = logging.getLogger(__name__)
    try:
        import sklearn
        logger.info(f"✅ scikit-learn version: {sklearn.__version__}")
        print(f"✅ scikit-learn version: {sklearn.__version__}")
    except Exception as e:
        logger.error(e)
        # logger.error(traceback.format_exc())


with DAG(
    dag_id="test_virtualenv_sklearn",
    start_date=datetime(2024, 1, 1),
    schedule_interval=None,
    catchup=False,
    tags=["test", "virtualenv"],
) as dag:

    sklearn_task = PythonVirtualenvOperator(
        task_id="print_sklearn_version",
        python_callable=print_sklearn_version,
        python_version="3.10",
        requirements=["scikit-learn"],  # you can change the version as needed
        system_site_packages=False,            # set to True if you want access to global packages
    )

    sklearn_task
