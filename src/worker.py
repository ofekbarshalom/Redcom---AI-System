# worker.py entrypoint [cite: 92]
from src.tasks import celery_app

if __name__ == '__main__':
    # This allows you to run the worker locally for testing 
    # outside of Docker if needed.
    celery_app.start()