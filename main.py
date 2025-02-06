# Press the green button in the gutter to run the script.
from app.services.queue_job_consumer_service import QueueJobService

queue_job_service = QueueJobService()

if __name__ == "__main__":
    try:
        queue_job_service.consume_jobs()
    except KeyboardInterrupt:
        print(' [*] Stopping job-executor...')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
