import subprocess
import multiprocessing

def run_server():
    subprocess.run(["python", "server.py"])

def run_fine_tuning_scheduler():
    subprocess.run(["python", "fine_tuning_scheduler.py"])

def main():
    # Create processes for server and scheduler
    server_process = multiprocessing.Process(target=run_server)
    scheduler_process = multiprocessing.Process(target=run_fine_tuning_scheduler)

    # Start both processes
    server_process.start()
    scheduler_process.start()

    # Wait for both processes to complete
    server_process.join()
    scheduler_process.join()

if __name__ == "__main__":
    main()