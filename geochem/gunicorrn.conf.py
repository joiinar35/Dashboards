# gunicorn.conf.py
workers = 4
worker_class = "sync"
worker_connections = 1000
timeout = 120