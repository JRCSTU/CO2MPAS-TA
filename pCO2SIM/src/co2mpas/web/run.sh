#!/usr/bin/env bash
gunicorn --worker-class eventlet -w 1 app:app