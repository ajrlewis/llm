#! /bin/bash

source .venv/bin/activate
PYTHONPATH=src pytest src/tests
