#!/bin/bash

source .venv/bin/activate
PYTHONPATH=src python3 -m llm "$1"
