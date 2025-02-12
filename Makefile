.PHONY: setup activate install style

print:
	echo $(SHELL)

all: setup install test

setup:
	pip install uv
	uv venv

activate:
	. .venv/bin/activate

install:
	uv sync

dev:
	uv sync --all-extras

style:
	uv run pre-commit run --all-files
