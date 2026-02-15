CPU_FLAG :=
ifneq ($(CPU),)
    CPU_FLAG := -f compose.cpu.yaml
endif

default: build

build:
	docker compose $(CPU_FLAG) build

bash:
	docker compose $(CPU_FLAG) run --rm kaggle bash

jupyter:
	docker compose $(CPU_FLAG) up

down:
	docker compose $(CPU_FLAG) down

# === uv環境用コマンド ===
uv-setup:
	uv sync --group dev

uv-jupyter:
	uv run jupyter lab --port=8889