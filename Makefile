CPU_FLAG := # 変数の宣言
ifneq ($(CPU),) #ifneqは変数が空でない場合に真
    CPU_FLAG := -f compose.CPU.yaml
endif

EXP_FLAG :=
ifneq ($(exp),)
    EXP_DIR := $(shell ls -d ./experiments/exp$(exp)_* 2>/dev/null | head -1)
    ifneq ($(EXP_DIR),)
        EXP_FLAG := -v $(EXP_DIR)/:/kaggle/working
    endif
endif

default: build

build:
	docker compose $(CPU_FLAG)  build

bash:
	docker compose $(CPU_FLAG)  run --rm $(EXP_FLAG) kaggle bash

jupyter:
	docker compose $(CPU_FLAG) $(EXP_FLAG) up

down:
	docker compose $(CPU_FLAG) $(EXP_FLAG) down