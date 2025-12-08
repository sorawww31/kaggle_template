import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv()


@dataclass
class EnvConfig:
    input_dir: Optional[str | Path] = os.getenv("INPUT_DIR")
    output_dir: Optional[str | Path] = os.getenv("OUTPUT_DIR")
    exp_output_dir: Optional[str | Path] = os.getenv("EXP_OUTPUT_DIR")
