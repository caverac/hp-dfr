from typing import Literal

ProblemTypes = Literal["sine", "arctan", "discontinuous", "delta"]

BackendType = Literal["tensorflow", "jax", "pytorch"]

NormType = Literal["l2", "h1"]
