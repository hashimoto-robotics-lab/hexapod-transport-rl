"""Array shapes are documented at their use sites; these aliases fix the dtypes."""

from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

type FloatArray = NDArray[np.float32]
type Observation = FloatArray
type Info = dict[str, Any]
type ResetResult = tuple[Observation, Info]
type StepResult = tuple[Observation, float, bool, bool, Info]


class AgentPolicy(Protocol):
    """Deterministic commands for evaluating either saved model format."""

    def act(self, observations: Observation) -> FloatArray: ...
