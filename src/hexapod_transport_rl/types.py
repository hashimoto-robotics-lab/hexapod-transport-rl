"""Array shapes are documented at their use sites; these aliases fix the dtypes."""

from typing import Any

import numpy as np
from numpy.typing import NDArray

type FloatArray = NDArray[np.float32]
type Observation = FloatArray
type Info = dict[str, Any]
type ResetResult = tuple[Observation, Info]
type StepResult = tuple[Observation, float, bool, bool, Info]
