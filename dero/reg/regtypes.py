import pandas as pd
from typing import Tuple, List, Union

InteractionTuple = Tuple[str, str]
InteractionTuples = List[InteractionTuple]

StrOrListOfStrs = Union[str, List[str]]

StrOrBool = Union[str, bool]
StrOrListOfStrsOrNone = Union[StrOrListOfStrs, None]