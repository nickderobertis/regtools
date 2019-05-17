import pandas as pd
from typing import Tuple, List, Union, Dict

InteractionTuple = Tuple[str, str]
InteractionTuples = List[InteractionTuple]

StrOrListOfStrs = Union[str, List[str]]

StrOrBool = Union[str, bool]
StrOrListOfStrsOrNone = Union[StrOrListOfStrs, None]

PdData = Union[pd.DataFrame, pd.Series]
LinearModelsKwargs = Dict[str, Union[bool, str, PdData]]
DummyColsDict = Dict[str, List[str]]