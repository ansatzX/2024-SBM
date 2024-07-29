import enum
import numpy as np
import os
from typing import List
from enum import Enum

def num_monotonic(p0: float, p1: float, p2: float) -> int :
    """
    filed 3
    return flag
    0: ascend
    1: decend
    2: non monotonic peak
    3: non monotonic valley
    """
    assert isinstance(p0, float)
    assert isinstance(p1, float)
    assert isinstance(p2, float)

    sorted_ascend: List[float] = sorted([p0, p1, p2])

    if sorted_ascend[0] == p2:
        return 3
    elif sorted_ascend[2] == p2:
        return 2
    elif sorted_ascend[0] == p0 and sorted_ascend[2] == p2:
        return 0
    elif sorted_ascend[0] == p2 and sorted_ascend[2] == p0:
        return 1
    
def line_monotonic_detect(data: List[float]) -> List[int]:

    if data[0] < data[1]:
        is_monotonic_results = [0]
    else: # can not be same
        is_monotonic_results = [1]

    for id in range(1, len(data)-1):
        result: int = num_monotonic(data[id-1], data[id], data[id+1])
        is_monotonic_results.append(result)

    if data[len(data)-2] < data[len(data)-1]:
        is_monotonic_results.append(0)
    else:
        is_monotonic_results.append(1)

    return is_monotonic_results

class Line_Type(Enum):
    # simply line
    One_Peak=0.1
    One_Valley=0.2
    Ascend=1.1
    Decend=1.2
    # complicated
    # Oscillation=2
    Oscillation_SamePeriod=2.1
    Oscillation_PartSamePeriod=2.2
    Oscillation_NoPeriod=2.3

    Peak_WithOscillation=3.1

def line_classify(is_monotonic_results: List[int]) -> int:
    """
    see def LINE TYPE
    """

    if is_monotonic_results.count(2) == 1:
        return Line_Type.One_Peak
    elif is_monotonic_results.count(3) == 1:
        return Line_Type.One_Valley
    elif all(is_monotonic_results) :
        # if oscillation, 0 1 2 atleast exists one, so if true , no 0, also no 2, Decend
        return Line_Type.Decend
    elif not all(is_monotonic_results) and is_monotonic_results.count(2)==0 and is_monotonic_results.count(3) == 0:
        return Line_Type.Ascend
    else:
        if is_monotonic_results.index(2) > is_monotonic_results.index(3):
            starting_label = 3
        else:
            starting_label = 2
        count = 0
        period = []
        for id in range(len(is_monotonic_results)):
            
            if is_monotonic_results[id] == starting_label:
                if len(period) != 0:
                    period.append(count)
                count = 0
            count: int = count + 1
        
        if len(period) > 2:
            pass