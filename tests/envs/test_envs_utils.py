import numpy as np
from ac_env_solver.envs.utils import convert_relators_to_presentation

# Test convert_relators_to_presentation function
def test_convert_relators_to_presentation():
    rel1 = [1, 1, -2, -2, -2]
    rel2 = [1, 2, 1, -2, -1, -2]
    max_relator_length = 7
    arr = convert_relators_to_presentation(rel1, rel2, max_relator_length)
    expected_array = np.array(
        [1, 1, -2, -2, -2, 0, 0, 1, 2, 1, -2, -1, -2, 0], dtype=np.int8
    )
    assert np.array_equal(arr, expected_array)