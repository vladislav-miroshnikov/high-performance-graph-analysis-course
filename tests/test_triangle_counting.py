import pygraphblas as pgb
import pytest

from project import count_triangles


@pytest.fixture(params=[pgb.INT64, pgb.INT32, pgb.FC64, pgb.UINT8])
def pgb_types(request):
    return request.param


def test_wrong_matrix_type_triangle_counting(pgb_types):
    adjacency_matrix = pgb.Matrix.dense(pgb_types, nrows=3, ncols=3)
    with pytest.raises(ValueError):
        count_triangles(adjacency_matrix)


def test_non_square_triangle_counting():
    adjacency_matrix = pgb.Matrix.dense(pgb.BOOL, nrows=2, ncols=5)
    with pytest.raises(ValueError):
        count_triangles(adjacency_matrix)


@pytest.mark.parametrize(
    "I, J, V, size, expected_ans",
    [
        ([1], [1], [True], 4, [0] * 4),
        ([0, 1, 1, 2, 2, 0], [1, 0, 2, 1, 0, 2], [True] * 6, 4, [1, 1, 1, 0]),
        (
            [0, 1, 0, 5, 0, 4, 4, 5, 4, 3, 3, 5, 3, 2, 5, 2, 1, 5, 1, 2],
            [1, 0, 5, 0, 4, 0, 5, 4, 3, 4, 5, 3, 2, 3, 2, 5, 5, 1, 2, 1],
            [True] * 20,
            12,
            [2, 2, 2, 2, 2, 5, 0, 0, 0, 0, 0, 0],
        ),
        (
            [0, 1, 1, 2, 2, 0, 3, 2, 3],
            [1, 0, 2, 1, 0, 2, 2, 3, 1],
            [True] * 9,
            5,
            [1, 1, 1, 0, 0],
        ),
    ],
)
def test_count_triangles(I, J, V, size, expected_ans):
    adj_matrix = pgb.Matrix.from_lists(I, J, V, nrows=size, ncols=size)
    assert count_triangles(adj_matrix) == expected_ans
