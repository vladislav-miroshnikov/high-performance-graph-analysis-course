import pygraphblas as pgb
import pytest

from project import sssp, mssp


def test_non_square_sssp():
    adj_matrix = pgb.Matrix.dense(pgb.BOOL, nrows=2, ncols=5)
    with pytest.raises(ValueError):
        sssp(adj_matrix, 0)


def test_non_square_mssp():
    adjacency_matrix = pgb.Matrix.dense(pgb.BOOL, nrows=6, ncols=7)
    with pytest.raises(ValueError):
        mssp(adjacency_matrix, [0])


def test_wrong_start_vertex_sssp():
    adjacency_matrix = pgb.Matrix.dense(pgb.BOOL, nrows=3, ncols=3)
    with pytest.raises(ValueError):
        sssp(adjacency_matrix, -1)


def test_wrong_start_vertex_mssp():
    adjacency_matrix = pgb.Matrix.dense(pgb.BOOL, nrows=3, ncols=3)
    with pytest.raises(ValueError):
        mssp(adjacency_matrix, [10])


@pytest.mark.parametrize(
    "I, J, V, size, start_vertex, expected_ans",
    [
        (
            [0, 0, 0, 1, 1, 1, 2, 2, 2],
            [0, 1, 2, 0, 1, 2, 0, 1, 2],
            [0, 2, 10, 100, 7, 9, 21, 12, 23],
            4,
            0,
            [0, 2, 10, -1],
        ),
        (
            [0, 0, 0, 1, 1, 1, 2, 2, 2],
            [0, 1, 2, 0, 1, 2, 0, 1, 2],
            [0.0, 2.0, 10.0, 100.0, 7.0, 9.0, 21.0, 12.0, 23.0],
            3,
            0,
            [0.0, 2.0, 10.0],
        ),
        ([0], [1], [7], 5, 0, [0, 7, -1, -1, -1]),
        ([0], [1], [2.0], 2, 0, [0.0, 2.0]),
    ],
)
def test_sssp(I, J, V, size, start_vertex, expected_ans):
    adj_matrix = pgb.Matrix.from_lists(I, J, V, nrows=size, ncols=size)
    assert sssp(adj_matrix, start_vertex) == expected_ans


@pytest.mark.parametrize(
    "I, J, V, size, start_vertices, expected_ans",
    [
        (
            [0, 0, 0, 1, 1, 1, 2, 2, 2],
            [0, 1, 2, 0, 1, 2, 0, 1, 2],
            [0, 2, 10, 100, 7, 9, 21, 12, 23],
            4,
            [0, 2],
            [(0, [0, 2, 10, -1]), (2, [21, 12, 0, -1])],
        ),
        (
            [0, 0, 0, 1, 1, 1, 2, 2, 2],
            [0, 1, 2, 0, 1, 2, 0, 1, 2],
            [0.0, 2.0, 10.0, 100.0, 7.0, 9.0, 21.0, 12.0, 23.0],
            3,
            [0, 1, 2],
            [(0, [0.0, 2.0, 10.0]), (1, [30.0, 0.0, 9.0]), (2, [21.0, 12.0, 0.0])],
        ),
        (
            [0],
            [1],
            [7],
            5,
            [3, 4],
            [(3, [-1, -1, -1, 0, -1]), (4, [-1, -1, -1, -1, 0])],
        ),
        ([0], [1], [2.0], 2, [0, 1], [(0, [0.0, 2.0]), (1, [-1, 0.0])]),
    ],
)
def test_mssp(I, J, V, size, start_vertices, expected_ans):
    adj_matrix = pgb.Matrix.from_lists(I, J, V, nrows=size, ncols=size)
    assert mssp(adj_matrix, start_vertices) == expected_ans
