from typing import List, Tuple

import pygraphblas as gb

__all__ = ["sssp", "mssp"]


def sssp(adjacency_matrix: gb.Matrix, start_vertex: int) -> List[int]:
    return tuple(next(iter(mssp(adjacency_matrix, [start_vertex]))))[1]


def mssp(
    adjacency_matrix: gb.Matrix, start_vertices: List[int]
) -> List[Tuple[int, List[int]]]:
    if not adjacency_matrix.square:
        raise ValueError("Adjacency matrix must be a square matrix")

    if any(start < 0 or start >= adjacency_matrix.nrows for start in start_vertices):
        raise ValueError(
            f"The number of the starting vertex or vertices must be between 0 and {adjacency_matrix.nrows - 1}"
        )

    current_front = gb.Matrix.sparse(
        adjacency_matrix.type, nrows=len(start_vertices), ncols=adjacency_matrix.ncols
    )

    for i, j in enumerate(start_vertices):
        current_front.assign_scalar(0, i, j)

    for _ in range(adjacency_matrix.nrows):
        current_front.mxm(
            adjacency_matrix,
            semiring=adjacency_matrix.type.min_plus,
            out=current_front,
            accum=adjacency_matrix.type.min,
        )

    def __calculate_sp(vertices, distances):
        result = [-1] * adjacency_matrix.nrows
        for i, vertex in enumerate(vertices):
            result[vertex] = distances[i]
        return result

    return [
        (vertex, __calculate_sp(*current_front[i].to_lists()))
        for i, vertex in enumerate(start_vertices)
    ]
