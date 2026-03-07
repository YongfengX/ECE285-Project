#!/usr/bin/env python3
"""Exact linear algebra helper based on SymPy."""

from __future__ import annotations

import argparse
import ast
import json
import sys
from typing import Any

try:
    import sympy as sp
except ImportError as exc:  # pragma: no cover - runtime environment dependent
    raise SystemExit(
        "SymPy is required for this helper. Install it with `uv pip install sympy` or `pip install sympy`."
    ) from exc


def parse_matrix(text: str) -> sp.Matrix:
    value = ast.literal_eval(text)
    return sp.Matrix(value)


def parse_vector(text: str) -> sp.Matrix:
    value = ast.literal_eval(text)
    if value and isinstance(value[0], list):
        return sp.Matrix(value)
    return sp.Matrix(value)


def serialize(value: Any) -> Any:
    if isinstance(value, sp.MatrixBase):
        return serialize(value.tolist())
    if isinstance(value, sp.Basic):
        return str(sp.simplify(value))
    if isinstance(value, tuple):
        return [serialize(item) for item in value]
    if isinstance(value, list):
        return [serialize(item) for item in value]
    if isinstance(value, dict):
        return {str(key): serialize(val) for key, val in value.items()}
    return str(sp.simplify(value))


def cmd_rref(args: argparse.Namespace) -> dict[str, Any]:
    matrix = parse_matrix(args.matrix)
    rref_matrix, pivots = matrix.rref()
    return {"rref": rref_matrix, "pivots": list(pivots), "rank": matrix.rank()}


def cmd_nullspace(args: argparse.Namespace) -> dict[str, Any]:
    matrix = parse_matrix(args.matrix)
    basis = matrix.nullspace()
    return {"nullspace_basis": basis, "nullity": len(basis)}


def cmd_colspace(args: argparse.Namespace) -> dict[str, Any]:
    matrix = parse_matrix(args.matrix)
    basis = matrix.columnspace()
    return {"columnspace_basis": basis, "rank": len(basis)}


def cmd_eigen(args: argparse.Namespace) -> dict[str, Any]:
    matrix = parse_matrix(args.matrix)
    eigen_data = []
    for eigenvalue, multiplicity, vectors in matrix.eigenvects():
        eigen_data.append(
            {
                "eigenvalue": eigenvalue,
                "algebraic_multiplicity": multiplicity,
                "basis": vectors,
                "geometric_multiplicity": len(vectors),
            }
        )
    return {"eigen_data": eigen_data}


def cmd_verify_eigenpair(args: argparse.Namespace) -> dict[str, Any]:
    matrix = parse_matrix(args.matrix)
    vector = parse_vector(args.vector)
    eigenvalue = sp.sympify(args.eigenvalue)
    lhs = matrix * vector
    rhs = eigenvalue * vector
    return {"A_times_v": lhs, "lambda_times_v": rhs, "matches": lhs.equals(rhs)}


def cmd_projection(args: argparse.Namespace) -> dict[str, Any]:
    basis_matrix = parse_matrix(args.matrix)
    vector = parse_vector(args.vector)
    if basis_matrix.cols == 1:
        u = basis_matrix
        projection = (vector.dot(u) / u.dot(u)) * u
    else:
        projection = basis_matrix * ((basis_matrix.T * basis_matrix).inv() * basis_matrix.T * vector)
    residual = vector - projection
    return {"projection": projection, "residual": residual}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Exact linear algebra helper")
    subparsers = parser.add_subparsers(dest="command", required=True)

    rref_parser = subparsers.add_parser("rref")
    rref_parser.add_argument("--matrix", required=True, help='Python literal, e.g. "[[1,2],[3,4]]"')
    rref_parser.set_defaults(func=cmd_rref)

    null_parser = subparsers.add_parser("nullspace")
    null_parser.add_argument("--matrix", required=True)
    null_parser.set_defaults(func=cmd_nullspace)

    col_parser = subparsers.add_parser("colspace")
    col_parser.add_argument("--matrix", required=True)
    col_parser.set_defaults(func=cmd_colspace)

    eigen_parser = subparsers.add_parser("eigen")
    eigen_parser.add_argument("--matrix", required=True)
    eigen_parser.set_defaults(func=cmd_eigen)

    verify_parser = subparsers.add_parser("verify-eigenpair")
    verify_parser.add_argument("--matrix", required=True)
    verify_parser.add_argument("--eigenvalue", required=True)
    verify_parser.add_argument("--vector", required=True)
    verify_parser.set_defaults(func=cmd_verify_eigenpair)

    projection_parser = subparsers.add_parser("projection")
    projection_parser.add_argument("--matrix", required=True, help="Single vector or basis matrix")
    projection_parser.add_argument("--vector", required=True, help="Target vector")
    projection_parser.set_defaults(func=cmd_projection)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    result = args.func(args)
    json.dump(serialize(result), sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
