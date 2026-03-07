---
name: linear-algebra-solver
description: Use this skill when the user wants stronger linear algebra problem solving, step-by-step matrix reasoning, proof scaffolds, or symbolic verification for topics such as linear systems, rank, null space, eigenvalues, eigenvectors, diagonalization, orthogonality, projections, SVD, and quadratic forms.
---

# Linear Algebra Solver

Use this skill for linear algebra questions that benefit from a stable solution routine instead of free-form reasoning.

Keep the answer compact, but do not skip algebraically critical steps.

## Goals

1. Classify the problem before solving.
2. Use the standard method for that problem type.
3. Check the result before presenting it.
4. State clearly when a claim is inferred rather than fully derived.

## Default Workflow

1. Identify the object type.
   - Matrix computation
   - Linear system
   - Subspace / basis / dimension
   - Eigen or diagonalization question
   - Orthogonality / projection
   - SVD or quadratic form
   - Proof-style question

2. Restate the target in mathematical terms.
   - Example: "Find a basis for the null space of A"
   - Example: "Determine whether A is diagonalizable"

3. Choose the standard method.
   - For linear systems, rank, null space, column space: row reduction first.
   - For eigen questions: characteristic polynomial, eigenspaces, multiplicities.
   - For orthogonal projection: inner product formula or projection matrix.
   - For SVD: use eigenstructure of `A^T A` and singular values.
   - For quadratic forms: associate the symmetric matrix and inspect eigenvalues or pivots.

4. Solve in a structured order.
   - Show the setup.
   - Show the key transformation steps.
   - State the final result in exact mathematical form.

5. Verify.
   - Substitute candidate vectors back into the original equation.
   - Check dimensions and ranks.
   - Check orthogonality or normalization when required.
   - If symbolic verification would help, use `scripts/verify_linear_algebra.py`.

## Output Format

Prefer this structure for nontrivial problems:

```text
Problem type
Knowns
Target
Method
Work
Check
Answer
```

For simple computations, a shorter direct answer is fine, but keep the `Check` mentally.

## Problem-Type Routing

- Linear systems / rank / bases / null space:
  Read `references/methods.md`
- Common failure checks:
  Read `references/checklist.md`
- Symbolic verification or exact computation:
  Use `scripts/verify_linear_algebra.py`

## Rules

- Prefer exact arithmetic over decimals unless the user asks for approximations.
- Do not claim a matrix is diagonalizable only from distinct-looking eigenvalues; verify multiplicities.
- Distinguish row space, column space, and null space explicitly.
- When proving a statement, separate the theorem used from the computation.
- If the problem is ambiguous, state the interpretation you are using.
