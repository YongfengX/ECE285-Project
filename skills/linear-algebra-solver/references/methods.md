# Methods

Use the smallest method that fully solves the problem.

## Linear Systems

1. Form the augmented matrix.
2. Row reduce to echelon or reduced echelon form.
3. Identify pivots and free variables.
4. Express the solution set parametrically.
5. Check by substitution.

## Rank, Null Space, Column Space, Row Space

- `rank(A)`: count pivots after row reduction.
- `null(A)`: solve `Ax = 0` using free variables.
- `col(A)`: choose pivot columns from the original matrix, not the reduced one.
- `row(A)`: nonzero rows of an echelon form give a basis.
- Use rank-nullity when dimension bookkeeping matters.

## Basis and Dimension

1. Test spanning or independence with row reduction.
2. Remove dependent vectors only after showing dependence.
3. State both a basis and the dimension.

## Eigenvalues and Eigenvectors

1. Solve `det(A - lambda I) = 0`.
2. For each eigenvalue, solve `(A - lambda I)x = 0`.
3. Compare algebraic and geometric multiplicities.
4. For diagonalizability, require enough linearly independent eigenvectors.

## Orthogonality and Projection

- Orthogonal projection of `b` onto nonzero `u`:
  `proj_u(b) = (b · u / u · u) u`
- For projection onto a subspace with orthonormal columns `Q`:
  `proj(b) = QQ^T b`
- If the basis is not orthonormal, orthonormalize first or use the normal equations.

## Gram-Schmidt

1. Process vectors in order.
2. Subtract projections onto previously built orthogonal vectors.
3. Normalize only if an orthonormal basis is requested.
4. Check that no produced vector is zero unless the input set is dependent.

## SVD

1. Compute `A^T A`.
2. Find its eigenvalues and eigenvectors.
3. Singular values are the square roots of the nonnegative eigenvalues.
4. Build right singular vectors from eigenvectors of `A^T A`.
5. Build left singular vectors from `u_i = A v_i / sigma_i` for nonzero singular values.
6. Verify shapes and ordering.

## Quadratic Forms

1. Write the form as `x^T A x` with `A` symmetric.
2. Determine definiteness by eigenvalues, Sylvester's criterion, or pivots if justified.
3. State whether the form is positive definite, semidefinite, indefinite, or negative definite.

## Proof-Style Questions

Use this pattern:

1. State the theorem or definition.
2. Map the theorem assumptions to the given data.
3. Perform the required computation or logical implication.
4. Conclude in one sentence that directly answers the claim.
