# Checklist

Run this checklist before finalizing an answer.

## Frequent Errors

- Pivot columns taken from the reduced matrix instead of the original matrix
- Confusing rank with the number of nonzero entries
- Missing free variables in a null-space basis
- Treating algebraic multiplicity as geometric multiplicity
- Forgetting that orthogonal projection formulas require the right inner product assumptions
- Using decimal approximations too early and breaking exact structure
- Claiming independence without a determinant, reduction, or theorem
- Forgetting to verify candidate eigenvectors satisfy the original equation

## Fast Sanity Checks

- Dimensions match at every multiplication
- Basis vectors are linearly independent
- Number of basis vectors matches the claimed dimension
- `rank(A) + nullity(A) = number of columns of A`
- Projection residual is orthogonal to the target subspace
- For diagonalization, `A = P D P^{-1}` reproduces the original matrix
- For SVD, singular values are nonnegative and ordered if an ordered SVD is requested

## When to Use the Script

Use `scripts/verify_linear_algebra.py` when:

- exact symbolic row reduction is useful
- you want to verify eigenpairs
- you want a null-space or column-space basis
- you want an exact projection or determinant
- the hand derivation is correct but error-prone
