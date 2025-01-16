# Numerical-Optimization

### Least Squares Problem Using Pseudoinverse, QR Decomposition, and SVD

The least squares problem aims to find the best-fitting solution to a system of linear equations that has no exact solution. Given a design matrix `X` and a target vector `y`, we want to find the coefficients `β` that minimize the residual sum of squares:

min ‖y - Xβ‖₂²


Here’s how to solve the least squares problem using three different methods: **Pseudoinverse**, **QR Decomposition**, and **SVD**.

---

#### 1. Using Pseudoinverse
The pseudoinverse `X⁺` directly solves the least squares problem:
β = X⁺ y
The pseudoinverse is computed using the Moore-Penrose inverse, which can handle non-square and singular matrices.

---

#### 2. Using QR Decomposition
QR decomposition factorizes `X` into an orthogonal matrix `Q` and an upper triangular matrix `R`:
X = Q R
The least squares solution is obtained by solving:
β = R⁻¹ Qᵀ y
This method is numerically stable and efficient for tall matrices (more rows than columns).

---

#### 3. Using SVD (Singular Value Decomposition)
SVD decomposes `X` into three matrices:
X = U Σ Vᵀ
The least squares solution is computed as:
β = V Σ⁺ Uᵀ y
Here, `Σ⁺` is the pseudoinverse of the diagonal matrix of singular values.

---
