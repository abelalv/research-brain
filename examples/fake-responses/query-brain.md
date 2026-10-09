The main limitations are **where the physics is enforced** and **what the data can determine**.

First, a PINN only penalises the ODE residual at a finite set of collocation points, so a small training loss does not imply that the trajectory satisfies the SIR equations everywhere [S1]. The deductive check in the vault makes this precise: the claim that the trained network equals the true trajectory is valid only after adding uniqueness of the initial value problem, and it stays questionable in practice because the zero-residual premise is not established [S2].

Second, identifiability: with an unknown reporting fraction $\rho$, early-epidemic data cannot separate $\rho$ from $I(0)$, and the network still returns a value for each parameter [S3, S4].

Third, results are sensitive to the loss weights $\lambda_r, \lambda_0$, and a time-varying $\beta(t)$ was unstable without a smoothness penalty [S5].

**Gaps.** The vault has no note on practical (noise-dependent) identifiability or on adaptive loss weighting. Useful next steps: `rbrain explain "identifiability"` and ingesting a review on loss balancing for PINNs.
