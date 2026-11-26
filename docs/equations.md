# Mathematical Formulation

## 1. RAM Covariance-Based SEM (`SEMFlow`)

$$\boldsymbol{\Sigma}(\boldsymbol{\theta}) = \mathbf{F}(\mathbf{I}-\mathbf{A})^{-1}\mathbf{S}_{\mathrm{ram}}(\mathbf{I}-\mathbf{A})^{-\top}\mathbf{F}^\top$$

$$F_{\mathrm{ML}}(\boldsymbol{\theta}) = \ln|\boldsymbol{\Sigma}(\boldsymbol{\theta})| + \operatorname{tr}\!\left(\mathbf{S}\boldsymbol{\Sigma}(\boldsymbol{\theta})^{-1}\right) - \ln|\mathbf{S}| - p$$

$$\Delta\boldsymbol{\eta} = (\mathbf{I}-\mathbf{B})^{-1}\boldsymbol{\Gamma}\Delta\boldsymbol{\xi}$$

## 2. White-Box Causal Deep Learning (`SEMFlow-DL`)

$$\boldsymbol{\eta} = \rho\mathbf{W}\boldsymbol{\eta} + \sum_{j \in \mathrm{pa}(k)} \phi_{kj}(\eta_j) + \boldsymbol{\zeta}, \quad \phi'_{kj}(z) \ge 0$$

$$\hat{C}_{1-\alpha}(\mathbf{x}) = \left[\hat{y}(\mathbf{x}) - \hat{q}_{1-\alpha},\; \hat{y}(\mathbf{x}) + \hat{q}_{1-\alpha}\right]$$
