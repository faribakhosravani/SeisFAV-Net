$$
\begin{align}
&\textbf{REQUIRE: } \text{Clean seismic traces } \{\mathbf{x}_i\}_{i=1}^{N} \in \mathbb{R}^{N \times 1}, \\
&\qquad\qquad \text{Reconstructed traces } \{\hat{\mathbf{x}}_i\}_{i=1}^{N} \in \mathbb{R}^{N \times 1}, \\
&\qquad\qquad \text{Latent distribution parameters } \{\mu_j, \sigma_j^2\}_{j=1}^{d} \text{ from VAE module}, \\
&\qquad\qquad \text{Batch size } N, \text{ latent dimension } d, \\
&\qquad\qquad \text{Hyperparameters } \lambda_1 \text{ (}L_1\text{ weight)}, \beta \text{ (KL weight)} \\
&\textbf{ENSURE: } \text{Total loss } \mathcal{L}_{\text{total}} \text{ for backpropagation} \\
\\
&\textbf{RECONSTRUCTION LOSS (MSE):} \\
&1: \quad \mathcal{L}_{\text{MSE}} \leftarrow 0 \quad \triangleright \text{Initialize mean squared error loss} \\
&2: \quad \textbf{for } i = 1 \text{ to } N \textbf{ do} \\
&3: \quad \quad \mathcal{L}_{\text{MSE}} \leftarrow \mathcal{L}_{\text{MSE}} + \|\mathbf{x}_i - \hat{\mathbf{x}}_i\|_2^2 \quad \triangleright \text{Accumulate squared } L_2 \text{ norm (Equation 17)} \\
&4: \quad \textbf{end for} \\
&5: \quad \mathcal{L}_{\text{MSE}} \leftarrow \mathcal{L}_{\text{MSE}} / N \quad \triangleright \text{Average over batch} \\
\\
&\textbf{SPARSITY REGULARIZATION (}L_1\textbf{):} \\
&6: \quad \mathcal{L}_{L_1} \leftarrow 0 \quad \triangleright \text{Initialize } L_1 \text{ regularization term} \\
&7: \quad \textbf{for } i = 1 \text{ to } N \textbf{ do} \\
&8: \quad \quad \mathcal{L}_{L_1} \leftarrow \mathcal{L}_{L_1} + \|\mathbf{x}_i - \hat{\mathbf{x}}_i\|_1 \quad \triangleright \text{Accumulate } L_1 \text{ norm (Equation 18)} \\
&9: \quad \textbf{end for} \\
&10: \quad \mathcal{L}_{L_1} \leftarrow \mathcal{L}_{L_1} / N \quad \triangleright \text{Average over batch} \\
\\
&\textbf{VAE LATENT REGULARIZATION (KL DIVERGENCE):} \\
&11: \quad \mathcal{L}_{\text{KL}} \leftarrow 0 \quad \triangleright \text{Initialize KL divergence term} \\
&12: \quad \textbf{for } j = 1 \text{ to } d \textbf{ do} \\
&13: \quad \quad \mathcal{L}_{\text{KL}} \leftarrow \mathcal{L}_{\text{KL}} + (1 + \log(\sigma_j^2) - \mu_j^2 - \sigma_j^2) \quad \triangleright \text{Per latent dimension (Equation 19)} \\
&14: \quad \textbf{end for} \\
&15: \quad \mathcal{L}_{\text{KL}} \leftarrow -0.5 \times \mathcal{L}_{\text{KL}} \quad \triangleright \text{Scale for } KL(q\|p) \text{ with } p = \mathcal{N}(0,I) \\
\\
&\textbf{COMPOSITE LOSS COMPUTATION:} \\
&16: \quad \mathcal{L}_{\text{total}} \leftarrow \mathcal{L}_{\text{MSE}} + \lambda_1 \cdot \mathcal{L}_{L_1} + \beta \cdot \mathcal{L}_{\text{KL}} \quad \triangleright \text{Weighted combination (Equation 16)} \\
&17: \quad \textbf{return } \mathcal{L}_{\text{total}} \quad \triangleright \text{Output total loss for gradient descent}
\end{align}
$$

**Key corrections:**
- Equation numbers updated: (15)→(16), (16)→(17), (17)→(18), (18)→(19)
- All underscores and subscripts properly wrapped in math mode
- Consistent notation with $\mathcal{L}_{\text{MSE}}$, $\mathcal{L}_{L_1}$, $\mathcal{L}_{\text{KL}}$, $\mathcal{L}_{\text{total}}$
