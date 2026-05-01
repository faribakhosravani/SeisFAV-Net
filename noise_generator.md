$$
\begin{align}
&\textbf{REQUIRE: } \text{Clean sample } S \in \mathbb{R}^{H \times W}, \text{ noise standard deviation } \sigma = 5000 \\
&\textbf{ENSURE: } \text{Normalized samples } S_{\text{noisy}}, S_{\text{clean}} \in \mathbb{R}^{H \times W}, \text{ statistics } \mu, \sigma_s \in \mathbb{R} \\
\\
&\textbf{NOISE GENERATION AND NORMALIZATION:} \\
&1: \quad \eta \sim \mathcal{N}(0, \sigma^2 I) \quad \triangleright \text{ Sample Gaussian noise with shape } H \times W \\
&2: \quad S_{\text{noisy}} \leftarrow S + \eta \quad \triangleright \text{ Add noise to clean sample} \\
&3: \quad \mu \leftarrow \frac{1}{HW} \sum_{i=1}^{H} \sum_{j=1}^{W} S(i,j) \quad \triangleright \text{ Compute mean of clean sample} \\
&4: \quad \sigma_s \leftarrow \sqrt{\frac{1}{HW} \sum_{i=1}^{H} \sum_{j=1}^{W} (S(i,j) - \mu)^2} \quad \triangleright \text{ Compute standard deviation of clean sample} \\
&5: \quad S_{\text{clean}} \leftarrow \frac{S - \mu}{\sigma_s} \quad \triangleright \text{ Normalize clean sample} \\
&6: \quad S_{\text{noisy}} \leftarrow \frac{S_{\text{noisy}} - \mu}{\sigma_s} \quad \triangleright \text{ Normalize noisy sample using clean statistics} \\
&7: \quad \textbf{return } S_{\text{noisy}}, S_{\text{clean}}, \mu, \sigma_s
\end{align}
$$
