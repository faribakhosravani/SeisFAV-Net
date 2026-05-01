$$
\begin{align}
&\textbf{REQUIRE: } \text{Noisy input } x_{\text{noisy}} \in \mathbb{R}^{C \times N}, \text{ kernel size } k = 3, \text{ stride } s = 2, \\
&\qquad\qquad\text{encoder depth } L = 3, \text{ initial channels } C_0 \\
&\textbf{ENSURE: } \text{Denoised output } \hat{x} = x_{\text{noisy}} - \hat{n} \text{ via noise residual prediction} \\
\\
&\textbf{ENCODER (Hierarchical Feature Extraction):} \\
&1: \quad h_1 \leftarrow x_{\text{noisy}} \quad \triangleright \text{ Input: } N \text{ temporal samples} \\
&2: \quad \textbf{for } l = 1 \text{ to } L \textbf{ do} \quad \triangleright \text{ Three residual blocks with downsampling} \\
&3: \quad\quad h_l \leftarrow \text{Conv}(h_l, k=3) \to \text{BN} \to \text{ReLU} \quad \triangleright \text{ First convolutional layer} \\
&4: \quad\quad h_l \leftarrow \text{Conv}(h_l, k=3) \to \text{BN} \to \text{ReLU} \quad \triangleright \text{ Second convolutional layer (residual block)} \\
&5: \quad\quad e_l \leftarrow h_l \quad \triangleright \text{ Store encoder features for skip connections} \\
&6: \quad\quad h_{l+1} \leftarrow \text{MaxPool}(h_l, \text{stride}=2) \quad \triangleright \text{ Downsample: } N \to N/2 \to N/4 \\
&7: \quad \textbf{end for} \\
&8: \quad h_{\text{bottleneck}} \leftarrow h_{L+1} \quad \triangleright \text{ Bottleneck features at resolution } N/2^L \\
\\
&\textbf{DECODER (Multi-Scale Reconstruction):} \\
&9: \quad \textbf{for } l = L \text{ down to } 1 \textbf{ do} \quad \triangleright \text{ Upsampling with skip connections} \\
&10: \quad\quad h_l \leftarrow \text{TransposedConv}(h_{l+1}, \text{stride}=2) \quad \triangleright \text{ Upsample: } N/4 \to N/2 \to N \\
&11: \quad\quad h_l \leftarrow \text{Concat}(h_l, e_l) \quad \triangleright \text{ Concatenate with encoder features} \\
&12: \quad\quad \textbf{if } l = L \textbf{ then} \quad \triangleright \text{ Apply self-attention at first decoder block} \\
&13: \quad\quad\quad h_l \leftarrow \text{SelfAttention}(h_l) \quad \triangleright \text{ Capture global dependencies for reflector continuity} \\
&14: \quad\quad \textbf{end if} \\
&15: \quad\quad h_l \leftarrow \text{Conv}(h_l, k=3) \to \text{BN} \to \text{ReLU} \quad \triangleright \text{ First convolutional layer} \\
&16: \quad\quad h_l \leftarrow \text{Conv}(h_l, k=3) \to \text{BN} \to \text{ReLU} \quad \triangleright \text{ Second convolutional layer (residual block)} \\
&17: \quad \textbf{end for} \\
\\
&\textbf{NOISE RESIDUAL PREDICTION:} \\
&18: \quad \hat{n} \leftarrow \text{Conv}(h_1, k=1\times1) \quad \triangleright \text{ } 1\times1 \text{ convolution predicts noise component} \\
&19: \quad \hat{x} \leftarrow x_{\text{noisy}} - \hat{n} \quad \triangleright \text{ Residual subtraction} \\
&20: \quad \textbf{return } \hat{x} \quad \triangleright \text{ Denoised seismic trace} \\
\\
&\textbf{RESIDUAL LEARNING FORMULATION: } \hat{x} = x_{\text{noisy}} - \hat{n}
\end{align}
$$
