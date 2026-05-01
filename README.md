# SeisFAV-Net: Denoising Framework Architecture

<img width="850" height="496" alt="SeisFAV-Neسیسسسسسt2 drawio" src="https://github.com/user-attachments/assets/de494935-cc66-4b0a-99de-e41d8bb01d72" />



The proposed denoising framework employs an integrated **Fourier Neural Operator Attention U-Net Variational Autoencoder** architecture designed to effectively suppress noise while preserving seismic signal characteristics. The model integrates spectral learning, hierarchical convolutional feature extraction, and probabilistic latent representation within a unified end-to-end framework.

## Input Specification

The input to the network is a normalized one-dimensional seismic trace $x \in \mathbb{R}^{1 \times N}$, where $N$ denotes the number of temporal samples. Prior to training, all traces are standardized using the global mean and standard deviation of the dataset to stabilize optimization and accelerate convergence.

## Architecture Components

The framework utilizes a **Fourier Neural Operator block** to learn long-range frequency dependencies, followed by an **Attention-augmented U-Net encoder-decoder structure** for multi-scale feature representation. 

Within the U-Net bottleneck, a **VAE** maps features to a stochastic latent space ($d=128$), enhancing the model's ability to generalize and regularize noise distributions. 

The decoder incorporates **self-attention modules** and **skip connections** to reconstruct high-fidelity seismic signals via residual learning, where the network specifically predicts the noise component for subtraction from the raw record.
