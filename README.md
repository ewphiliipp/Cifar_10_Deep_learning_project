# Human-Model Alignment & Generative Latent Spaces

## 1. Project Objective

The primary goal of this project is to investigate **Human-Model Alignment**—evaluating how closely the internal representations of artificial neural networks mirror human visual perception. By utilizing the **CIFAR-10H** dataset (human soft-labels), we bridge the gap between behavioral data and computational feature spaces. The project further explores **Generative Latent Modeling** via a VAE to understand how these representations can be compressed and reconstructed.

---

## 2. Model Architectures

### PhilNet (Classification & Alignment)

* **Purpose:** A custom CNN optimized to align its categorical boundaries with human visual decisions.
* **Architecture:** Features three blocks (32 → 512 filters), utilizing `BatchNorm2d` for stability and `Dropout` (0.4) to enhance generalization.
* **Training:** Optimized over 50 epochs using `AdamW` with a `CosineAnnealingLR` scheduler.

### Variational Autoencoder (VAE)

* **Purpose:** A generative framework designed to learn a structured, continuous latent space for image synthesis and compression.
* **Encoder:** Maps input images to a Gaussian distribution defined by mean () and standard deviation ().
* **Reparameterization:** Samples a latent vector  to allow backpropagation through stochastic layers.
* **Decoder:** Reconstructs the original image from the latent vector  using transposed convolutions.
* **Optimization:** A dual-objective loss function combining **L1 Reconstruction Loss** (fidelity) and **KL-Divergence** (latent space regularization).

---

## 3. Analytical Methods

### Representational Similarity Analysis (RSA)

RSA serves as the bridge between all models (PhilNet, ResNet, VAE) and human data:

* **RDM (Representational Dissimilarity Matrix):** Quantifies the pairwise "distance" between image representations ().
* **Alignment Results:** PhilNet shows a high Spearman correlation to humans (), while ResNet-50 shows significantly lower alignment ().
* **Latent Analysis:** RSA can be applied to the VAE’s latent space to determine if generative compression preserves human-like categorical structures.

### Error & Perceptual Analysis

* **Confusion & Error Matrices:** We visualize specific misclassifications (e.g., Dog vs. Cat) to identify shared perceptual biases between humans and models.
* **VAE Reconstruction Samples:** Periodic sampling during training (`sample_image`) tracks how well the model captures essential visual features.

---

## 4. Server Workflow & Infrastructure (HPC)

Designed for high-performance computing, the project ensures data safety and efficient resource usage:

* **Process Persistence:** Using **Tmux** to maintain long-running training sessions (100+ epochs) regardless of connection status.
* **Remote Monitoring:** Port forwarding enables local access to **Jupyter Lab** and **TensorBoard** for real-time loss tracking.
* **OOM Prevention:** RSA feature extraction is implemented with **batch-processing** to handle 10,000+ images without exceeding GPU VRAM.
* **Persistent Storage:** Model checkpoints (`.pt`), RDM matrices (`.npy`), and training logs (`.pkl`) are saved automatically to the server's disk.

---

## 5. Conclusion

This project demonstrates a holistic approach to modern AI research:

1. **PhilNet** proves that targeted training can align artificial features with human semantic categories.
2. **RSA** provides the mathematical framework to prove this alignment quantitatively.
3. **The VAE** completes the cycle by demonstrating that these complex visual features can be successfully compressed into a generative latent space while maintaining reconstruction fidelity.

Together, these components show that "human-like" AI is not just about accuracy, but about the structural organization of internal knowledge—whether that knowledge is used to classify a bird or to reconstruct it from a 256-dimensional vector.
