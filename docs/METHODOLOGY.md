# Theoretical Background: Discrete Diffusion for Language

## From Continuous to Discrete Diffusion

### Diffusion Models for Images vs. Text

Diffusion models generate data by learning to **reverse a corruption process**:
1. **Forward process (training):** Gradually corrupt clean data with noise
2. **Reverse process (inference):** Learn to denoise step by step

For **continuous** data (images), this means adding Gaussian noise. For **discrete** data (text tokens), we need a different corruption mechanism since we can't add continuous noise to categorical variables.

### Absorbing State Diffusion

The key idea: replace tokens with a special `[MASK]` token instead of adding noise.

**Forward process (corruption):**
```
t=0.0: "The quick brown fox jumps over the lazy dog"    ← Clean
t=0.2: "The [M] brown fox jumps [M] the lazy dog"       ← 20% masked
t=0.5: "[M] [M] brown [M] [M] [M] [M] lazy [M]"        ← 50% masked
t=1.0: "[M] [M] [M] [M] [M] [M] [M] [M] [M]"           ← Fully masked
```

**Reverse process (generation):**
```
Step T:  [M] [M] [M] [M] [M] [M] [M] [M] [M]           ← Start fully masked
Step ~3: [M] [M] [M] [M] quick [M] [M] [M] [M]          ← Highest confidence first
Step ~2: The [M] [M] [M] quick brown [M] [M] dog
Step 1:  The quick brown fox jumps over the lazy dog     ← Complete
```

---

## Mathematical Formulation

### Noise Schedule

$$\alpha_t = 1 - t, \quad t \in [0, 1]$$

Where $\alpha_t$ = probability of a token being **unmasked** at time $t$:
- $t = 0$: $\alpha_0 = 1$ (all tokens clean)
- $t = 1$: $\alpha_1 = 0$ (all tokens masked)

### Transition Probability

Each token is independently masked:

$$q(x_t | x_0) = \begin{cases} x_0 & \text{with probability } \alpha_t \\ [\text{MASK}] & \text{with probability } 1 - \alpha_t \end{cases}$$

### Training Objective

$$\mathcal{L} = \mathbb{E}_{t \sim U(0,1)} \left[ \frac{1}{t} \sum_{i \in \text{masked}} \text{CE}\big(f_\theta(x_t)_i,\; x_0^{i+1}\big) \right]$$

Where:
- $t \sim U(0,1)$: timestep sampled uniformly
- $\text{masked}$: set of masked positions at time $t$
- $f_\theta(x_t)_i$: model prediction at position $i$
- $x_0^{i+1}$: target is the **next** token (shift operation for AR compatibility)
- $1/t$: reweighting that emphasizes precision at low noise levels

---

## AR→Diffusion Adaptation (DiffuLLaMA Method)

Rather than training a discrete diffusion model from scratch (which requires trillions of pretraining tokens), DiffuLLaMA (Gong et al., arXiv:2410.17891) showed that pretrained autoregressive models can be converted to diffusion models with three key techniques.

### Technique 1: Attention Mask Annealing

**Problem:** AR models use causal (lower-triangular) attention masks — they have never seen right-side context. Switching to bidirectional attention immediately causes model collapse.

```
Causal (AR):                 Bidirectional (Diffusion):
Q\K: 1  2  3  4  5          Q\K: 1  2  3  4  5
1:   ✓  ✗  ✗  ✗  ✗          1:   ✓  ✓  ✓  ✓  ✓
2:   ✓  ✓  ✗  ✗  ✗          2:   ✓  ✓  ✓  ✓  ✓
3:   ✓  ✓  ✓  ✗  ✗    →     3:   ✓  ✓  ✓  ✓  ✓
4:   ✓  ✓  ✓  ✓  ✗          4:   ✓  ✓  ✓  ✓  ✓
5:   ✓  ✓  ✓  ✓  ✓          5:   ✓  ✓  ✓  ✓  ✓
```

**Solution:** Gradually reveal right-side context over $N$ training steps (10,000 in our experiments):

$$\text{progress} = \min\left(\frac{\text{step}}{N},\; 1\right)$$

The amount of visible right-side context increases linearly from zero (pure causal) to full (bidirectional).

### Technique 2: Shift Operation

**Problem:** Standard denoising predicts token $x_i$ at position $i$. But the AR model was pretrained to predict $x_{i+1}$ from position $i$ (next-token prediction). At unmasked positions, the model would trivially see its own answer.

**Solution:** Shift labels: $\text{target}[i] = \text{input}[i+1]$

```
Input:  [M]   [M]   brown  [M]   [M]
Target: quick brown  fox    jumps over    ← each position predicts NEXT token
```

During inference: prepend a start token, run the model, shift predictions back by one position.

### Technique 3: 1/t Loss Reweighting

At high noise ($t$ near 1, many masks), each individual prediction matters less — there's little context to learn from. At low noise ($t$ near 0, few masks), each prediction must be highly precise because the output is nearly complete.

$$\mathcal{L}_\text{weighted} = \frac{1}{t} \cdot \mathcal{L}_\text{CE}$$

This ensures the model focuses capacity on precise predictions when most text is already revealed — exactly the final denoising steps that determine output quality.

---

## Inference: Iterative Denoising

### Algorithm (from DiffuLLaMA)

```
Input:  Model f_θ, num_steps T, max_tokens N
Output: Generated token sequence

1. x_T = [MASK, MASK, ..., MASK]         (N tokens)

2. For t = T, T-1, ..., 1:
     α_t = 1 - t/T
     α_{t-1} = 1 - (t-1)/T

     x_shifted = [START] + x_t[:-1]       (prepend start token)
     logits = f_θ(x_shifted)               (full bidirectional attention)
     predictions = sample(logits)           (temperature, top-p, top-k)

     For each masked position i:
       p_unmask = (α_{t-1} - α_t) / (1 - α_t)
       if random() < p_unmask:
         x_{t-1}[i] = predictions[i]       (unmask)
       else:
         x_{t-1}[i] = [MASK]               (keep masked)

3. Trim at first EOS token
4. Return decoded text
```

### Key Properties

| Property | Autoregressive | Discrete Diffusion |
|----------|:--------------:|:------------------:|
| Generation order | Left → Right | Confidence-based (anywhere) |
| Context | Left only | Bidirectional |
| Revision | None (no backtracking) | Iterative refinement |
| Speed | O(N) steps with KV cache | O(T) full forward passes |
| Determinism | Temperature-controlled | Inherently stochastic |
| Exact reproduction | Strong | Weak |

---

## Why Diffusion Is Theoretically Attractive for OCR

1. **Degraded text recovery:** When characters are partially obscured, both sides provide context: `"experim_ntal"` can be filled using both "experi" and "ntal"
2. **Error correction:** Iterative refinement allows the model to fix mistakes from earlier steps
3. **Parallel generation:** Multiple tokens decoded per step could yield faster inference

## Why It Fails in Practice

1. **OCR is exact transcription:** Every character must be correct — there's only one valid output. Diffusion's "exploration" of alternatives is harmful.
2. **Structural constraints can't be enforced:** LaTeX syntax (`\frac{}{}`), markdown tables (`| | |`), and reading order require sequential structure that per-token independent sampling cannot guarantee.
3. **Cost without benefit:** 64 forward passes with no KV caching yields 6–50× slower inference, without compensating quality improvements on structured content.

This fundamental tension — between diffusion's exploratory nature and OCR's demand for deterministic precision — is the central finding of this project.
