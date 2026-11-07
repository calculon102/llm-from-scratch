# Phase 1 — XOR: the smallest non-linear learning problem

This note explains the math behind the M1 XOR phase: the dataset, the
2-layer MLP, the loss, the gradients that training follows, and the ways
the experiment can fail. It is the conceptual companion to:

- `core/xor/dataset.py` — the four truth-table rows, repeated and
  optionally noised.
- `core/xor/model.py` — `XorMLP` (`Linear → activation → Linear`).
- `core/xor/__init__.py` — `xor_loss` (binary cross-entropy with logits),
  the phase config, and the phase factory.
- `core/trainer.py` — the shared training/evaluation/checkpoint loop.
- `core/gradcheck.py` — the finite-difference check that verifies the
  gradients derived below.
- `core/xor/plots.py` — decision boundary, hidden activations, loss curve.

The phase is deliberately tiny. It is the first end-to-end exercise of the
machinery that later phases reuse (forward pass, loss, backpropagation,
optimizer, validation, checkpointing, artifacts), on a problem where every
number can be checked by hand.

---

## 1. The problem

The XOR dataset has four input/label pairs:

| $x_1$ | $x_2$ | $y = x_1 \oplus x_2$ |
|---:|---:|---:|
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 0 |

Let $\mathbf{x} = (x_1, x_2)^\top \in \{0,1\}^2$ and $y \in \{0,1\}$.
The target is the parity function: $y = 1$ when the inputs differ, and
$y = 0$ when they are equal.

**Why XOR is the canonical "you need a hidden layer" example.** A single
linear layer computes

$$
f(\mathbf{x}) = w_1 x_1 + w_2 x_2 + b,
$$

and classifies by the sign of $f$. Its decision boundary is one straight
line. Suppose such a line separated the classes, with

$$
f(0,0)<0,\quad f(1,1)<0,\quad f(0,1)>0,\quad f(1,0)>0.
$$

Writing the four inequalities out gives

$$
b<0,\qquad w_1+w_2+b<0,\qquad w_2+b>0,\qquad w_1+b>0.
$$

Adding the last two yields $w_1+w_2+2b>0$, hence
$w_1+w_2+b>-b>0$ (because $b<0$), which contradicts
$w_1+w_2+b<0$. No line can do it. Equivalently, the two class-1 points
lie on one diagonal and the two class-0 points on the other; any line
that keeps the class-1 diagonal on one side leaves at least one class-0
point on that same side.

The best a single linear model can do is essentially predict the
constant $p=1/2$ everywhere, with loss $\ln 2 \approx 0.693$. That is
also the loss of a randomly initialized network that outputs a logit of
zero, which is why $\ln 2$ is the diagnostic starting point of every
XOR loss curve.

---

## 2. The model: a 2-layer MLP

`XorMLP` is

$$
\mathbf{x} \in \mathbb{R}^{2}
\;\xrightarrow{\;\mathbf{W}_1,\mathbf{b}_1\;}\;
\mathbf{a} \in \mathbb{R}^{H}
\;\xrightarrow{\;\varphi\;}\;
\mathbf{h} \in \mathbb{R}^{H}
\;\xrightarrow{\;\mathbf{w}_2,b_2\;}\;
z \in \mathbb{R},
$$

with equations

$$
\mathbf{a} = \mathbf{W}_1\mathbf{x} + \mathbf{b}_1,
\qquad
\mathbf{h} = \varphi(\mathbf{a}),
\qquad
z = \mathbf{w}_2^\top \mathbf{h} + b_2.
$$

Shapes:

- $\mathbf{W}_1 \in \mathbb{R}^{H \times 2}$, $\mathbf{b}_1 \in \mathbb{R}^{H}$;
- $\mathbf{w}_2 \in \mathbb{R}^{H}$, $b_2 \in \mathbb{R}$;
- $H$ is `phase_config.hidden_size` (default **8**);
- $\varphi$ is `tanh` (default) or `relu` (`phase_config.activation`).

The output $z$ is a **logit**, not a probability. The probability is
obtained with the sigmoid

$$
p = \sigma(z) = \frac{1}{1+e^{-z}},
\qquad
\sigma^{-1}(p) = \log\frac{p}{1-p}.
$$

A positive logit means "lean toward class 1"; a negative logit means
"lean toward class 0". The Trainer's `xor_loss` deliberately uses
`BCEWithLogitsLoss`, which combines the sigmoid and the log-loss in one
numerically stable operation (see §3).

The parameter count is

$$
\underbrace{2H}_{\mathbf{W}_1}
+ \underbrace{H}_{\mathbf{b}_1}
+ \underbrace{H}_{\mathbf{w}_2}
+ \underbrace{1}_{b_2}
= 4H+1.
$$

For the default $H=8$ that is **33 weight elements**, which is why the
finite-difference gradient check (two forward passes per element) is
cheap here.

### 2.1 What each layer does, geometrically

Each hidden unit computes an affine score
$a_j = \mathbf{w}_{1,j}^\top \mathbf{x} + b_{1,j}$.
The equation $a_j = 0$ is a line in the input plane; the sign of $a_j$
says which side of that line the point is on.
The activation then bends the score:

$$
\tanh(a) = \frac{e^{a}-e^{-a}}{e^{a}+e^{-a}},
\qquad
\operatorname{ReLU}(a) = \max(0,a).
$$

The output layer draws **one straight line in the hidden feature space**
$\mathbf{h}$:

$$
\mathbf{w}_2^\top \mathbf{h} + b_2 = 0.
$$

So the network's decision boundary in the original input plane is the
preimage of a line under the non-linear feature map
$\mathbf{x} \mapsto \mathbf{h}(\mathbf{x})$. The feature map folds the
plane, and the fold is what lets a straight line in hidden space become
a checkerboard boundary in input space.

With two ReLU units there is a particularly clean solution:

$$
\mathbf{W}_1 =
\begin{pmatrix}
1 & -1\\
-1 & 1
\end{pmatrix},
\quad
\mathbf{b}_1 = \mathbf{0},
\qquad
\mathbf{w}_2 = \begin{pmatrix}1\\1\end{pmatrix},
\quad
b_2 = -\tfrac12.
$$

Then

$$
h_1 = \operatorname{ReLU}(x_1-x_2), \qquad
h_2 = \operatorname{ReLU}(x_2-x_1),
$$

and

$$
z = h_1+h_2-\tfrac12 = |x_1-x_2|-\tfrac12.
$$

On the four XOR points:

| $(\mathbf{x})$ | $h_1$ | $h_2$ | $z$ | $\sigma(z)$ | class |
|---|---:|---:|---:|---:|---:|
| (0,0) | 0 | 0 | -0.5 | 0.378 | 0 |
| (0,1) | 0 | 1 | 0.5 | 0.622 | 1 |
| (1,0) | 1 | 0 | 0.5 | 0.622 | 1 |
| (1,1) | 0 | 0 | -0.5 | 0.378 | 0 |

The threshold at $p=0.5$ (equivalently $z=0$) classifies all four
points correctly. The probabilities are not yet extreme because the
weights are small; gradient descent can scale $\mathbf{w}_2$ up to
sharpen them without changing the decision boundary. In hidden space the
two class-0 points both map to $(0,0)$, while the class-1 points map to
$(0,1)$ and $(1,0)$ — now the classes **are** linearly separable, by
$h_1+h_2=\tfrac12$. That is the whole trick: the hidden layer learns a
representation in which the problem is linear.

The default $H=8$ is not required mathematically (two units suffice),
but it gives the optimizer spare capacity and redundancy, making success
much less dependent on the random seed. Extra units can specialize on the
same half-planes or on useful margins without preventing the others from
solving the task.

---

## 3. The loss: binary cross-entropy

For one example with label $y\in\{0,1\}$ and logit $z$, the loss is

$$
\ell(z,y) = -y\log\sigma(z) - (1-y)\log\bigl(1-\sigma(z)\bigr).
$$

This is the negative log-likelihood of a Bernoulli model: if the network
assigns probability $p_y$ to the true label, the loss is $-\log p_y$,
the "surprise" of the observed label. A confident correct prediction pays
almost nothing; a confident wrong prediction pays a large penalty; an
unsure prediction ($p=0.5$) pays a fixed $\ln 2$.

Using $\sigma(z)=1/(1+e^{-z})$, the expression can be rewritten in the
numerically stable **log-sum-exp / softplus** form:

$$
\ell(z,y)
= \log\bigl(1+e^{z}\bigr) - y z
= \operatorname{softplus}(z) - yz.
$$

For $y=1$ this is $\log(1+e^{-z})$; for $y=0$ it is $\log(1+e^{z})$.
Both are evaluated without ever computing $\log(0)$ or overflowing a
sigmoid, which is exactly what `F.binary_cross_entropy_with_logits` does.

For a batch of $B$ examples the training objective is the mean

$$
L = \frac{1}{B}\sum_{b=1}^{B}\Bigl[\operatorname{softplus}(z_b)
- y_b z_b\Bigr].
$$

At initialization the weights are small and random, so the logits are
near zero and the loss is near $\ln 2$ per example (hence
$\ln 2 \approx 0.693$). A loss curve that starts far above $\ln 2$ is
a sign of badly scaled initialization or a bug; one that starts exactly
at $\ln 2$ and stays there is a sign that nothing is learning.

---

## 4. Backpropagation: the gradients training follows

Training needs, for every parameter, the partial derivative of the loss
with respect to that parameter. Autograd computes them with the chain
rule; here is the same derivation explicitly, because this is the math
the finite-difference check verifies.

For **one** example, define the probability error

$$
\delta \equiv \frac{\partial \ell}{\partial z}
= \sigma(z)-y = p-y.
$$

This single scalar is the entire "error signal" at the output. Then

$$
\frac{\partial \ell}{\partial \mathbf{w}_2} = \delta\,\mathbf{h},
\qquad
\frac{\partial \ell}{\partial b_2} = \delta,
$$

and the error propagated back to the hidden pre-activations is

$$
\boldsymbol{\delta}_1
\equiv \frac{\partial \ell}{\partial \mathbf{a}}
= \frac{\partial \ell}{\partial \mathbf{h}} \odot \varphi'(\mathbf{a})
= \delta\,\mathbf{w}_2 \odot \varphi'(\mathbf{a}),
$$

where $\odot$ is the element-wise product. Finally,

$$
\frac{\partial \ell}{\partial \mathbf{W}_1}
= \boldsymbol{\delta}_1\,\mathbf{x}^\top,
\qquad
\frac{\partial \ell}{\partial \mathbf{b}_1}
= \boldsymbol{\delta}_1.
$$

For a batch, each of these is averaged over the $B$ examples (the mean
and the sum differ only by the constant factor $1/B$, which the
learning rate absorbs).

The activation derivatives are

$$
\varphi'(a) =
\begin{cases}
1-\tanh^2(a), & \varphi=\tanh,\\
\mathbf{1}[a>0], & \varphi=\operatorname{ReLU}.
\end{cases}
$$

A useful sanity check: if the prediction is already correct and confident
($p\to y$), then $\delta\to 0$, so every gradient goes to zero and the
weights stop moving. If the prediction is confidently wrong, $\delta$
is large and the update is large.

### 4.1 The optimizer

Plain gradient descent would update every parameter as

$$
\theta \leftarrow \theta - \eta\,\frac{\partial L}{\partial \theta},
$$

where $\eta$ is the learning rate (`learning_rate`, default **0.05**
for the XOR tiny profile). The shared Trainer instead uses **AdamW**,
which keeps a running first moment $m_t$ and second moment $v_t$ of
each gradient and normalizes the step per parameter:

$$
m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t,
\qquad
v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2,
$$

$$
\hat m_t = \frac{m_t}{1-\beta_1^t},
\qquad
\hat v_t = \frac{v_t}{1-\beta_2^t},
$$

$$
\theta_t = \theta_{t-1}
- \eta\,\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}
- \eta\lambda\,\theta_{t-1}.
$$

The last term is **decoupled weight decay** ($\lambda=$
`weight_decay`, PyTorch's default $0.01$ here); it is not the same as
L2 regularization inside the gradient, which is why `AdamW` is used
rather than `Adam`. Adam's normalization means the effective step size is
roughly $\eta$ per parameter even when gradient magnitudes differ by
orders of magnitude — convenient, but it also means a too-large $\eta$
can still blow up the weights.

The Trainer also clips the global gradient norm to `grad_clip = 1.0`
before the optimizer step. That is a safety net for noisy or
pathological batches, not a substitute for a sane learning rate.

### 4.2 Why random initialization matters

`nn.Linear` initializes weights randomly (PyTorch uses Kaiming-uniform
weights and small uniform biases). This is not cosmetic:

1. **Symmetry breaking.** If all hidden units started with identical
   weights, they would receive identical gradients and remain identical
   forever. Randomness lets units differentiate and specialize on
   different half-planes.
2. **A starting point.** Gradient descent is a local procedure; it walks
   downhill from wherever it starts. Different initializations can reach
   different solutions (different placements of the boundary), all of
   which may fit XOR.

Because of (2), "learns XOR reliably" is a statement about a distribution
of seeds and configurations, not a guarantee for every single run. The
default $H=8$ and the seed in the config are chosen to make success
robust.

### 4.3 Finite-difference verification

Autograd is fast but still code; a transposed matrix or a missing term
in a hand-written backward pass would silently produce wrong gradients.
The gradient check compares the analytic gradient with a numerical
estimate obtained from the definition of the derivative:

$$
g_i^{\text{num}}
= \frac{L(\theta_i+\epsilon)-L(\theta_i-\epsilon)}{2\epsilon}
\approx \frac{\partial L}{\partial \theta_i}.
$$

This uses only forward passes and no chain rule. The two estimates are
compared element-wise with

$$
\text{rel\_err}
= \frac{|g_i^{\text{analytic}}-g_i^{\text{num}}|}
{\max\bigl(1,\,|g_i^{\text{analytic}}|,\,|g_i^{\text{num}}|\bigr)},
$$

and the check passes when every element is at or below `rtol = 1e-3`.
The normalization by $\max(1,\cdot)$ makes the test absolute for tiny
gradients, where relative error is meaningless.

The step size $\epsilon$ trades off two errors: truncation error
($\propto \epsilon^2$ for the central difference) and floating-point
round-off error ($\propto \epsilon_{\text{machine}}/\epsilon$). The
sweet spot scales like $\epsilon_{\text{machine}}^{1/3}$. Float32
has about seven decimal digits, so the default `eps = 1e-3` is used; the
check runs on a CPU float32 copy of the model because fp16/bf16 have too
few mantissa bits for the difference of two nearly equal losses to be
meaningful.

In the tiny profile `gradcheck: true` runs the check **before training**,
writes `gradcheck_report.txt` into the run folder, and logs PASS/FAIL.
It is a diagnostic, not a gate: training continues even on FAIL so you
can observe what wrong gradients do.

---

## 5. Intuition: what training actually does

Put the pieces together. At initialization the four points produce
roughly the same small logit, the probabilities are near $1/2$, and the
loss is near $\ln 2$. The gradient $\delta=p-y$ tells every weight
which way to move to make the current batch's predictions less
surprising:

- a point with $y=1$ but $p<1$ pushes the logit up;
- a point with $y=0$ but $p>0$ pushes the logit down;
- because the same hidden units feed all four points, the weights must
  find a representation that satisfies all of them at once — the hidden
  layer is forced to become a feature detector rather than memorizing
  individual outputs.

Over many steps the hidden units rotate their half-planes and the output
layer combines them. In the ReLU picture from §2.1, the network
discovers something like "one unit fires when $x_1>x_2$, the other
when $x_2>x_1$"; their sum is $|x_1-x_2|$, which is exactly XOR on
the grid. With tanh the same geometry appears as two smooth saturating
half-planes whose difference carves out the two diagonal corners. The
black $p=0.5$ contour in `plots/decision_boundary.png` should end up
between the class-0 diagonal and the class-1 diagonal; the panels in
`plots/activations.png` should show individual units responding to
different half-planes, not all looking identical.

The loss curve should start near $0.693$ and fall toward zero. The
validation curve (one point per epoch) should follow it. Note the
caveat in §6: in this phase the validation points are repeats of the
same four locations, so the val curve measures fit, not true
generalization to unseen inputs.

---

## 6. Failure modes and how to diagnose them

The table lists symptoms, likely causes, and first fixes. Most are
visible in the loss curve, the accuracy metric, the plots, or the
gradient-check report.

| Symptom | Likely cause | What to do |
|---|---|---|
| Loss stays at $\ln 2\approx0.693$, accuracy $\approx0.5$ | No hidden layer, `hidden_size: 1`, a linear activation, `learning_rate: 0`, or a broken backward pass | Use $H\ge2$; check `phase_config`; run with `gradcheck: true`; inspect the report's worst parameter |
| Loss falls then plateaus well above zero; one diagonal always wrong | Too few hidden units (`H=2`) for this seed; tanh saturation; dead ReLU units; learning rate too high | Increase `hidden_size` (default 8), lower `learning_rate`, try `activation: relu` or `tanh`, change `seed` |
| Loss oscillates or becomes `NaN` / `inf` | Learning rate too large; inputs or initialization badly scaled; exploding activations | Lower `learning_rate` (try 0.01), keep gradient clipping, check the loss is finite after the first steps |
| Loss decreases extremely slowly | Learning rate too small; saturated tanh; dead ReLU units | Raise `learning_rate`; check activation histograms/plots; switch activation |
| Training loss falls, validation loss rises | Overfitting — but on XOR with repeats this usually means noise or a very small val split; check whether `noise_std` is too large | Reduce `noise_std`, increase `repeats`, or treat the val curve as a fit check rather than a generalization estimate |
| Val loss is high while train accuracy is perfect | The val split is drawn from the same repeated points, so this is unusual; if it happens, inspect the split and the accuracy computation | Verify `val_split` and that labels are not shuffled independently of inputs |
| `gradcheck` reports FAIL | A real backward-pass bug (wrong transpose, half gradient, missing bias term) **or** a poorly chosen $\epsilon$ / reduced precision | Start with the `worst_param` in `gradcheck_report.txt`; re-run with the CPU float32 check; remember the report is a diagnostic, not a training gate |
| Gradients look fine but the model still fails | Optimization issue, not math: bad learning rate, saturated units, insufficient hidden units, unlucky seed | Try the fixes above; compare several seeds before concluding the task is unsolvable |
| Loss is lower than plausible and accuracy is suspiciously perfect | Data leakage or duplicate points between train and val | Remember that `XORDataset` repeats the same four points; do not interpret val accuracy as generalization |
| Decision-boundary plot looks blank or constant | Model outputs a constant logit; plotting applies sigmoid to logits, so a constant logit gives a constant probability field | Check the loss curve and accuracy first; the model has not learned anything |
| Plots are missing | `matplotlib` (the optional `viz` extra) is not installed | `pip install -e ".[viz]"`; training itself still succeeds and logs a warning |

A few of these deserve emphasis.

**Why "stuck at $\ln 2$" is the signature failure.** With balanced
classes and a near-zero logit, $p=1/2$ for every point, so every
example contributes $\ln 2$. A model that has learned nothing produces
exactly this number. If the loss is not moving, the first question is
whether gradients are flowing at all; the gradient check answers that
directly.

**Why `tanh` can saturate.** $\tanh'(a)=1-\tanh^2(a)$ is largest at
$a=0$ and vanishes as $|a|\to\infty$. If a large learning rate pushes
pre-activations far from zero, the hidden units saturate, their gradients
shrink, and learning stalls. A smaller learning rate or ReLU usually
fixes it. ReLU has the opposite failure mode: a unit whose pre-activation
is negative for every input has zero gradient forever ("dead ReLU").
Extra hidden units make both failure modes less likely to be fatal,
because the network can route around a saturated or dead unit.

**Why two hidden units are the minimum, not the recommendation.** The
construction in §2.1 proves that two ReLU units are sufficient in
principle. It does not prove that gradient descent from a random start
will find that construction. With $H=2$ the loss landscape is narrow and
seed-dependent; with $H=8$ there are many redundant solutions and the
optimizer is far more likely to find one. This is a small-scale preview
of a general lesson: overparameterization makes optimization easier even
when it is not needed for representation.

**Why the validation split is not a generalization test here.** The
dataset is four points repeated `repeats` times; a random 25% split
contains more copies of the same four points. If the model fits the
training copies, it will fit the validation copies almost by
construction. The val curve is still useful — it catches divergence and
gross overfitting to noise — but it cannot tell you whether the model
would classify a new input near the decision boundary correctly. That
question requires genuinely held-out points, which this toy phase does
not have.

**Why the gradient check can fail for non-bug reasons.** Finite
differences are an estimate. Too small an $\epsilon$ turns the
subtraction into round-off noise; too large an $\epsilon$ lets
curvature contaminate the slope; half precision cannot resolve the
difference at all. That is why the check fixes float32 on CPU and uses
$\epsilon=10^{-3}$. A borderline FAIL with errors near the threshold
may be numerical; an O(1) mismatch localized to one parameter is almost
always a real bug in the backward path through that parameter.

---

## 7. What success looks like

With the checked-in `configs/xor_tiny_debug.yaml` (seed 42, $H=8$,
`tanh`, learning rate 0.05, batch size 32, 50 epochs, 400 samples,
25% validation, gradient check enabled):

- 400 samples split into 300 train / 100 validation;
- 10 batches per epoch, so 50 epochs = 500 optimizer steps;
- the loss starts near $\ln 2\approx0.693$, falls toward zero, and the
  reported accuracy reaches 1.0;
- `gradcheck_report.txt` reports **PASS** for all 4 parameters (33 weight
  elements) with a worst relative error on the order of $10^{-4}$;
- `plots/decision_boundary.png` shows the two class-0 corners on one side
  of the black $p=0.5$ contour and the two class-1 corners on the
  other;
- `plots/activations.png` shows hidden units responding to different
  half-planes.

An observed run with these settings reached a final validation loss of
about $0.0021$ with validation accuracy $1.0$, and the gradient check
passed with a worst relative error of $7.3\times10^{-5}$ — consistent
with the analysis above (the remaining loss is small but non-zero because
the probabilities are sharpened, not perfectly saturated).

To reproduce:

```bash
python -m scripts.train --config configs/xor_tiny_debug.yaml
```

The run folder under `artifacts/` contains `config.yaml`, `run.log`,
`checkpoints/` (`step_*.pt` plus `final.pt`), `gradcheck_report.txt`, and
the decision-boundary/activation plots. The loss curve
(`plots/loss_curve.png`) is produced by the UI job runner (T-104), which
also streams the per-step history; the CLI path uses the phase's `on_end`
callback for the model plots.

To watch the same run in the browser:

```bash
python -m scripts.ui
# open http://127.0.0.1:8080/xor
```

---

## 8. Summary of the equations

For reference, the whole phase in one place:

$$
\begin{aligned}
\mathbf{a} &= \mathbf{W}_1\mathbf{x}+\mathbf{b}_1, &
\mathbf{h} &= \varphi(\mathbf{a}), &
z &= \mathbf{w}_2^\top\mathbf{h}+b_2,\\
p &= \sigma(z), &
L &= \frac{1}{B}\sum_b\bigl[\operatorname{softplus}(z_b)-y_b z_b\bigr],\\
\frac{\partial L}{\partial z} &= p-y, &
\frac{\partial L}{\partial \mathbf{w}_2} &= (p-y)\mathbf{h}, &
\frac{\partial L}{\partial b_2} &= p-y,\\
\boldsymbol{\delta}_1 &= (p-y)\,\mathbf{w}_2\odot\varphi'(\mathbf{a}), &
\frac{\partial L}{\partial \mathbf{W}_1} &= \boldsymbol{\delta}_1\mathbf{x}^\top, &
\frac{\partial L}{\partial \mathbf{b}_1} &= \boldsymbol{\delta}_1,\\
\theta &\leftarrow \theta-\eta\,\frac{\hat m}{\sqrt{\hat v}+\epsilon}
-\eta\lambda\theta, &
g_i^{\text{num}} &= \frac{L(\theta_i+\epsilon)-L(\theta_i-\epsilon)}{2\epsilon}.
\end{aligned}
$$
