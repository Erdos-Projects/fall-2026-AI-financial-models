# Literature and architecture rationale

Source access date: October 22, 2026.

This companion explains the architecture choices for the Nifty next-minute
option log-return study. The measured results, training protocol, and runtime
comparisons belong in the main study report. The statements about expected
behavior below are hypotheses informed by model structure and prior research;
they are not experimental findings for this dataset.

## Why start with MLPs?

The prepared input is one vector of 15 engineered features derived from six
consecutive bars. A multilayer perceptron (MLP) directly consumes this tabular
representation and predicts a signed scalar return. It provides a relatively
small, easily reproducible nonlinear benchmark.

Gorishniy et al. identify ordinary MLPs and ResNet-like networks as useful
tabular baselines, compare them under common evaluation protocols, and find no
universally superior solution across deep learning and boosted trees. Their
MLP uses Linear–ReLU–Dropout blocks; their tabular residual network combines
normalization, dense layers, dropout, and skip connections. These families
motivate our comparison. Their benchmark results do not establish performance
on Nifty options. [Gorishniy et al., 2021][tabular]

## Architectures included in this study

All models have 15 inputs and one linear output. Hidden-layer notation below
omits the input and output dimensions. Dropout probability is 0.1 unless stated
otherwise. **The exact widths, depths, and dropout rates are project choices,
not exact replications of architectures from the cited papers.**

| Model | Hidden structure | Question being examined |
| --- | --- | --- |
| Compact | `32 → 16`, ReLU and dropout | Is the existing small neural benchmark sufficient? |
| Standard | `64 → 32`, ReLU and dropout | Does moderate additional capacity help? |
| Wide | `128 → 64 → 32`, ReLU and dropout | Does a larger network improve forecasting enough to justify its cost? |
| Shallow | `64`, ReLU and dropout | Is one hidden layer sufficient for these engineered features? |
| Standard, plain | `64 → 32`, ReLU, no dropout | Does dropout help when hidden widths are held fixed? |
| Deep | `64 → 64 → 64 → 64`, ReLU and dropout | Does additional nonlinear composition improve performance? |
| Batch-normalized | `64 → 32`, each block Linear–BatchNorm–ReLU–Dropout | Does minibatch normalization help the standard network? |
| Residual | `15 → 64` ReLU stem, two width-64 residual blocks, linear head | Can skip connections and per-example normalization support a useful deeper representation? |

Each residual block computes
`ReLU(h + Linear64(Dropout(ReLU(Linear64(LayerNorm(h))))))`.
The residual design is an adaptation of the residual-MLP family: it uses
LayerNorm and its own activation placement. The published tabular ResNet in
Gorishniy et al. uses BatchNorm, so our residual model should not be described
as a faithful reproduction of that model. [Gorishniy et al., 2021][tabular]

The standard/plain pair isolates the presence of dropout within an otherwise
matched architecture. The standard/batch-normalized pair tests adding
BatchNorm. Other comparisons change several properties, including depth,
width, normalization, or parameter count; they compare practical architecture
choices rather than isolating one causal effect.

## Expected performance, training cost, and inference cost

**Width and depth.** Wider layers increase the number of learned interactions
and matrix operations. More hidden layers add nonlinear composition but also
add sequential computation and can make optimization harder. A dense layer
with input width `a` and output width `b` has `a*b + b` learned parameters and
approximately `a*b` multiply-accumulate operations per example. These are
structural cost estimates; measured latency also depends on batch size,
hardware, software, and memory movement. Greater capacity does not guarantee
better chronological validation performance.

**Dropout.** Randomly suppressing hidden units during training is a standard
regularization approach. It may help when correlated examples permit
overfitting, but can also limit useful capacity. Ordinary dropout is disabled
for deterministic inference and adds no learned parameters, so the matched
plain/dropout networks have essentially the same dense inference work.
[Srivastava et al., 2014][dropout]

**BatchNorm.** Normalization using minibatch statistics can improve training
behavior, while adding affine parameters and normalization work. Evaluation
must use the fitted running statistics; validation observations must not
update them. BatchNorm's originally reported acceleration on image models is
not an expected numerical speedup for this financial task.
[Ioffe and Szegedy, 2015][batchnorm]

**Residual connections and LayerNorm.** Skip connections provide a direct path
through a deeper network, motivating their use when ordinary deep networks
are hard to optimize. LayerNorm computes statistics within each example and
uses the same normalization calculation during training and evaluation. The
residual model adds dense and normalization operations, and its benefit must
be judged against that cost. Its normalization behavior does not depend on
other examples in an inference batch. [He et al., 2015/2016][resnet];
[Ba et al., 2016][layernorm]

Overlapping windows and shared market conditions mean the number of rows
overstates the number of independent observations. Architecture choice should
therefore consider chronological validation error, variation across seeds and
dates, and the persistence comparator alongside runtime. A lower return RMSE
alone does not demonstrate better absolute error, price accuracy, or trading
performance.

## Other deep learning techniques to consider next

The following alternatives are discussed from the literature and have **not
been trained in this MLP comparison**. Their relative performance and runtime
on this dataset remain unknown.

| Technique | Potential value for this project | Compute, inference, and input implications |
| --- | --- | --- |
| LSTM | Gated recurrent state can preserve temporal information that summary features discard. | Requires a meaningful per-bar sequence tensor. Recurrent processing introduces a sequential dependency across time steps. Longer windows increase work. |
| Causal temporal convolution / TCN | Can learn local and longer-range temporal patterns through causal and potentially dilated convolutions. | Requires sequence inputs and a chosen receptive field. Convolution processes positions in parallel during training; kernel sizes, channels, and depth determine cost. |
| FT-Transformer | Learns interactions among feature embeddings and can use the current 15-feature representation. | Adds feature embeddings and attention layers. Attention interactions grow quadratically with token count, although with only 15 features the actual runtime also depends strongly on implementation and hidden width. |

LSTM was developed to improve learning across long temporal dependencies;
its memory mechanism is a reason to test sequence inputs, not evidence that
it will outperform the engineered-feature MLPs here.
[Hochreiter and Schmidhuber, 1997][lstm]

Bai et al. found temporal convolutional models competitive with or better
than recurrent models across their sequence benchmarks. Their results make
a causal TCN a reasonable future comparator; they do not predict its ranking
on option returns. [Bai et al., 2018][tcn]

FT-Transformer operates on feature tokens rather than requiring a raw time
sequence. It performed strongly in the cited tabular benchmark, with higher
computational costs than simpler networks in the authors' comparisons.
[Gorishniy et al., 2021][tabular]

The original Transformer paper explains the parallelism and sequence-length
tradeoffs of attention, recurrence, and convolution. Its complexity analysis
supports the qualitative comparison above, but absolute timing must be
measured locally. [Vaswani et al., 2017][attention]

## Primary references

1. Gorishniy, Y., Rubachev, I., Khrulkov, V., and Babenko, A. (2021).
   [*Revisiting Deep Learning Models for Tabular Data*][tabular]. NeurIPS 2021.
   Sections 3.1–3.3 describe MLP, ResNet, and FT-Transformer; Appendix F gives
   architecture tuning spaces.
2. Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., and Salakhutdinov,
   R. (2014). [*Dropout: A Simple Way to Prevent Neural Networks from
   Overfitting*][dropout]. Journal of Machine Learning Research, 15:1929–1958.
3. Ioffe, S., and Szegedy, C. (2015). [*Batch Normalization: Accelerating Deep
   Network Training by Reducing Internal Covariate Shift*][batchnorm]. ICML,
   PMLR 37:448–456.
4. Ba, J. L., Kiros, J. R., and Hinton, G. E. (2016).
   [*Layer Normalization*][layernorm]. arXiv:1607.06450.
5. He, K., Zhang, X., Ren, S., and Sun, J. (2015 preprint; CVPR 2016).
   [*Deep Residual Learning for Image Recognition*][resnet].
6. Hochreiter, S., and Schmidhuber, J. (1997).
   [*Long Short-Term Memory*][lstm]. Neural Computation, 9(8):1735–1780.
7. Bai, S., Kolter, J. Z., and Koltun, V. (2018).
   [*An Empirical Evaluation of Generic Convolutional and Recurrent Networks
   for Sequence Modeling*][tcn]. arXiv:1803.01271.
8. Vaswani, A., et al. (2017). [*Attention Is All You Need*][attention].
   NeurIPS 2017. Table 1 compares layer complexity and sequential operations.

[tabular]: https://arxiv.org/html/2106.11959v5
[dropout]: https://jmlr.org/papers/v15/srivastava14a.html
[batchnorm]: https://proceedings.mlr.press/v37/ioffe15.html
[layernorm]: https://arxiv.org/abs/1607.06450
[resnet]: https://arxiv.org/abs/1512.03385
[lstm]: https://doi.org/10.1162/neco.1997.9.8.1735
[tcn]: https://arxiv.org/abs/1803.01271
[attention]: https://arxiv.org/html/1706.03762v7
