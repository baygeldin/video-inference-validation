# Validation strategy

## TL;DR

The previous experiments showed that final videos are not enough for validation, and that saving only a prefix of the diffusion trace is also not enough. A verifier needs access to the full execution trace: every model prediction at every denoising step.

This experiment tested whether a full-trace validation strategy can catch the remaining malicious deviation: running Wan2.2-T2V-A14B with INT4/W4A16 quantized weights while claiming to have run the full model.

The short answer is: **full traces help, but they do not remove the need for hardware-aware calibration**. With a narrowly chosen verifier configuration, a two-bucket relative L2 threshold caught all INT4 pilot runs with zero honest rejections. But a sanity check with a different honest verifier configuration overlapped with the INT4 deviation, which means the threshold would either reject honest work or allow the quantized run.

For Wan2.2-T2V-A14B, the practical choices are:

- Deploy the quantized model as the baseline model, if its quality is acceptable.
- Restrict validation to calibrated executor/verifier hardware pairs and maintain thresholds per pair.

The second option may work, but it changes the problem from "validate this model" to "validate this model on this exact hardware and kernel family."

## Experiment design

I generated a larger prompt set for this experiment:

- `medium`: 105 examples, with 15 prompts from each T2V-CompBench category.
- `pilot`: the 14 examples used in the earlier experiments, reused here for detailed per-step plots.

For the `medium` set, I generated each prompt on H100 in two configurations:

- H100 default, full model.
- H100 INT4 quantization. More precisely, this is W4A16: 4-bit weights with 16-bit activations/math.

Then I compared the saved prediction traces in three validation settings:

- **Malicious case:** H100 INT4 executor checked by an H100 full-model verifier.
- **Initial honest case:** H100 full-model executor checked by an A100 TP=2 verifier.
- **Sanity-check honest case:** H100 full-model executor checked by a single A100 verifier.

For each comparison, I pinned the initial latent and prompt embeddings. At each denoising step, the verifier recomputed the model prediction, compared it with the executor's saved prediction, and then advanced the scheduler using the executor's saved prediction. That keeps the executor and verifier model states synchronized after every step, so the metric measures per-step prediction disagreement rather than accumulated latent drift.

The target was strict: choose thresholds that reject **0 honest runs** while catching as many dishonest runs as possible.

## Validation metric

I tested several prediction-tensor closeness metrics:

- Relative L1 error.
- Relative L2 error.
- Cosine similarity.
- A TOPLOC-inspired top-k comparison over high-magnitude values in the prediction tensor.

The best-performing metric was also the simplest: **relative L2 error**. For a saved executor prediction `x` and verifier prediction `y`, the comparison is:

```text
relative_l2_error = ||y - x||_2 / ||x||_2
similarity = 1 / (1 + relative_l2_error)
```

For each prompt, I used median per-step relative L2 error as the aggregate score. For thresholding, I also split the denoising steps by the Wan2.2 expert boundary:

- **High-noise expert:** steps `0-16`, where `sigma >= 0.875`.
- **Low-noise expert:** steps `17-39`, where `sigma < 0.875`.

The split matters because Wan2.2 uses different experts before and after the boundary. This is also consistent with recent Wan2.2 quantization work, which notes that the high-noise and low-noise experts have distinct quantization sensitivities ([arXiv:2605.27003](https://arxiv.org/abs/2605.27003)).

## Results

The first result was encouraging. When the honest reference was H100 executor -> A100 TP=2 verifier, a two-bucket threshold separated the INT4 runs from the honest runs.

The separation came from the low-noise bucket:

- H100 honest -> A100 TP=2 low-noise per-prompt medians: max `0.029`.
- H100 INT4 -> H100 low-noise per-prompt medians: min `0.048`.

So, under this verifier assumption, every INT4 pilot prompt failed while every honest H100 -> A100 TP=2 prompt passed.

However, that result did not survive the sanity check. When the honest verifier was a single A100 rather than A100 TP=2, the low-noise bucket moved into the same range as INT4:

- H100 honest -> single A100 low-noise per-prompt medians: `0.043` to `0.096`.
- H100 INT4 -> H100 low-noise per-prompt medians: `0.048` to `0.102`.

That overlap is the core problem. A threshold permissive enough to accept the single-A100 honest verifier would also accept at least some INT4 executions. A threshold strict enough to catch INT4 would reject honest single-A100 verification.

The likely explanation is kernel-level numeric behavior. A100 TP=2 appears to use a kernel path that happens to land closer to H100 numerics after the expert boundary, while single A100 lands farther away. The important point is not the exact kernel detail; it is that normal verifier configuration changes can be as large as the malicious quantization deviation.

### Per-prompt traces

The plots below were generated from the comparison files in `notes/03-validation-strategy/artifacts/comparisons`. Each plot shows relative L2 error by denoising step for one pilot prompt:

- Blue: H100 honest executor -> A100 TP=2 verifier.
- Green: H100 honest executor -> single A100 verifier.
- Red: H100 INT4 executor -> H100 verifier.

<img src="plots/action-binding-001.png" alt="Relative L2 error by step for action-binding-001" width="500">

<img src="plots/action-binding-200.png" alt="Relative L2 error by step for action-binding-200" width="500">

<img src="plots/consistent-attr-001.png" alt="Relative L2 error by step for consistent-attr-001" width="500">

<img src="plots/consistent-attr-200.png" alt="Relative L2 error by step for consistent-attr-200" width="500">

<img src="plots/dynamic-attr-001.png" alt="Relative L2 error by step for dynamic-attr-001" width="500">

<img src="plots/dynamic-attr-200.png" alt="Relative L2 error by step for dynamic-attr-200" width="500">

<img src="plots/interaction-001.png" alt="Relative L2 error by step for interaction-001" width="500">

<img src="plots/interaction-200.png" alt="Relative L2 error by step for interaction-200" width="500">

<img src="plots/motion-binding-001.png" alt="Relative L2 error by step for motion-binding-001" width="500">

<img src="plots/motion-binding-200.png" alt="Relative L2 error by step for motion-binding-200" width="500">

<img src="plots/numeracy-001.png" alt="Relative L2 error by step for numeracy-001" width="500">

<img src="plots/numeracy-200.png" alt="Relative L2 error by step for numeracy-200" width="500">

<img src="plots/spatial-relationship-001.png" alt="Relative L2 error by step for spatial-relationship-001" width="500">

<img src="plots/spatial-relationship-200.png" alt="Relative L2 error by step for spatial-relationship-200" width="500">

## Implications

The high-noise bucket does not provide enough separation to catch INT4 reliably. An executor could run the high-noise steps with the quantized model, switch to the full model for low-noise steps, and plausibly pass the bucketed validation. That strategy is probably not very attractive in practice, because it requires switching model variants mid-generation and gives less benefit than running INT4 end to end. Still, it shows that the validation signal is not uniformly strong across the denoising trajectory.

The bigger issue is the single-A100 sanity check. It shows that **the validation threshold is not only model-dependent; it is executor/verifier-pair-dependent**. For a decentralized network, that is operationally awkward:

- The network would need to know which GPU and kernel combinations can validate each other.
- Each accepted pair would need calibration data and thresholds.
- Verifier availability would become constrained by matching hardware, not just model support.

This does not make full-trace validation useless. It means full-trace validation is reliable only under a hardware compatibility policy.

## Conclusion

For Wan2.2-T2V-A14B, I do not think we can reliably distinguish the full model from the INT4/W4A16 model across arbitrary verifier hardware. The malicious deviation is too close to normal numeric drift from some honest verifier configurations.

The safest decentralized strategy is to deploy the "least common denominator" model variant: choose a baseline that does not have a cheaper counterpart capable of passing as that baseline. If INT4 quality is acceptable, then deploying INT4 directly may be cleaner than trying to prove that everyone ran BF16.

That does not necessarily mean lower-quality outputs. The INT4 generations are often close to the full-model generations, and some are subjectively competitive. You can compare:

- INT4 medium outputs: `notes/03-validation-strategy/artifacts/medium__h100__int4_quantization`
- Full-model medium outputs: `notes/03-validation-strategy/artifacts/medium__h100__default`
- INT4 pilot outputs: `notes/03-validation-strategy/artifacts/pilot__h100__int4_quantization`
- Full-model pilot outputs from the previous experiment: `notes/02-intermediate-artifacts/artifacts/h100_sxm__default`

For the full `medium` set, see `notes/03-validation-strategy/artifacts/comparisons/medium__h100_int4_vs_h100.json`. Across 105 examples, the median per-prompt relative L2 error is `0.08555`, corresponding to a similarity score of `0.921`. The worst per-prompt median is `0.15015`, corresponding to a similarity score of `0.869`.
