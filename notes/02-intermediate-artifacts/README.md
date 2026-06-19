# Intermediate artifacts

## TL;DR

The pilot experiment showed that output videos alone are not a reliable basis for inference validation. Differences between GPU models can be large enough to mask small but malicious deviations, so the verifier needs access to intermediate artifacts from the diffusion process.

So, this experiment tries to answer these questions:
- Which intermediate artifacts should we save?
- How many artifacts are enough to save?

The answer is unfortunately clear: saving only a portion of the intermediate artifacts is not enough. After reusing the artifacts from the first 20 steps, a malicious H100 run with one fewer denoising step stayed closer to the baseline than a benign A100 run across every prompt and every metric tested. To make validation robust, we should save prediction latents for all denoising steps.

The storage cost is acceptable: for Wan2.2-T2V-A14B at 480p, 16 fps, and 81 frames, this is roughly 168 MB per generated video, or about 45 GB per fully utilized H100 per day. There are also a couple of additional benefits to saving artifacts from every denoising step.

## Artifact choice

I chose prediction latents as the intermediate artifact to test. The remaining question was how many of them we could safely keep.

The starting hypothesis was that we might only need the early artifacts. Early denoising steps primarily determine motion and layout, while later steps refine appearance ([Characterizing Motion Encoding in Video Diffusion Timesteps](https://arxiv.org/abs/2512.22175)). If that held strongly enough, we might have been able to save only the first `N` prediction latents, reuse them during verification, and let the verifier finish the remaining steps independently. That would reduce storage while still preserving the high-level structure of the generation.

Saving prediction latents is preferable to saving the current video latent for two reasons.

First, prediction latents are scheduler-independent. A scheduler applies the model prediction to the current video latent to produce the next latent. In Wan2.2, the default UniPC scheduler computes the next latent from the current denoising step input, current prediction, previous denoising step input, and previous prediction. Saving the scheduler state at step `N` would therefore require retaining artifacts from both `N` and `N-1` steps. Different video models may use different schedulers, so supporting new video models could get tricky. Model predictions are the more universal artifact.

Second, prediction latents are smaller. Model weights are typically `bfloat16`, so predictions are naturally `bfloat16`. On the other hand, intermediate video latents may be `float32`, because schedulers often do their arithmetic in higher precision. In practice, storing video latents could double the artifact size.

## Experiment design

I used two examples from each T2V-CompBench category, rather than a single prompt as in the pilot experiment. Each baseline video was generated on an H100 with the default configuration and 40 denoising steps.

Then I compared two deviations after reusing the artifacts from the first 20 denoising steps:
- **Benign deviation:** A100 SXM with TP=2 tensor parallelism.
- **Malicious deviation:** H100 with 39 total denoising steps instead of 40.

I originally wanted to use TP=4 as the benign case, but sourcing a 4xA100 machine consistently was difficult. TP=2 was still a hard benign deviation, as the results show.

The malicious case was chosen because the pilot experiment showed that "one fewer denoising step" was especially hard to distinguish from an honest generation.

### Prefix length

I started by saving and reusing the artifacts from the first 20 steps out of 40 denoising steps. If that had separated the benign and malicious cases cleanly, the next step would have been to bisect downward and test 10 saved steps, then 5, and so on.

I did not plan to bisect upward. Saving 20 out of 40 steps already means only a 2x storage reduction. If 20 steps were not enough, saving a larger prefix would provide less than a 2x storage reduction, which is not enough to justify the added implementation complexity.

I made sure that the logic related to saving artifacts during inference accounts for scheduler timesteps. The scheduler is part of a video generation pipeline that updates the current video latent using model predictions. Scheduler updates at each denoising step depend on the current sigma value, also called the *timestep*. A naive 39-step run would use a different sigma schedule from the beginning, which would make the first 20 predictions diverge immediately and make the malicious run too easy to catch.

Instead, the malicious run should reuse the first 20 saved artifacts exactly as if it were doing an honest 40-step generation, then compress the remaining 20 denoising steps into 19 steps. This models a more realistic cheating strategy: match the saved prefix to pass validation against the saved artifacts, then save compute in the suffix to cheat validation against the final video file.

## Results

After reusing the artifacts from the first 20 steps, both benign and malicious outputs were visually difficult to distinguish in the sampled videos, but the metrics showed the real problem: the malicious deviation was consistently closer to the H100 baseline than the benign A100 TP=2 deviation.

Across all 14 prompts:
- Per-frame SSIM (Structural Similarity Index Measure) calculated from the video files is higher for the malicious run than for the benign run.
- Final-latent error is lower for the malicious run than for the benign run.

| Configuration | Mean SSIM | Min SSIM | Max SSIM | Final latent RMSE | Final latent relative L2 error |
| --- | ---: | ---: | ---: | ---: | ---: |
| H100, 39 steps | 0.974124 | 0.964744 | 0.977736 | 0.047433 | 0.053096 |
| A100, TP=2 | 0.966597 | 0.953093 | 0.971449 | 0.064184 | 0.072434 |

Plots:
- [Mean SSIM](plots/mean_ssim_scatter.png)
- [Min SSIM](plots/min_ssim_scatter.png)
- [Max SSIM](plots/max_ssim_scatter.png)
- [Final latent RMSE](plots/rmse_scatter.png)
- [Final latent relative L2 error](plots/relative_l2_error_scatter.png)

Detailed per-prompt measurements can be found in [comparison.json](comparison.json).

## Conclusions

Saving only a portion of the intermediate artifacts is not sufficient. In this test, the benign A100 TP=2 run drifted farther from the H100 baseline over the unsaved suffix than the malicious H100 run that skipped one denoising step.

That means a verifier using only this 20-step prefix would have no reliable way to distinguish the malicious shortcut from normal hardware variation. Saving one step out of 40 reduces diffusion compute by 2.5%, which may seem small per video, but it's meaningful at scale. And since a host has a direct economic incentive to skip work whenever validation permits it, this failure mode is not acceptable.

However, let's see what this means in practice, and whether there are benefits to saving so many artifacts.

### Storage estimate

For Wan2.2-T2V-A14B at 480p, 16 fps, and 81 frames, one prediction latent has shape:

```text
[
  16,   # latent channels
  21,   # latent time
  60,   # latent height
  104   # latent width
]
```

That is `16 * 21 * 60 * 104 = 2,096,640` values. Stored as `bfloat16`, one prediction latent is about 4.2 MB, which was confirmed during the experiment.

For 40 denoising steps:

- Artifact size per video: `40 * 4.2 MB = 168 MB`
- Maximum H100 throughput: about one 480p, 81-frame video every 5.4 minutes, based on the [Wan2.2-T2V-A14B model card](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B#computational-efficiency-on-different-gpus)
- Maximum daily throughput per H100: about 266 videos
- Maximum daily artifact volume per H100: `266 * 168 MB = 44,688 MB`, or about 45 GB

For 720p, each latent is larger, but generation throughput is lower, so similar calculations show that the total artifact storage per day would be about 30% lower than in the 480p case.

45 GB may seem like a lot considering that the videos themselves would take only about 120 MB in storage. Even so, the cost is modest compared with H100 rental cost: retaining three days of artifacts would require about 150 GB of additional storage per fully utilized H100. Network transfer costs still need to be accounted for separately, since they depend heavily on the provider.

### Simpler video validation

Saving all prediction latents makes the final video file less important for inference validation.

The verifier can validate inference directly against the saved intermediate artifacts. After that, it can pass the saved predictions through the scheduler to reconstruct the final video latent, then decode that latent with the VAE to produce the final frames. The resulting video should be effectively identical to the submitted one, aside from small nondeterminism in decoding or encoding.

As a result, we should not need complex perceptual metrics such as DISTS or LPIPS to decide whether inference was honest. Final-video validation can be much narrower: check that the submitted video matches the video reconstructed from the final video latent derived from the validated artifacts, and treat mismatches as file tampering (e.g. censorship) or post-processing issues rather than inference drift.

### Potential compute savings

Saving every prediction latent also gives the verifier a cheaper path than replaying the full diffusion process.

To reconstruct the final video, the verifier does not need to run the diffusion transformer at all. It can feed the saved predictions into the scheduler, recover the final video latent, and decode it with the VAE.

For inference validation, the verifier can then sample a small number of denoising steps at random, rerun only those model forward passes, and compare the new predictions with the saved predictions. The executor does not know which steps will be checked, so it has to generate all steps honestly. The verifier, meanwhile, avoids most of the diffusion cost.
