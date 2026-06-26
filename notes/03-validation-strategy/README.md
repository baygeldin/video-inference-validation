# validation strategy

## TL;DR
the previous experiments showed that we need to save the full trace (every model prediction) despite the storage concerns because otherwise there would be a sure way to cheat the inference process by performing less denoising steps as actually claimed.

in this experiment i tried to test the full validation strategy relying on the full execution trace using INT4 as the malicious deviation as it's the only one left. the results are somewhat surprising and still show that it depends not only on the GPU configuration differences between executor and verifier. and on some combinations it's impossible to distinguish them.

thus, for Wan2.2-T2V-A14B specifically if we want to catch INT4 quantization cheating, i think we're left with two choices:
- deploy the quantized model instead of the full one
- limit validation to specific subsets (more on this below)

## experiment design

i generated a larger dataset (105 examples in medium, 15 per category + 14 used examples from pilot as well).
i generated 105 prompts from pilot and medium datasets for each of these configs:
- h100 int4
- h100 full

then i tried to validate:
- h100 int4 on h100 machine (hardest cheating case to catch due to the least differences)
- h100 honest run on a100 machine with TP=2 parallelism (faster to generate and hardest honest case to pass, as i initially assumed)
- i also did a sanity check validating h100 honest run on single a100 on a smaller pilot dataset (more on this below)

i run the same generations pinning initial latent, prompt emebds used for conditioning, and then i saved model prediction tensors after every step, but reused the prediction tensors from the baseline executor runs, so that model state is synchronized after every step.

then i experimented with various ways for comparing executor model predictions vs verifier model predictions and aggregating them into a similarity score. i used held out sets to test how metrics performed. i aimed for 0% false negatives (marking honest inferences as invalid) while trying to catch as many dishonest variations as possible.

## validation algorithm

I compared several similarity similarity/closeness ideas including relative l1 error, relative l2 error, cosine similarity, and also i tried a TOPLOC-paper insipired idea to use topK largest values from the prediction tensors (not from the hidden layer, but i think that "high-magnitude activations are more stable under GPU nondeterminism" still applies to the prediction tensor as well). The results showed that the best and also the simplest strategy is still relative L2.

so, i calculated similarity score as `1 / (1 + per_step_medium_relative_l2_error)`.

## results

results quickly showed that the error differed noticeably depending on the denoising step it was related to. during the initial steps where high-noise expert was active, INT4 performed very much similarly to the full model on A100 TP=2, but on low-noise expert steps it performed clearly worse than TP=2.

this seems like an expected result because even this paper about W4A4 Quantization of Wan2.2-I2V (https://arxiv.org/abs/2605.27003) says that "high-noise and low-noise experts exhibit distinct quantization sensitivities that a single global calibration policy cannot capture".

so, it means that we can't catch int4 reliably on low noise steps meaning that an executor could perform high-noise steps on quantized model, then swith to full one. but since quantized models do well on high noise steps preseving enough quality indistinguishable from honest generations, then for the end user it won't make a difference. also it's not much of a saving strategy, it may prove to be more of a hassle in fact.

so, i decided to split the validation into buckets for low and high noise steps, with different thresholds. to pass the validtaion the inference must pass both thresholds in each bucket. if an executor did everything with int4, it'd have failed because it's easy to catch on low-noise steps. 

i calculated the thresholds, tested them and found that the results of running this showed that we were able to catch all INT4 runs while not invalidating a single honest inference.

however, then i tried the sanity check where i validated the H100 honest inference with a single A100 without tensor parallelism, and the results were surprising. turns out a single A100 deviation gets WORSE on high noise steps compared to the H100 than A100 TP=2.  it's hard to say precicely why this happens, but most likely TP=2 changed the kernel in a way that happened to land closer to H100 numerics.

(here i need to paste charts from the `plots` subfolder, style them similarly to how i did with charts in `notes/02-intermediate-artifacts/README.md`)

note: charts were build from the comparisons in `notes/03-validation-strategy/artifacts/comparisons`.

## conclusion

it's not currently possible to reliably distinguish between INT4 quantized version and the full model unless we narrow the validation environment to exactly the same GPU configuration. but this could get tricky because it assumes there is always a host available on the network running the same exact configuration of GPU and the same exact model. theoretically, it could be possible if the network is large, but it requires a different kind of threat analysis.

we could theoretically loosen the requirements a bit if we conclude that certain GPU configuration combinations are "workable" (e.g. if "B200 executor could be verified by H200 verifier reliably and catch all dishonest inferences", etc). though it would compicate things even more and require to condust cross calibration across many GPU combinations. we can use this harness to facilitate that, but it still takes time and compute.

thus, i think the best strategy for a decentralized network is to deploy the "least common denominator", i.e. the version of the model that doesn't have a "cheaper" counterpart that could pass as that version. for such models we could use generous thresholds that could allow benign hardware differences.

it doesn't mean that the results would be subpar though. in case of INT4 quantization (more precicely it's W4A16, i.e. weights are 4bit, but all math is in 16 bit), the results are pretty close, sometimes one could say even better subjectively.

check out all of the videos generated with int4 in `notes/03-validation-strategy/artifacts/medium__h100_sxm__int4_quantization`. compare them to `notes/03-validation-strategy/artifacts/medium__h100_default`. or compare the pilot int4 generations in `notes/03-validation-strategy/artifacts/pilot__h100__int4_quantization` with the honest pilot generations from the previous experiment in  `notes/02-intermediate-artifacts/artifacts/h100_sxm__default`.

also take a look at `notes/03-validation-strategy/artifacts/comparisons/medium__h100_int4_vs_h100.json`. for each generated example from `medium` dataset it contains the median per-step relative L2 error. if we compute the median value per-example then we get `0.086`, i.e. 92% similarity score as per out formula. the maximum is `0.15`, or 86% similarity in the forst case.