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
- h100 honest run on a100 machine with TP=2 parallelism (hardest honest case to pass, as i initially assumed)
- i also did a sanity check validating h100 honest run on single a100 on a smaller pilot dataset (more on this below)

i run the same generations pinning initial latent, prompt emebds used for conditioning, and then i saved model prediction tensors after every step, but reused the prediction tensors from the baseline executor runs, so that model state is synchronized after every step.

then i experimented with various ways for comparing executor model predictions vs verifier model predictions and aggregating them into a similarity score. i used held out sets to test how metrics performed. i aimed for 0% false negatives (marking honest inferences as invalid) while trying to catch as many dishonest variations as possible.

## validation algorithm

I compared several similarity similarity/closeness ideas including relative l1 error, relative l2 error, cosine similarity, and also i tried a TOPLOC-paper insipired idea to use topK largest values from the prediction tensors (not from the hidden layer, but i think that "high-magnitude activations are more stable under GPU nondeterminism" still applies to the prediction tensor as well). The results showed that the best and also the simplest strategy is still relative L2.

so, i calculated similarity as `1 / (1 + relative_l2_error)`.

## results

results quickly showed that the error differed noticeably depending on the denoising step it was related to. during the initial steps where high-noise expert was active, INT4 performed very much similarly to the full model on A100 TP=2, but on low-noise expert steps it performed clearly worse than TP=2.

this seems like an expected result 


we can't catch int4 reliably on low noise steps

even this paper about W4A4 Quantization of Wan2.2-I2V (https://arxiv.org/abs/2605.27003) says that "high-noise and low-noise experts exhibit distinct quantization sensitivities that a single global calibration policy cannot capture"




 high noise impossible to separate. low noise was possible to separate.



thus, we are left with 3 choices as i see it:
- try to save even 

conduct cross calibration across many 

i used int4 as cheater strategy for this experiment. int4 quantization is W4A16, i.e. weights are 4bit, but all math is in 16 bit. you can check how it performs in `notes/03-validation-strategy/artifacts/h100_sxm__int4_quantization`, there are generted videos from pilot dataset.