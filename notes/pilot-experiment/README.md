# Pilot experiment

## Conclusions

### GPU model matters *a lot*

When the GPU model is **exactly the same**, the results are consistent, meaning that we can expect basically the same result across different generations. However, when the GPU is different, the results are unpredictable.

For example, the default configuration on H100 SXM, H100 NVL, and even H200 SXM look very similar and would likely be comparable algorithmically. However, the same generation on A100 PCIe and A100 SXM is visually distinct enough that it's hard to say if one of them simply didn't have one less denoising step.

It's particularly interesting that it matters, perhaps, more than even the attention backend. I tried to use the `TORCH_SDPA` on H100 SXM and it also looks very similar to the default configuration.

Additionally, I tried this on B200 (because it doesn't seem to support `FLASH_ATTN` in vLLM-Omni currently) and the result looks rather similar to the `tp4_parallelism` config.

### Hardest malign deviation is "one less denoising step"

The good news is that most malign configurations are significantly different from the default one:
- I managed to set up INT4 W4A16 quantization, and it looks blurry and low quality compared to original.
- Cache-DiT is not too bad, but still distinctly different. 
- 30 instead of 40 denoising steps also leads to different results.
- Disabled CFG results in a complete mess (makes sense, of course).

However, when it comes to the `steps_minus_1` config, it sometimes hard to distinguish it from legit benign deviations.

## Configurations tested

To see the full parameters and environment configuration for each generation, check out the `.json` sidecars in this folder.

### Overview
- H100 SXM
  - default
  - tp4_parallelism
  - int4_quantization
  - cfg_disabled
  - cache_dit
  - steps_minus_1
  - steps_minus_10
  - default with `TORCH_SDPA`
- H100 NVL
  - default
- A100 PCIe
  - default
  - steps_minus_1
- A100 SXM
  - default
  - steps_minus_1
- B200
  - default with `TORCH_SDPA`

### H100 SXM

#### Default

https://github.com/user-attachments/assets/2ccfd999-b4d0-4913-8669-6957e914a29a

#### TP4 parallelism

https://github.com/user-attachments/assets/def1eebc-2452-4cba-9824-b03806d2dedd

#### INT4 quantization

https://github.com/user-attachments/assets/74e2d32f-50c7-4919-9df2-61d02acbcae1

#### Cache-DiT

https://github.com/user-attachments/assets/2dd36f53-0d61-4510-8fc5-3042f3357cc7

#### 39 denoising steps

https://github.com/user-attachments/assets/85e21b54-e00d-4d36-a415-4196872b1bca

#### 30 denoising steps

https://github.com/user-attachments/assets/9deabb16-cff2-4a7c-b348-4135635a493c

#### CFG disabled

https://github.com/user-attachments/assets/aa60e821-b8a7-4529-87b9-059a2e456e63

#### TORCH_SDPA attention backend

https://github.com/user-attachments/assets/9b1a33eb-cf7b-49a3-bb06-9bdb6902d8f4

### H100 NVL

#### Default

https://github.com/user-attachments/assets/36ec3ba4-dfe8-4b80-ab98-8d09d3dd21fd

### H200 SXM

#### Default

https://github.com/user-attachments/assets/5875a2b3-4427-434e-af32-690d8835e6a4


### A100 SXM

#### Default

https://github.com/user-attachments/assets/54cc3367-01cf-403e-a21d-4bed312b60f5

#### 39 denoising steps

https://github.com/user-attachments/assets/e60c19a6-260e-4325-b8ca-698f86b1f60e

### A100 PCIe

#### Default

https://github.com/user-attachments/assets/49c90d95-1ed7-4701-a9de-6be1a06d0851

#### 39 denoising steps

https://github.com/user-attachments/assets/5e76dacc-e1c0-4184-95b4-90b6755a7923

### B200

#### TORCH_SDPA attention backend

https://github.com/user-attachments/assets/a7b4b5f8-39bc-444a-86cb-da53069009f6

