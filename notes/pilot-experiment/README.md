# Pilot experiment

## Configurations tested
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

