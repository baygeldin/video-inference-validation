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

Check the output section at the bottom to see the results of the generation.

## Output

### H100 SXM
- [Default run](./h100_sxm_default.mp4)
- [TP4 parallelism](./h100_sxm_tp4.mp4)
- [INT4 quantization](./h100_sxm_int4.mp4)
- [CFG disabled](./h100_sxm_cfg_disabled.mp4)
- [Cache DiT enabled](./h100_sxm_cache_dit.mp4)
- [Steps minus 1](./h100_sxm_steps_minus_1.mp4)
- [Steps minus 10](./h100_sxm_steps_minus_10.mp4)
- [Default run with `TORCH_SDPA`](./h100_sxm_torch_sdpa.mp4)

### H100 NVL
- [Default run](./h100_nvl_default.mp4)

### A100 PCIe
- [Default run](./a100_pcie_default.mp4)
- [Steps minus 1](./a100_pcie_steps_minus_1.mp4)

### A100 SXM
- [Default run](./a100_sxm_default.mp4)
- [Steps minus 1](./a100_sxm_steps_minus_1.mp4)

### B200
- [Default run with `TORCH_SDPA`](./b200_torch_sdpa.mp4)
