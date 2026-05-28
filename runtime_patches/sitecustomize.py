from __future__ import annotations

import logging

try:
    from viv.vllm_omni_monkey_patch import install

    install()
except Exception:
    logging.getLogger(__name__).exception("Failed to install VIV runtime monkey patches")
