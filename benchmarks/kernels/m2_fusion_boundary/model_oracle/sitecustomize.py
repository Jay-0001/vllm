"""Install the M2 Q/K experiment only in explicitly opted-in processes."""

import os

if os.environ.get("M2_QK_MODEL_ORACLE") == "1":
    from m2_qk_model_oracle import install

    install()
