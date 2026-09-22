"""Differentially private training with Opacus.

Planned contents:
    dp_sgd: Attach an Opacus `PrivacyEngine` to the shared training loop and
            report the privacy budget (epsilon) actually spent.

Note on what is being protected: applying DP-SGD inside each client gives
*record-level* privacy within that hospital, which is the guarantee this
repository measures. That is a different and weaker statement than
*client-level* privacy, which protects the participation of an entire hospital
and requires noise to be added at aggregation time instead.
"""
