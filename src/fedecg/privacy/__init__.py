"""Differentially private training with Opacus.

Contents:
    dp_sgd: Calibrate DP-SGD noise to a target epsilon, attach an Opacus
            `PrivacyEngine` to the shared training loop, and report the
            privacy budget actually spent.

Note on what is being protected: applying DP-SGD inside each client gives
*record-level* privacy within that hospital, which is the guarantee this
repository measures. That is a different and weaker statement than
*client-level* privacy, which protects the participation of an entire hospital
and requires noise to be added at aggregation time instead.
"""
