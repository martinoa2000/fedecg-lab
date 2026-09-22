"""Saliency over raw ECG waveforms.

Planned contents:
    saliency: Integrated Gradients / Grad-CAM 1D attributions.
    plot:     Overlay attribution heat on the signal, per lead.

The goal is a sanity check a cardiologist could argue with: for a myocardial
infarction prediction, does the attribution concentrate on the ST segment and
Q waves, or on baseline noise between beats?
"""
