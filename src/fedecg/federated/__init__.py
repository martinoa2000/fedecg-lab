"""Flower clients, strategies and simulation entry points.

Contents:
    client:     A Flower `NumPyClient` wrapping the shared training loop.
    strategy:   Flower's FedAvg and FedProx, configured from a config.
    simulation: Drive N simulated hospitals round by round in one process.

Everything runs in simulation mode: no network, no real hospitals. The point is
to reproduce the statistical consequences of decentralization (non-IID data,
partial participation, limited local epochs), not its networking.
"""
