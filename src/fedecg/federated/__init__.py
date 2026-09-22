"""Flower clients, strategies and simulation entry points.

Planned contents:
    client:     A Flower client wrapping the shared training loop.
    strategy:   FedAvg and FedProx configuration.
    simulation: Run N simulated hospitals in one process.

Everything runs in simulation mode: no network, no real hospitals. The point is
to reproduce the statistical consequences of decentralization (non-IID data,
partial participation, limited local epochs), not its networking.
"""
