"""Simulation module — MuJoCo-based home environment (CPU-only, no GPU overhead).

Two-phase pipeline: offline trajectory collection → static HDF5 dataset → training from disk.
Future: migrate to Isaac Sim for higher-fidelity rendering when GPU resources permit.
"""
