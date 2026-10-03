"""Bundle-backed resolver for the Hugging Face ``kernels`` contract.

The contract is adopted as-is; only resolution is inverted. On mobile, kernel
variants are selected and compiled at build time and read from the app bundle,
because iOS will not execute downloaded native code. Desktop keeps resolving
against the Hub. See DESIGN.md §8.
"""
