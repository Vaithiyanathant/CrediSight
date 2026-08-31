"""Serving layer.

Thin orchestration between the FastAPI routers and the ML core. Routers stay
free of modelling logic; these services own it, are unit-testable without HTTP,
and are the only place that knows how the pieces compose into one bundle.
"""
