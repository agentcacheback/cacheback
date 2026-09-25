"""The vLLM KV connector and the torch-only pieces it is built from.

Nothing in this package imports vllm at module import time; the connector builds
its vllm-facing class behind a guarded import.
"""
