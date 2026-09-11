# llm-inference

A focused inference runtime for decoder-only language models.

The project separates model/backend execution, request semantics, effective
inference planning, and batch scheduling. Backend-specific dependencies are
optional and loaded lazily.
