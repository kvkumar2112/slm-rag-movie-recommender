"""Generative-retrieval movie recommender.

Flow: user's liked movies -> SLM writes a target profile -> profile is embedded ->
ChromaDB returns the real movies closest to that profile.
"""
