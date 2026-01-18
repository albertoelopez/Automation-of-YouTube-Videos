"""LLM module for script generation"""
from .ollama import OllamaClient, generate_script

__all__ = ["OllamaClient", "generate_script"]
