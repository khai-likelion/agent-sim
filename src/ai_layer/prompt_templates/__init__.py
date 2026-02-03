"""
Prompt template loader utility.
Loads .txt files from the prompt_templates directory and supports variable substitution.
Similar to Stanford Generative Agents' prompt_template/v2/ pattern.
"""

from pathlib import Path

from config import get_settings


def load_template(template_name: str) -> str:
    """Load a prompt template by name (without .txt extension)."""
    settings = get_settings()
    template_path = settings.paths.prompt_templates_dir / f"{template_name}.txt"
    if not template_path.exists():
        raise FileNotFoundError(f"Prompt template not found: {template_path}")
    return template_path.read_text(encoding="utf-8")


def render_template(template_name: str, **variables) -> str:
    """Load and render a template with variable substitution."""
    template = load_template(template_name)
    return template.format(**variables)
