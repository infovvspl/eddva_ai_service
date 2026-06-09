"""
Per-vertical prompt overrides.

Each module here holds the *pure content* (system-prompt strings) for one
vertical's deviations from the canonical base in prompt_templates.py. Modules
expose plain strings only (no imports back into prompt_templates) so there is
no circular dependency — prompt_templates.py wires them into VERTICAL_OVERRIDES.
"""
