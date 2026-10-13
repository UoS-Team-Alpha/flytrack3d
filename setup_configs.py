"""Create the local configs from their templates: configs/<name>.example.json -> configs/<name>.json.

Run it once after cloning, and again whenever a new template is added:
    python setup_configs.py

A config that already exists is never touched. A template with something to fill in (a
"path/to/..." placeholder, or an empty camera calibration) keeps its '_example' line, which the
main_*.py scripts refuse to run: fill the config in, then delete that line. A template with nothing
to fill in is ready to use as copied. Templates for an alternative set-up (<name>.<variant>.example.json,
e.g. detection_source.batch) are not copied: copy one by hand if you want it.
"""

import re
from pathlib import Path

CONFIGS = Path(__file__).resolve().parent / "configs"
PLACEHOLDERS = ("path/to", '"matrix": null')   # what marks a template as needing to be filled in


def setup_configs(configs_dir: Path = CONFIGS) -> None:
    created, to_fill, existing, variants = [], [], [], []
    for template in sorted(configs_dir.rglob("*.example.json")):
        name = template.name.replace(".example", "")
        target, shown = template.with_name(name), template.with_name(name).relative_to(configs_dir.parent)
        if name.count(".") > 1:
            variants.append(template.relative_to(configs_dir.parent))
        elif target.exists():
            existing.append(shown)
        else:
            text = template.read_text()
            if any(p in text for p in PLACEHOLDERS):
                to_fill.append(shown)
            else:
                text = re.sub(r'^\s*"_example": true,\n\n?', "", text, flags=re.M)
            target.write_text(text)
            created.append(shown)

    for title, paths in (("Created", created), ("Already there, left as they are", existing),
                         ("Alternative templates, not copied", variants)):
        if paths:
            print(f"{title}:\n" + "\n".join(f"  {p}" for p in paths))
    if to_fill:
        print("\nFill these in, then delete their '_example' line:\n" + "\n".join(f"  {p}" for p in to_fill))
    elif not created:
        print("\nNothing to do: every config exists.")


if __name__ == "__main__":
    setup_configs()
