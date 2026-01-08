#!/usr/bin/env python3
"""Script to migrate use_mixed_precision from optimization to performance_optimization section.

This script fixes YAML configs where use_mixed_precision is incorrectly placed
under the optimization section instead of performance_optimization.
"""

import re
from pathlib import Path


def fix_config_file(filepath: Path, dry_run: bool = False) -> bool:
    """Fix a single config file.

    Returns True if the file was modified.
    """
    content = filepath.read_text()

    # Skip if no use_mixed_precision in file
    if "use_mixed_precision" not in content:
        return False

    # Skip if already has performance_optimization section with use_mixed_precision
    if "performance_optimization:" in content and re.search(
        r"performance_optimization:\s*\n\s+use_mixed_precision:", content
    ):
        return False

    lines = content.split("\n")
    new_lines = []
    in_optimization = False
    optimization_indent = 0
    found_use_mixed = False
    use_mixed_val = "true"

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.lstrip()
        current_indent = len(line) - len(stripped)

        # Detect optimization section start
        if stripped.startswith("optimization:") and not stripped.startswith("performance_optimization:"):
            in_optimization = True
            optimization_indent = current_indent
            new_lines.append(line)
            i += 1
            continue

        # Check if we're exiting optimization section (next top-level key)
        if in_optimization and stripped and not stripped.startswith("#"):
            if current_indent <= optimization_indent and ":" in stripped:
                in_optimization = False
                # Insert performance_optimization section before this line
                if found_use_mixed:
                    indent = " " * optimization_indent
                    new_lines.append(f"{indent}performance_optimization:")
                    new_lines.append(f"{indent}  use_mixed_precision: {use_mixed_val}")
                    found_use_mixed = False

        # Skip use_mixed_precision line in optimization section
        if (
            in_optimization
            and "use_mixed_precision" in stripped
            and "performance_optimization" not in lines[max(0, i - 5) : i]
        ):
            # Extract value
            match = re.search(r"use_mixed_precision:\s*(true|false)", stripped, re.IGNORECASE)
            if match:
                use_mixed_val = match.group(1).lower()
            found_use_mixed = True
            i += 1
            continue

        new_lines.append(line)
        i += 1

    # Handle case where optimization was the last section
    if found_use_mixed:
        indent = " " * optimization_indent
        new_lines.append(f"{indent}performance_optimization:")
        new_lines.append(f"{indent}  use_mixed_precision: {use_mixed_val}")

    new_content = "\n".join(new_lines)

    if new_content != content:
        if not dry_run:
            filepath.write_text(new_content)
        return True
    return False


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Fix config files")
    parser.add_argument("--dry-run", action="store_true", help="Don't write changes")
    args = parser.parse_args()

    config_dir = Path("configs")
    fixed_count = 0

    for filepath in sorted(config_dir.rglob("*.yaml")):
        if fix_config_file(filepath, dry_run=args.dry_run):
            print(f"{'[DRY-RUN] Would fix' if args.dry_run else 'Fixed'}: {filepath}")
            fixed_count += 1

    print(f"\nTotal files {'to fix' if args.dry_run else 'fixed'}: {fixed_count}")


if __name__ == "__main__":
    main()
