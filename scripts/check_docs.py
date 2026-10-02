"""Check Markdown fences, math delimiters, table widths and local link targets."""
from pathlib import Path
import re
from urllib.parse import unquote, urlparse

ROOT = Path(__file__).resolve().parents[1]


def check(path: Path) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    errors, fence, display, table_width = [], None, False, None
    for n, line in enumerate(lines, 1):
        prefix = f"{path.relative_to(ROOT)}:{n}"
        match = re.match(r"^\s*(`{3,}|~{3,})", line)
        if match:
            marker = match[1]
            if fence is None:
                fence = marker
            elif marker[0] == fence[0] and len(marker) >= len(fence):
                fence = None
            table_width = None
            continue
        if fence is not None:
            continue
        if line.strip() == "$$":
            if not display and n > 1 and lines[n-2].strip():
                errors.append(f"{prefix}: display math needs a blank line before it")
            if display and n < len(lines) and lines[n].strip():
                errors.append(f"{prefix}: display math needs a blank line after it")
            display = not display
            continue
        if display:
            continue
        if "$$" in line:
            errors.append(f"{prefix}: display delimiters must be on their own lines")
        prose = re.sub(r"`[^`]*`", "", line)
        if len(re.findall(r"(?<!\\)\$", prose)) % 2:
            errors.append(f"{prefix}: unpaired inline math delimiter")
        if any(token in prose for token in (r"\(", r"\)", r"\[", r"\]")):
            errors.append(f"{prefix}: use dollar math delimiters")
        if line.strip().startswith("|"):
            width = len(re.split(r"(?<!\\)\|", line.strip())) - 2
            if table_width is None:
                table_width = width
            elif width != table_width:
                errors.append(f"{prefix}: inconsistent table width")
            if "$" in line:
                errors.append(f"{prefix}: move table formulas to surrounding prose")
        else:
            table_width = None
        for target in re.findall(r"\]\(([^\s)]+)\)", line):
            target = target.strip("<>")
            parsed = urlparse(target)
            if parsed.scheme or target.startswith("#"):
                continue
            if not (path.parent / unquote(parsed.path)).exists():
                errors.append(f"{prefix}: missing local link target: {target}")
    if fence is not None:
        errors.append(f"{path.relative_to(ROOT)}: unclosed code fence")
    if display:
        errors.append(f"{path.relative_to(ROOT)}: unclosed display math")
    return errors


def main():
    files = [ROOT / "README.md", ROOT / "README.zh.md"]
    files += sorted((ROOT / "docs").glob("*.md"))
    files += [ROOT / name / "README.md" for name in
              ("feature_extraction", "sentiment_model", "explainability")]
    errors = [error for path in files for error in check(path)]
    if errors:
        raise SystemExit("\n".join(errors))
    print(f"Checked {len(files)} Markdown files: fences, math, tables and local links.")


if __name__ == "__main__":
    main()
