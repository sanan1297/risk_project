"""Extract Mermaid diagrams from MD files and render them as PNG images."""

import re
import subprocess
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
DOCS = PROJECT / "docs"
DIAGRAMS = DOCS / "diagrams"
MMDC = "npx.cmd" if sys.platform == "win32" else "npx"


def extract_mermaid(filepath: Path):
    """Extract all mermaid code blocks from a markdown file with context."""
    content = filepath.read_text(encoding="utf-8")
    blocks = []
    pattern = re.compile(r"```mermaid\n(.*?)```", re.DOTALL)
    for i, match in enumerate(pattern.finditer(content)):
        code = match.group(1).strip()
        # Find the section context (look backwards for ## headings)
        pre = content[: match.start()]
        headings = re.findall(r"^###?\s+(.+)$", pre, re.MULTILINE)
        section = headings[-1] if headings else f"diagram_{i}"
        slug = re.sub(r"[^a-z0-9]+", "_", section.lower()).strip("_")
        blocks.append({
            "code": code,
            "section": section,
            "slug": slug,
            "match": match,
        })
    return blocks


def render_mermaid(code: str, output_path: Path, width: int = 1200):
    mmd_path = output_path.with_suffix(".mmd")
    mmd_path.write_text(code, encoding="utf-8")

    cmd = [
        MMDC, "-y", "@mermaid-js/mermaid-cli",
        "-i", str(mmd_path),
        "-o", str(output_path),
        "-w", str(width),
        "-b", "white",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        print(f"[ERROR] {output_path.name}: {result.stderr[:500]}")
        return False
    print(f"[OK] {output_path.name}")
    return True


def main():
    DIAGRAMS.mkdir(parents=True, exist_ok=True)

    md_files = [
        DOCS / "proceso.md",
        PROJECT / "README.md",
    ]

    for md_file in md_files:
        if not md_file.exists():
            continue
        blocks = extract_mermaid(md_file)
        if not blocks:
            continue

        rel = md_file.relative_to(PROJECT)
        for b in blocks:
            name = f"{b['slug']}.png"
            img_path = DIAGRAMS / name
            rel_img = f"diagrams/{name}"

            ok = render_mermaid(b["code"], img_path)
            if ok:
                print(f"  {rel} :: {b['section']} -> {rel_img}")


if __name__ == "__main__":
    main()
