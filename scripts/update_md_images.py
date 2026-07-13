"""Replace Mermaid code blocks in MD files with image references."""

import re
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
DOCS = PROJECT / "docs"
DIAGRAMS = DOCS / "diagrams"

# Mapping from slug to human-readable caption
DIAGRAM_CAPTIONS = {
    "8_1_visi_n_general": "Arquitectura del prototipo — Frontend (Streamlit), Backend (FastAPI), Datos estáticos y Almacenamiento dinámico",
    "8_4_flujo_de_datos_predicci_n": "Flujo de datos — Predicción: usuario sube CSV, Streamlit llama a la API, feature engineering + SVR + Monte Carlo, resultados en SQLite",
    "8_5_flujo_de_datos_dashboard": "Flujo de datos — Dashboard: consulta stats de usage y training desde la API",
    "8_6_flujo_de_datos_historial": "Flujo de datos — Historial: paginación, vista completa, validación de sobrecosto real",
    "8_7_pipeline_completo": "Pipeline completo: desde SECOP API hasta el frontend, pasando por extracción, feature engineering y modelo SVR",
    "8_8_arquitectura_del_sistema": "Arquitectura del sistema: navegador → Streamlit → FastAPI → módulos backend → archivos de datos",
    "8_9_modelo_de_datos_sqlite_history_db": "Modelo de datos ER — Tabla predicciones en SQLite con campos cualitativos y cuantitativos",
}


def replace_mermaid_with_images(md_path: Path):
    content = md_path.read_text(encoding="utf-8")

    # Find all ```mermaid ... ``` blocks
    pattern = re.compile(r"```mermaid\n(.*?)```", re.DOTALL)

    replacements = []
    for i, match in enumerate(pattern.finditer(content)):
        code = match.group(1).strip()
        pre = content[: match.start()]
        headings = re.findall(r"^###?\s+(.+)$", pre, re.MULTILINE)
        section = headings[-1] if headings else f"diagram_{i}"
        slug = re.sub(r"[^a-z0-9]+", "_", section.lower()).strip("_")

        img_file = f"{slug}.png"
        img_path = DIAGRAMS / img_file

        if not img_path.exists():
            continue

        caption = DIAGRAM_CAPTIONS.get(slug, section)
        replacement = (
            f"![{caption}](diagrams/{img_file})\n\n"
            f"*{caption}*\n"
        )
        replacements.append((match.start(), match.end(), replacement))

    if not replacements:
        print(f"[NOOP] {md_path.name} — sin cambios")
        return

    # Apply replacements in reverse order
    for start, end, replacement in reversed(replacements):
        content = content[:start] + replacement + "\n" + content[end:]

    md_path.write_text(content, encoding="utf-8")
    print(f"[OK] {md_path.name} — {len(replacements)} diagramas reemplazados")


def main():
    replace_mermaid_with_images(DOCS / "proceso.md")


if __name__ == "__main__":
    main()
