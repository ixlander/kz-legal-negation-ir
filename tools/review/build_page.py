"""Inline review_data.json into the page template -> tools/review/review_page.html (published as an Artifact)."""
from pathlib import Path

HERE = Path(__file__).parent
data = (HERE / "review_data.json").read_text(encoding="utf-8").replace("</", r"<\/")
html = (HERE / "page_template.html").read_text(encoding="utf-8").replace("__DATA__", data)
(HERE / "review_page.html").write_text(html, encoding="utf-8")
print(f"{len(html) / 1024:.0f} KB")
