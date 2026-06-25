import sys, markdown
from weasyprint import HTML

CSS = """
@page { size: A4; margin: 2cm 1.8cm; @bottom-center { content: "GeoCell — " counter(page) " / " counter(pages); font-size: 8pt; color: #888; } }
* { box-sizing: border-box; }
body { font-family: "DejaVu Sans", Helvetica, Arial, sans-serif; font-size: 10pt; line-height: 1.45; color: #1a1a1a; }
h1 { font-size: 20pt; color: #0b3d5c; border-bottom: 3px solid #0b3d5c; padding-bottom: 6px; margin-top: 0.6em; }
h2 { font-size: 14pt; color: #0b3d5c; border-bottom: 1px solid #cfd8dc; padding-bottom: 3px; margin-top: 1.2em; page-break-after: avoid; }
h3 { font-size: 11.5pt; color: #14506e; margin-top: 1em; page-break-after: avoid; }
p, li { orphans: 3; widows: 3; }
code { font-family: "DejaVu Sans Mono", monospace; font-size: 8.5pt; background: #f3f5f7; padding: 1px 4px; border-radius: 3px; }
pre { background: #f3f5f7; border: 1px solid #e0e4e8; border-radius: 5px; padding: 8px 10px; font-size: 8pt; line-height: 1.35; overflow-x: hidden; white-space: pre-wrap; word-wrap: break-word; page-break-inside: avoid; }
pre code { background: none; padding: 0; }
table { border-collapse: collapse; width: 100%; margin: 0.6em 0; font-size: 8.5pt; page-break-inside: avoid; }
th, td { border: 1px solid #cfd8dc; padding: 4px 7px; text-align: left; vertical-align: top; }
th { background: #0b3d5c; color: #fff; font-weight: 600; }
tr:nth-child(even) td { background: #f6f8fa; }
blockquote { border-left: 4px solid #0b3d5c; margin: 0.6em 0; padding: 2px 12px; color: #333; background: #f6f8fa; font-style: italic; }
hr { border: none; border-top: 1px solid #cfd8dc; margin: 1.2em 0; }
strong { color: #0b2a3d; }
a { color: #14506e; text-decoration: none; }
"""

src, out = sys.argv[1], sys.argv[2]
text = open(src, encoding="utf-8").read()
html_body = markdown.markdown(text, extensions=["tables", "fenced_code", "toc", "sane_lists"])
html = f"<html><head><meta charset='utf-8'><style>{CSS}</style></head><body>{html_body}</body></html>"
HTML(string=html).write_pdf(out)
print("wrote", out)
