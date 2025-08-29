import markdown

with open("docs/README.md", "r", encoding="utf-8") as f:
    content = f.read()

html = markdown.markdown(
    content,
     extensions=[
        "tables",            # soporte para tablas
        "fenced_code",       # bloques de código con ```
        "codehilite",        # resaltado de sintaxis
        "toc",               # genera tabla de contenidos
        "footnotes",         # soporta notas al pie
        "attr_list",         # atributos extra en HTML ({: .class })
        "def_list",          # listas de definiciones
        "abbr",              # soporta abreviaturas
        "admonition"         # bloques tipo notas/alertas
        ]
    )

with open("docs/README.html", "w", encoding="utf-8") as f:
    f.write(html)
