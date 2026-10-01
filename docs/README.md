# Documentation

Start with the [quickstart](quickstart.md). See [metrics](metrics.md) for shapes
and masks, [CRDT notes](crdt_notes.md) for solver controls, and
[provenance](provenance.md) for references and licensing.

Build locally with `make docs` after installing `".[docs,geo]"`. Documentation
notebooks execute during the build; Python examples in Markdown are also tested
by pytest. The HTML output is in `docs/_build/html`.
