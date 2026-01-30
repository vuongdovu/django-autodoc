# django-autodoc
Automatic documentation for Django projects using Mermaid diagrams.

## Quick start
1. Update `autodoc.toml` with your project root and purpose.
2. Run `python file_extractor.py --config autodoc.toml`.

Outputs:
- `docs/mermaid.md` for the full project view.
- `docs/models/<app>_models.md` for per-app model diagrams.
- `docs/summaries/` for AI summaries (when enabled).

## AI summaries (OpenAI)
Enable `[ai]` in `autodoc.toml` and export your API key:

```bash
export OPENAI_API_KEY="..."
```

Then run the extractor. It will generate:
- `docs/summaries/apps/<app>.md`
- `docs/summaries/models/<app>_<model>.md`
- `docs/summaries/views/<app>_<view>.md`

## Config
`autodoc.toml` supports:
- `project_root`: Django project root to scan.
- `project_purpose`: Used in the generated docs.
- `output_dir`, `mermaid_doc`, `model_diagrams_dir`: Output locations.
- `ai.*`: OpenAI summary settings and output directories.
- `include_apps`/`exclude_apps`: Filter which apps to document.
- `ignore_paths`: Skip directories like `venv` or `node_modules`.
- `mermaid.direction`: Diagram direction (`LR`, `TB`, etc).
