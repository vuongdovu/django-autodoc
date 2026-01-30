from __future__ import annotations

import argparse
import ast
import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import tomllib


DEFAULT_CONFIG = {
    "title": "Django Autodoc",
    "project_root": ".",
    "output_dir": "docs",
    "mermaid_doc": "docs/mermaid.md",
    "model_diagrams_dir": "docs/models",
    "project_purpose": "",
    "include_apps": [],
    "exclude_apps": [],
    "ignore_paths": ["venv", ".git", "__pycache__", "node_modules", "dist", "build"],
    "mermaid": {"direction": "LR"},
    "ai": {
        "enabled": False,
        "provider": "openai",
        "model": "gpt-4o-mini",
        "temperature": 0.2,
        "max_tokens": 400,
        "api_key_env": "OPENAI_API_KEY",
        "app_summaries_dir": "docs/summaries/apps",
        "model_summaries_dir": "docs/summaries/models",
        "view_summaries_dir": "docs/summaries/views",
    },
}

RELATION_FIELDS = {
    "ForeignKey": ("*", "1"),
    "OneToOneField": ("1", "1"),
    "ManyToManyField": ("*", "*"),
}

VIEW_BASE_NAMES = {
    "APIView",
    "GenericAPIView",
    "View",
    "TemplateView",
    "DetailView",
    "ListView",
    "CreateView",
    "UpdateView",
    "DeleteView",
    "FormView",
    "ViewSet",
    "GenericViewSet",
    "ModelViewSet",
    "ReadOnlyModelViewSet",
}


@dataclass
class FieldInfo:
    name: str
    field_type: str
    related_model: str | None = None


@dataclass
class ModelInfo:
    name: str
    fields: list[FieldInfo] = field(default_factory=list)


@dataclass
class RelationInfo:
    source: str
    target: str
    rel_type: str
    field_name: str


@dataclass
class ViewInfo:
    name: str
    kind: str
    bases: list[str] = field(default_factory=list)
    methods: list[str] = field(default_factory=list)
    decorators: list[str] = field(default_factory=list)
    docstring: str | None = None


@dataclass
class AppInfo:
    name: str
    path: Path
    models: list[ModelInfo] = field(default_factory=list)
    relations: list[RelationInfo] = field(default_factory=list)
    views: list[ViewInfo] = field(default_factory=list)


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def resolve_path(base_dir: Path, value: str) -> Path:
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate
    return (base_dir / candidate).resolve()


def load_config(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        raise SystemExit(f"Config file not found: {config_path}")
    with config_path.open("rb") as handle:
        data = tomllib.load(handle)
    config = deep_merge(DEFAULT_CONFIG, data)
    base_dir = config_path.parent
    config["project_root"] = resolve_path(base_dir, config["project_root"])
    config["output_dir"] = resolve_path(base_dir, config["output_dir"])
    config["mermaid_doc"] = resolve_path(base_dir, config["mermaid_doc"])
    config["model_diagrams_dir"] = resolve_path(base_dir, config["model_diagrams_dir"])
    ai_config = config.get("ai", {})
    if isinstance(ai_config, dict):
        for key in ("app_summaries_dir", "model_summaries_dir", "view_summaries_dir"):
            if key in ai_config:
                ai_config[key] = resolve_path(base_dir, ai_config[key])
    config["ai"] = ai_config
    return config


def should_ignore(path: Path, ignore_names: set[str]) -> bool:
    return any(part in ignore_names for part in path.parts)


def discover_apps(project_root: Path, ignore_names: set[str]) -> dict[str, Path]:
    apps: dict[str, Path] = {}
    for models_path in project_root.rglob("models.py"):
        if should_ignore(models_path, ignore_names):
            continue
        app_dir = models_path.parent
        apps[app_dir.name] = app_dir
    return apps


def collect_model_bases(tree: ast.AST) -> tuple[set[str], set[str]]:
    models_aliases = {"models"}
    model_names = {"Model"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "django.db.models":
                    models_aliases.add(alias.asname or "models")
        elif isinstance(node, ast.ImportFrom):
            if node.module == "django.db":
                for alias in node.names:
                    if alias.name == "models":
                        models_aliases.add(alias.asname or "models")
            if node.module == "django.db.models":
                for alias in node.names:
                    if alias.name == "Model":
                        model_names.add(alias.asname or "Model")
    return models_aliases, model_names


def get_call_name(call: ast.Call) -> str | None:
    func = call.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def is_model_class(node: ast.ClassDef, models_aliases: set[str], model_names: set[str]) -> bool:
    for base in node.bases:
        if isinstance(base, ast.Attribute):
            if isinstance(base.value, ast.Name) and base.value.id in models_aliases:
                if base.attr == "Model":
                    return True
        elif isinstance(base, ast.Name):
            if base.id in model_names or base.id == "Model":
                return True
    return False


def stringify_node(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = stringify_node(node.value)
        if prefix:
            return f"{prefix}.{node.attr}"
        return node.attr
    if isinstance(node, ast.Call):
        return stringify_node(node.func)
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def decorator_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Call):
        return stringify_node(node.func)
    return stringify_node(node)


def looks_like_view_base(raw_name: str) -> bool:
    base_name = raw_name.split(".")[-1]
    return base_name in VIEW_BASE_NAMES or base_name.endswith("View") or base_name.endswith(
        "ViewSet"
    )


def normalize_relation_target(raw_target: str, current_model: str) -> str:
    if raw_target.lower() == "self":
        return current_model
    return raw_target


def parse_model_class(
    node: ast.ClassDef, models_aliases: set[str], model_names: set[str]
) -> tuple[ModelInfo, list[RelationInfo]] | None:
    if not is_model_class(node, models_aliases, model_names):
        return None

    model = ModelInfo(name=node.name)
    relations: list[RelationInfo] = []

    for stmt in node.body:
        if isinstance(stmt, ast.Assign):
            if len(stmt.targets) != 1 or not isinstance(stmt.targets[0], ast.Name):
                continue
            field_name = stmt.targets[0].id
            value = stmt.value
        elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
            field_name = stmt.target.id
            value = stmt.value
        else:
            continue

        if not isinstance(value, ast.Call):
            continue

        field_type = get_call_name(value)
        if not field_type:
            continue
        if not (field_type.endswith("Field") or field_type in RELATION_FIELDS):
            continue

        related_model = None
        if field_type in RELATION_FIELDS and value.args:
            raw_target = stringify_node(value.args[0])
            if raw_target:
                related_model = normalize_relation_target(raw_target, node.name)
                relations.append(
                    RelationInfo(
                        source=node.name,
                        target=related_model,
                        rel_type=field_type,
                        field_name=field_name,
                    )
                )

        model.fields.append(
            FieldInfo(name=field_name, field_type=field_type, related_model=related_model)
        )

    return model, relations


def parse_models_file(models_path: Path) -> tuple[list[ModelInfo], list[RelationInfo]]:
    try:
        source = models_path.read_text(encoding="utf-8")
    except OSError:
        return [], []

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return [], []

    models_aliases, model_names = collect_model_bases(tree)
    models: list[ModelInfo] = []
    relations: list[RelationInfo] = []

    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        parsed = parse_model_class(node, models_aliases, model_names)
        if parsed:
            model, model_relations = parsed
            models.append(model)
            relations.extend(model_relations)

    return models, relations


def parse_views_file(views_path: Path) -> list[ViewInfo]:
    try:
        source = views_path.read_text(encoding="utf-8")
    except OSError:
        return []

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    views: list[ViewInfo] = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            bases = [stringify_node(base) for base in node.bases]
            bases = [base for base in bases if base]
            if not any(looks_like_view_base(base) for base in bases):
                continue
            methods = []
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if item.name.startswith("__") and item.name.endswith("__"):
                        continue
                    methods.append(item.name)
            views.append(
                ViewInfo(
                    name=node.name,
                    kind="class",
                    bases=bases,
                    methods=methods,
                    docstring=ast.get_docstring(node),
                )
            )
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name.startswith("_"):
                continue
            decorators = [decorator_name(dec) for dec in node.decorator_list]
            decorators = [name for name in decorators if name]
            is_api_view = any(
                name.split(".")[-1] in {"api_view"} for name in decorators
            )
            if not is_api_view:
                continue
            views.append(
                ViewInfo(
                    name=node.name,
                    kind="function",
                    decorators=decorators,
                    docstring=ast.get_docstring(node),
                )
            )

    return views


def build_project_index(config: dict[str, Any]) -> list[AppInfo]:
    project_root: Path = config["project_root"]
    ignore_names = set(config["ignore_paths"])
    include_apps = set(config["include_apps"])
    exclude_apps = set(config["exclude_apps"])

    apps = []
    discovered = discover_apps(project_root, ignore_names)
    for app_name, app_path in sorted(discovered.items()):
        if include_apps and app_name not in include_apps:
            continue
        if app_name in exclude_apps:
            continue
        models_path = app_path / "models.py"
        models, relations = parse_models_file(models_path)
        views: list[ViewInfo] = []
        for view_file in ("views.py", "api.py"):
            candidate = app_path / view_file
            if candidate.exists():
                views.extend(parse_views_file(candidate))
        apps.append(
            AppInfo(
                name=app_name,
                path=app_path,
                models=models,
                relations=relations,
                views=views,
            )
        )

    return apps


def normalize_model_name(raw_name: str) -> str:
    if "." in raw_name:
        parts = raw_name.split(".")
        raw_name = "_".join(part for part in parts if part)
    safe = []
    for char in raw_name:
        if char.isalnum() or char == "_":
            safe.append(char)
        else:
            safe.append("_")
    normalized = "".join(safe)
    return normalized or "Model"


def render_app_mermaid(app: AppInfo, direction: str) -> str:
    lines = ["```mermaid", "classDiagram"]
    if direction:
        lines.append(f"direction {direction}")

    model_ids = {model.name: normalize_model_name(model.name) for model in app.models}
    external_models: set[str] = set()
    rendered_relations: list[tuple[str, str, str, str]] = []

    for relation in app.relations:
        source_id = model_ids.get(relation.source, normalize_model_name(relation.source))
        target_raw = relation.target
        target_id = model_ids.get(target_raw, normalize_model_name(target_raw))
        if target_raw not in model_ids:
            external_models.add(target_raw)
        rendered_relations.append((source_id, target_id, relation.rel_type, relation.field_name))

    for model in app.models:
        model_id = model_ids[model.name]
        lines.append(f"class {model_id} {{")
        for field in model.fields:
            field_type = field.field_type
            if field.related_model:
                field_type = f"{field_type}({normalize_model_name(field.related_model)})"
            lines.append(f"  +{field.name}: {field_type}")
        lines.append("}")

    for external in sorted(external_models):
        lines.append(f"class {normalize_model_name(external)}")

    for source_id, target_id, rel_type, field_name in rendered_relations:
        source_card, target_card = RELATION_FIELDS.get(rel_type, ("*", "*"))
        lines.append(
            f"{source_id} \"{source_card}\" --> \"{target_card}\" {target_id} : {field_name}"
        )

    lines.append("```")
    return "\n".join(lines)


def render_mermaid_doc(config: dict[str, Any], apps: Iterable[AppInfo]) -> str:
    title = config.get("title", "Django Autodoc")
    purpose = config.get("project_purpose", "")
    mermaid_config = config.get("mermaid", {})
    direction = mermaid_config.get("direction", "LR")

    lines = [f"# {title}"]
    if purpose:
        lines.append("")
        lines.append(f"Purpose: {purpose}")

    lines.append("")
    lines.append("## Apps")
    for app in apps:
        lines.append(f"- {app.name}")

    lines.append("")
    lines.append("## Model Diagrams")
    for app in apps:
        lines.append("")
        lines.append(f"### {app.name}")
        lines.append(render_app_mermaid(app, direction))

    return "\n".join(lines)


def render_app_doc(app: AppInfo, direction: str) -> str:
    lines = [f"# {app.name} Models", ""]
    lines.append(render_app_mermaid(app, direction))
    return "\n".join(lines)


def write_docs(config: dict[str, Any], apps: list[AppInfo]) -> None:
    output_dir: Path = config["output_dir"]
    mermaid_doc: Path = config["mermaid_doc"]
    model_diagrams_dir: Path = config["model_diagrams_dir"]
    mermaid_config = config.get("mermaid", {})
    direction = mermaid_config.get("direction", "LR")

    output_dir.mkdir(parents=True, exist_ok=True)
    mermaid_doc.parent.mkdir(parents=True, exist_ok=True)
    model_diagrams_dir.mkdir(parents=True, exist_ok=True)

    mermaid_content = render_mermaid_doc(config, apps)
    mermaid_doc.write_text(mermaid_content, encoding="utf-8")

    for app in apps:
        app_doc = render_app_doc(app, direction)
        app_path = model_diagrams_dir / f"{app.name}_models.md"
        app_path.write_text(app_doc, encoding="utf-8")


def openai_chat_completion(
    api_key: str,
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    max_tokens: int,
) -> str:
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    request = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "ignore")
        raise SystemExit(
            f"OpenAI request failed ({exc.code}): {detail}"
        ) from exc
    except urllib.error.URLError as exc:
        raise SystemExit(f"OpenAI request failed: {exc.reason}") from exc

    data = json.loads(body)
    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise SystemExit("OpenAI response missing content.") from exc
    return content.strip()


def normalize_filename(raw_name: str) -> str:
    return normalize_model_name(raw_name)


def build_model_facts(app: AppInfo, model: ModelInfo) -> dict[str, Any]:
    relationships = []
    for relation in app.relations:
        if relation.source == model.name or relation.target == model.name:
            relationships.append(
                {
                    "source": relation.source,
                    "target": relation.target,
                    "type": relation.rel_type,
                    "field_name": relation.field_name,
                }
            )
    return {
        "app": app.name,
        "model": model.name,
        "fields": [
            {
                "name": field.name,
                "type": field.field_type,
                "related_model": field.related_model,
            }
            for field in model.fields
        ],
        "relationships": relationships,
    }


def build_view_facts(app: AppInfo, view: ViewInfo) -> dict[str, Any]:
    return {
        "app": app.name,
        "view": view.name,
        "kind": view.kind,
        "bases": view.bases,
        "methods": view.methods,
        "decorators": view.decorators,
        "docstring": view.docstring or "",
    }


def build_app_facts(app: AppInfo) -> dict[str, Any]:
    return {
        "app": app.name,
        "models": [model.name for model in app.models],
        "views": [view.name for view in app.views],
        "model_count": len(app.models),
        "view_count": len(app.views),
    }


def write_ai_summaries(config: dict[str, Any], apps: list[AppInfo]) -> None:
    ai_config = config.get("ai", {})
    if not ai_config or not ai_config.get("enabled"):
        return
    if ai_config.get("provider") != "openai":
        raise SystemExit("Only OpenAI provider is supported right now.")

    api_key_env = ai_config.get("api_key_env", "OPENAI_API_KEY")
    api_key = os.getenv(api_key_env)
    if not api_key:
        raise SystemExit(f"Missing OpenAI API key in ${api_key_env}.")

    model = ai_config.get("model", "gpt-4o-mini")
    temperature = float(ai_config.get("temperature", 0.2))
    max_tokens = int(ai_config.get("max_tokens", 400))

    app_dir: Path = ai_config["app_summaries_dir"]
    model_dir: Path = ai_config["model_summaries_dir"]
    view_dir: Path = ai_config["view_summaries_dir"]
    app_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    view_dir.mkdir(parents=True, exist_ok=True)

    system_prompt = (
        "You are a documentation assistant. Use only the provided facts. "
        "If a detail is missing, say 'Unknown' or omit it. Be concise."
    )
    purpose = config.get("project_purpose", "")

    for app in apps:
        app_facts = build_app_facts(app)
        app_prompt = (
            "Write a concise Markdown summary for this Django app. "
            "Do not include a title heading. Provide 1 short paragraph "
            "and a bullet list of key points.\n\n"
            f"Project purpose: {purpose or 'Unknown'}\n"
            f"Facts (JSON):\n{json.dumps(app_facts, indent=2)}"
        )
        app_summary = openai_chat_completion(
            api_key,
            model,
            [{"role": "system", "content": system_prompt}, {"role": "user", "content": app_prompt}],
            temperature,
            max_tokens,
        )
        app_path = app_dir / f"{normalize_filename(app.name)}.md"
        app_path.write_text(
            f"# {app.name} App Summary\n\n{app_summary}\n", encoding="utf-8"
        )

        for model_info in app.models:
            model_facts = build_model_facts(app, model_info)
            model_prompt = (
                "Write a concise Markdown summary for this Django model. "
                "Do not include a title heading. Provide 1 short paragraph, "
                "then a bullet list of fields and relationships.\n\n"
                f"Project purpose: {purpose or 'Unknown'}\n"
                f"Facts (JSON):\n{json.dumps(model_facts, indent=2)}"
            )
            model_summary = openai_chat_completion(
                api_key,
                model,
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": model_prompt},
                ],
                temperature,
                max_tokens,
            )
            model_path = model_dir / f"{normalize_filename(app.name)}_{normalize_filename(model_info.name)}.md"
            model_path.write_text(
                f"# {model_info.name} Model Summary\n\n{model_summary}\n",
                encoding="utf-8",
            )

        for view in app.views:
            view_facts = build_view_facts(app, view)
            view_prompt = (
                "Write a concise Markdown summary for this API view. "
                "Do not include a title heading. Provide 1 short paragraph, "
                "then a bullet list of methods or decorators.\n\n"
                f"Project purpose: {purpose or 'Unknown'}\n"
                f"Facts (JSON):\n{json.dumps(view_facts, indent=2)}"
            )
            view_summary = openai_chat_completion(
                api_key,
                model,
                [{"role": "system", "content": system_prompt}, {"role": "user", "content": view_prompt}],
                temperature,
                max_tokens,
            )
            view_path = view_dir / f"{normalize_filename(app.name)}_{normalize_filename(view.name)}.md"
            view_path.write_text(
                f"# {view.name} API View Summary\n\n{view_summary}\n",
                encoding="utf-8",
            )


def export_index_json(apps: list[AppInfo]) -> dict[str, Any]:
    data = []
    for app in apps:
        data.append(
            {
                "name": app.name,
                "path": os.fspath(app.path),
                "models": [
                    {
                        "name": model.name,
                        "fields": [
                            {
                                "name": field.name,
                                "type": field.field_type,
                                "related_model": field.related_model,
                            }
                            for field in model.fields
                        ],
                    }
                    for model in app.models
                ],
                "views": [
                    {
                        "name": view.name,
                        "kind": view.kind,
                        "bases": view.bases,
                        "methods": view.methods,
                        "decorators": view.decorators,
                        "docstring": view.docstring,
                    }
                    for view in app.views
                ],
            }
        )
    return {"apps": data}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Mermaid docs from a Django project."
    )
    parser.add_argument(
        "--config", default="autodoc.toml", help="Path to autodoc config file."
    )
    parser.add_argument(
        "--print-index",
        action="store_true",
        help="Print extracted JSON index to stdout.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    config = load_config(config_path)
    apps = build_project_index(config)
    write_docs(config, apps)
    write_ai_summaries(config, apps)
    if args.print_index:
        print(json.dumps(export_index_json(apps), indent=2))


if __name__ == "__main__":
    main()
