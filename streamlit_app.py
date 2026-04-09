import io
import json
import os
import re
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple
from urllib.parse import urljoin, urlparse

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import requests
import streamlit as st
from bs4 import BeautifulSoup
from google import genai
from google.genai import types
from PIL import Image, UnidentifiedImageError
from scipy.spatial.distance import cosine


@dataclass
class AuditTarget:
    url: str
    query: Optional[str] = None
    client: Optional[str] = None


def slugify(value: str) -> str:
    value = (value or "").strip().lower()
    value = re.sub(r"https?://", "", value)
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_")[:80] or "target"


def parse_targets(raw: str, default_client: str, global_query: str) -> List[AuditTarget]:
    targets: List[AuditTarget] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) == 1:
            targets.append(AuditTarget(url=parts[0], query=global_query or None, client=default_client))
        elif len(parts) == 2:
            targets.append(AuditTarget(url=parts[0], query=parts[1] or global_query or None, client=default_client))
        else:
            targets.append(
                AuditTarget(
                    url=parts[0],
                    query=parts[1] or global_query or None,
                    client=parts[2] or default_client,
                )
            )
    return targets


def zip_directory_bytes(folder: Path) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for file_path in folder.rglob("*"):
            if file_path.is_file():
                zf.write(file_path, arcname=str(file_path.relative_to(folder)))
    buffer.seek(0)
    return buffer.getvalue()


PLOTLY_PREMIUM_TEMPLATE = {
    "layout": {
        "paper_bgcolor": "#FBFBFD",
        "plot_bgcolor": "#FBFBFD",
        "font": {"family": "Inter, Segoe UI, sans-serif", "color": "#0F172A", "size": 14},
        "title": {"font": {"size": 22, "color": "#0F172A"}, "x": 0.02},
        "legend": {"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0.02},
        "margin": {"l": 55, "r": 35, "t": 90, "b": 60},
        "colorway": ["#2563EB", "#14B8A6", "#F59E0B", "#EF4444", "#8B5CF6", "#10B981"],
        "xaxis": {
            "gridcolor": "rgba(15, 23, 42, 0.08)",
            "zerolinecolor": "rgba(15, 23, 42, 0.12)",
            "linecolor": "rgba(15, 23, 42, 0.25)",
            "ticks": "outside",
        },
        "yaxis": {
            "gridcolor": "rgba(15, 23, 42, 0.08)",
            "zerolinecolor": "rgba(15, 23, 42, 0.12)",
            "linecolor": "rgba(15, 23, 42, 0.25)",
            "ticks": "outside",
        },
    }
}
pio.templates["geo_premium"] = go.layout.Template(PLOTLY_PREMIUM_TEMPLATE)


def style_figure(fig: go.Figure, title: str | None = None, subtitle: str | None = None) -> go.Figure:
    if title:
        title_text = f"<b>{title}</b>"
        if subtitle:
            title_text += f"<br><span style='font-size:14px;color:#475569'>{subtitle}</span>"
        fig.update_layout(title=title_text)
    fig.update_layout(template="geo_premium")
    return fig


def setup_credentials_from_json_bytes(data: bytes) -> str:
    creds = json.loads(data.decode("utf-8"))
    if "project_id" not in creds:
        raise ValueError("credentials.json must contain project_id")

    app_dir = Path(".streamlit_runtime")
    app_dir.mkdir(parents=True, exist_ok=True)
    cred_path = app_dir / "credentials.json"
    cred_path.write_bytes(data)

    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(cred_path.resolve())
    os.environ["GOOGLE_CLOUD_PROJECT"] = creds["project_id"]
    return creds["project_id"]


def setup_credentials_from_file(path: str) -> str:
    cred_path = Path(path).expanduser().resolve()
    if not cred_path.exists():
        raise FileNotFoundError(f"credentials file not found: {cred_path}")

    creds = json.loads(cred_path.read_text(encoding="utf-8"))
    if "project_id" not in creds:
        raise ValueError("credentials.json must contain project_id")

    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(cred_path)
    os.environ["GOOGLE_CLOUD_PROJECT"] = creds["project_id"]
    return creds["project_id"]


class GeoMultimodalAuditor:
    def __init__(self, project_id: str, location: str = "us-central1"):
        self.client = genai.Client(vertexai=True, project=project_id, location=location)
        self.embed_model = "gemini-embedding-2-preview"
        self.gen_model = "gemini-2.5-flash"
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        }
        self.supported_image_formats = {"JPEG", "PNG", "WEBP", "HEIC"}

    def detect_page_intent(self, html_content: str) -> str:
        prompt = (
            "You are a senior SEO quality rater. Analyze this HTML and return the core search intent "
            "or primary entity as a short phrase only."
        )
        response = self.client.models.generate_content(
            model=self.gen_model,
            contents=[prompt, html_content[:20000]],
        )
        return response.text.strip()

    def _validate_image(self, url: str) -> Tuple[Optional[bytes], Optional[str]]:
        try:
            r = requests.get(url, headers=self.headers, timeout=10, stream=True)
            if r.status_code != 200 or len(r.content) < 2048 or len(r.content) > 5 * 1024 * 1024:
                return None, None

            img = Image.open(io.BytesIO(r.content))
            if img.format not in self.supported_image_formats:
                if img.mode in ("RGBA", "P"):
                    img = img.convert("RGB")
                img_byte_arr = io.BytesIO()
                img.save(img_byte_arr, format="JPEG")
                return img_byte_arr.getvalue(), "image/jpeg"

            return r.content, f"image/{img.format.lower()}"
        except (requests.exceptions.RequestException, OSError, UnidentifiedImageError):
            return None, None

    def extract_assets(self, url: str):
        r = requests.get(url, headers=self.headers, timeout=20)
        r.raise_for_status()
        soup = BeautifulSoup(r.content, "html.parser")
        raw_html = r.text

        for tag in soup(["script", "style", "nav", "footer", "aside", "noscript"]):
            tag.decompose()

        items = []
        position = 1
        current_heading = "Page Intro"
        current_text = []

        for el in soup.find_all(["h1", "h2", "h3", "p", "img", "video", "audio", "a"]):
            if el.name in ["h1", "h2", "h3"]:
                if current_text and len(" ".join(current_text)) > 50:
                    items.append(
                        {
                            "type": "html",
                            "label": re.sub(r"\s+", " ", current_heading)[:50],
                            "position": position,
                            "content": f"{current_heading}: {' '.join(current_text)}",
                        }
                    )
                    position += 1
                current_heading = el.get_text(" ", strip=True)
                current_text = []

            elif el.name == "p":
                current_text.append(el.get_text(" ", strip=True))

            elif el.name == "img" and el.get("src"):
                img_url = urljoin(url, el.get("src"))
                img_bytes, mime = self._validate_image(img_url)
                context_text = f"{current_heading}: {' '.join(current_text[-1:])} [Alt: {el.get('alt', '')}]"

                if img_bytes:
                    items.append(
                        {
                            "type": "image",
                            "label": re.sub(r"\s+", " ", el.get("alt", "No Alt"))[:40],
                            "position": position + 0.1,
                            "mode": "bytes",
                            "bytes": img_bytes,
                            "mime": mime,
                            "content": context_text,
                        }
                    )
                else:
                    items.append(
                        {
                            "type": "image_metadata",
                            "label": "Image Metadata",
                            "position": position + 0.1,
                            "mode": "text",
                            "content": context_text,
                        }
                    )

            elif el.name in ["video", "audio"]:
                src = el.get("src") or (el.find("source") and el.find("source").get("src"))
                if src:
                    items.append(
                        {
                            "type": el.name,
                            "label": f"{el.name} asset",
                            "position": position + 0.2,
                            "mode": "text",
                            "content": f"{current_heading}: {el.name} media present at {urljoin(url, src)}",
                        }
                    )

            elif el.name == "a" and el.get("href", "").lower().endswith(".pdf"):
                pdf_url = urljoin(url, el.get("href"))
                items.append(
                    {
                        "type": "pdf",
                        "label": re.sub(r"\s+", " ", el.get_text(" ", strip=True) or "PDF")[:40],
                        "position": position + 0.3,
                        "mode": "text",
                        "content": f"{current_heading}: PDF linked -> {pdf_url}",
                    }
                )

        if current_text and len(" ".join(current_text)) > 50:
            items.append(
                {
                    "type": "html",
                    "label": re.sub(r"\s+", " ", current_heading)[:50],
                    "position": position,
                    "content": f"{current_heading}: {' '.join(current_text)}",
                }
            )

        title = soup.title.string.strip() if soup.title and soup.title.string else url
        return items, title, raw_html

    def score_assets(self, target_query: str, items: list) -> pd.DataFrame:
        query_vec = self.client.models.embed_content(
            model=self.embed_model,
            contents=f"task: search result | {target_query}",
        ).embeddings[0].values

        scored_items = []
        for item in items:
            try:
                if item.get("mode") == "bytes":
                    content = types.Content(
                        parts=[
                            types.Part.from_text(text=item["content"]),
                            types.Part.from_bytes(data=item["bytes"], mime_type=item["mime"]),
                        ]
                    )
                    vec = self.client.models.embed_content(model=self.embed_model, contents=[content]).embeddings[0].values
                else:
                    vec = self.client.models.embed_content(
                        model=self.embed_model,
                        contents=f"task: RETRIEVAL_DOCUMENT | {item['content']}",
                    ).embeddings[0].values

                item["score"] = float(1 - cosine(query_vec, vec))
                scored_items.append(item)
            except Exception:
                continue

        return pd.DataFrame(scored_items)

    def generate_reports(self, df: pd.DataFrame, client_name: str, target_query: str, output_dir: str):
        if df.empty:
            raise ValueError("No scored assets found for this URL.")

        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        prefix = out_path / slugify(client_name)

        weights = {
            "html": 0.45,
            "image": 0.25,
            "image_metadata": 0.10,
            "video": 0.10,
            "audio": 0.05,
            "pdf": 0.05,
        }

        df = df.copy()
        df["weight"] = df["type"].map(weights).fillna(0.02)
        df["gap"] = (0.75 - df["score"]).clip(lower=0)
        df["priority"] = (df["gap"] * df["weight"]).round(4)
        global_score = float(np.average(df["score"], weights=df["weight"]))

        score_pct = round(global_score * 100, 2)
        fig_global = go.Figure(
            go.Indicator(
                mode="gauge+number+delta",
                value=score_pct,
                number={"suffix": " / 100", "font": {"size": 54}},
                delta={"reference": 75},
                title={
                    "text": f"<b>Global Multimodal GEO Score</b><br><span style='font-size:16px'>{target_query}</span>",
                    "font": {"size": 22},
                },
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#0F172A", "thickness": 0.3},
                    "steps": [
                        {"range": [0, 50], "color": "#FDE2E1"},
                        {"range": [50, 75], "color": "#FFF4CC"},
                        {"range": [75, 90], "color": "#D8F5DD"},
                        {"range": [90, 100], "color": "#C7F9CC"},
                    ],
                    "threshold": {"line": {"color": "#D93025", "width": 4}, "value": 75},
                },
            )
        )
        fig_global = style_figure(fig_global, "Global Multimodal GEO Score", target_query)
        fig_global.update_layout(margin=dict(l=40, r=40, t=120, b=40))

        by_type = df.groupby("type", as_index=False).agg(
            mean_score=("score", "mean"),
            asset_count=("score", "size"),
            mean_gap=("gap", "mean"),
        )
        by_type["target"] = 0.75

        fig_radar = go.Figure()
        fig_radar.add_trace(
            go.Scatterpolar(
                r=by_type["mean_score"],
                theta=by_type["type"],
                fill="toself",
                name="Mean Score",
                line=dict(color="#2563EB", width=3),
            )
        )
        fig_radar.add_trace(
            go.Scatterpolar(
                r=by_type["target"],
                theta=by_type["type"],
                fill="none",
                name="Target 0.75",
                line=dict(color="#DC2626", width=2, dash="dash"),
            )
        )
        fig_radar = style_figure(fig_radar, "Empreinte Multimodale par Type", "Score moyen vs cible 0.75")
        fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])))

        df_html = df[df["type"] == "html"].sort_values("position").copy()
        fig_flow = None
        if not df_html.empty:
            df_html["rolling_mean"] = df_html["score"].rolling(window=3, min_periods=1).mean()
            fig_flow = go.Figure()
            fig_flow.add_trace(
                go.Scatter(
                    x=df_html["label"],
                    y=df_html["score"],
                    mode="markers+lines",
                    name="Passages",
                    marker=dict(size=8, color="#6B7280"),
                    line=dict(color="#9CA3AF", width=1),
                )
            )
            fig_flow.add_trace(
                go.Scatter(
                    x=df_html["label"],
                    y=df_html["rolling_mean"],
                    mode="lines",
                    name="Trend",
                    line=dict(color="#2563EB", width=4),
                )
            )
            fig_flow.add_hline(y=0.75, line_dash="dash", line_color="#DC2626")
            fig_flow = style_figure(fig_flow, "Flux Sémantique des Passages", "Passage Indexing et tendance mobile")
            fig_flow.update_layout(xaxis_tickangle=-45)

        fig_box = px.box(
            df,
            x="type",
            y="score",
            color="type",
            points="all",
            hover_data=["label", "gap", "priority"],
            template="plotly_white",
            title="Distribution de pertinence sémantique par type",
        )
        fig_box = style_figure(fig_box, "Distribution de pertinence sémantique par type", "Vue multimodale par asset")

        backlog = df[df["gap"] > 0].sort_values(["priority", "gap"], ascending=False)
        fig_priority = px.scatter(
            backlog.head(30),
            x="gap",
            y="score",
            color="type",
            size="priority",
            hover_data=["label"],
            template="plotly_white",
            title="Priority Matrix (Top 30)",
        ) if not backlog.empty else None
        if fig_priority is not None:
            fig_priority = style_figure(fig_priority, "Matrice de priorisation", "Top 30 actions à traiter en premier")

        image_export_errors: List[str] = []

        def safe_write_image(fig, path: str, width: int, height: int, scale: int = 2):
            try:
                fig.write_image(path, width=width, height=height, scale=scale)
            except Exception as e:
                image_export_errors.append(f"{Path(path).name}: {e}")

        # Write artifacts per URL
        fig_global.write_html(str(prefix) + "_00_global_score.html")
        safe_write_image(fig_global, str(prefix) + "_00_global_score.png", width=1100, height=620, scale=2)

        fig_radar.write_html(str(prefix) + "_01_multimodal_radar.html")
        safe_write_image(fig_radar, str(prefix) + "_01_multimodal_radar.png", width=900, height=700, scale=2)

        if fig_flow is not None:
            fig_flow.write_html(str(prefix) + "_02_semantic_flow.html")
            safe_write_image(fig_flow, str(prefix) + "_02_semantic_flow.png", width=1300, height=650, scale=2)

        fig_box.write_html(str(prefix) + "_03_multimodal_distribution.html")
        safe_write_image(fig_box, str(prefix) + "_03_multimodal_distribution.png", width=1100, height=650, scale=2)

        if fig_priority is not None:
            fig_priority.write_html(str(prefix) + "_05_priority_matrix.html")
            safe_write_image(fig_priority, str(prefix) + "_05_priority_matrix.png", width=1100, height=650, scale=2)

        backlog[["type", "label", "score", "gap", "weight", "priority", "content"]].to_csv(
            str(prefix) + "_04_action_backlog.csv", index=False
        )
        by_type.to_csv(str(prefix) + "_06_type_diagnostics.csv", index=False)
        df.to_csv(str(prefix) + "_07_all_scored_assets.csv", index=False)

        return {
            "global_score": global_score,
            "by_type": by_type,
            "df": df,
            "fig_global": fig_global,
            "fig_radar": fig_radar,
            "fig_flow": fig_flow,
            "fig_box": fig_box,
            "fig_priority": fig_priority,
            "backlog": backlog,
            "output_dir": str(out_path),
            "image_export_errors": image_export_errors,
        }


st.set_page_config(page_title="Multimodal GEO Auditor", page_icon="📊", layout="wide")
st.title("Multimodal GEO Auditor")
st.caption("Batch URL audit with per-URL reports, charts, and CSV exports")

with st.sidebar:
    st.header("Configuration")
    location = st.text_input("Vertex location", value="us-central1")
    default_client = st.text_input("Default client name", value="Client Demo")
    output_root = st.text_input("Output root folder", value="streamlit_outputs")

    st.subheader("Credentials")
    cred_mode = st.radio("Credential source", ["Local file", "Upload JSON"], horizontal=True)
    local_cred_path = st.text_input("Local credentials path", value="credentials.json")
    uploaded_cred = st.file_uploader("Upload credentials.json", type=["json"])

st.markdown("### Targets")
st.markdown(
    "Enter one target per line in one of these formats:\n"
    "- `url`\n"
    "- `url|query`\n"
    "- `url|query|client`"
)
default_targets = "https://www.searchenginejournal.com/what-is-seo/\nhttps://www.riverly.com/bateaux/"
target_lines = st.text_area("Batch targets", value=default_targets, height=160)
global_query = st.text_input("Global query (optional)", value="")

run = st.button("Run Multimodal Audit", type="primary")

if run:
    try:
        if cred_mode == "Upload JSON":
            if uploaded_cred is None:
                st.error("Please upload credentials.json")
                st.stop()
            project_id = setup_credentials_from_json_bytes(uploaded_cred.read())
        else:
            project_id = setup_credentials_from_file(local_cred_path)

        targets = parse_targets(target_lines, default_client=default_client, global_query=global_query)
        if not targets:
            st.error("No valid targets found")
            st.stop()

        run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")
        run_dir = Path(output_root) / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        st.success(f"Project: {project_id}")
        st.info(f"Run directory: {run_dir}")

        auditor = GeoMultimodalAuditor(project_id=project_id, location=location)
        summary_rows = []

        progress = st.progress(0)
        for idx, t in enumerate(targets, start=1):
            with st.status(f"Auditing {t.url}", expanded=False) as status:
                url_slug = slugify(urlparse(t.url).netloc + urlparse(t.url).path)
                client_name = t.client or default_client
                target_dir = run_dir / f"{slugify(client_name)}__{url_slug}"

                items, page_title, raw_html = auditor.extract_assets(t.url)
                resolved_query = t.query
                if not resolved_query:
                    resolved_query = auditor.detect_page_intent(raw_html)

                df_results = auditor.score_assets(resolved_query, items)
                reports = auditor.generate_reports(
                    df=df_results,
                    client_name=client_name,
                    target_query=resolved_query,
                    output_dir=str(target_dir),
                )

                summary_rows.append(
                    {
                        "url": t.url,
                        "page_title": page_title,
                        "client": client_name,
                        "resolved_query": resolved_query,
                        "assets_scored": len(df_results),
                        "global_score": round(reports["global_score"], 4),
                        "png_export_errors": len(reports["image_export_errors"]),
                        "output_dir": str(target_dir),
                    }
                )

                status.update(label=f"Done: {t.url}", state="complete")

                with st.expander(f"Results: {t.url}", expanded=False):
                    c1, c2 = st.columns(2)
                    c1.metric("Global GEO score", f"{reports['global_score'] * 100:.2f}/100")
                    c2.write(f"Output folder: `{target_dir}`")
                    if reports["image_export_errors"]:
                        st.warning(
                            "PNG export skipped for some charts (Kaleido/Chrome runtime not available). "
                            "HTML charts and CSV files were still generated successfully."
                        )
                    st.plotly_chart(reports["fig_global"], use_container_width=True, theme=None)
                    st.plotly_chart(reports["fig_radar"], use_container_width=True, theme=None)
                    if reports["fig_flow"] is not None:
                        st.plotly_chart(reports["fig_flow"], use_container_width=True, theme=None)
                    st.plotly_chart(reports["fig_box"], use_container_width=True, theme=None)
                    if reports["fig_priority"] is not None:
                        st.plotly_chart(reports["fig_priority"], use_container_width=True, theme=None)
                    st.dataframe(reports["by_type"], use_container_width=True)
                    st.dataframe(reports["backlog"].head(20), use_container_width=True)

            progress.progress(idx / len(targets))

        summary_df = pd.DataFrame(summary_rows).sort_values("global_score", ascending=False)
        summary_csv_path = run_dir / "run_summary.csv"
        summary_df.to_csv(summary_csv_path, index=False)

        st.subheader("Batch Summary")
        st.dataframe(summary_df, use_container_width=True)

        st.download_button(
            "Download run summary CSV",
            data=summary_df.to_csv(index=False).encode("utf-8"),
            file_name="run_summary.csv",
            mime="text/csv",
        )

        run_zip = zip_directory_bytes(run_dir)
        st.download_button(
            "Download full run (ZIP)",
            data=run_zip,
            file_name=f"{run_id}.zip",
            mime="application/zip",
        )

        st.success("Batch audit completed successfully.")
        st.write(f"All files saved under: `{run_dir}`")

    except Exception as e:
        st.exception(e)
