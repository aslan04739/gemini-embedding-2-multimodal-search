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
    template: Optional[str] = None


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
            targets.append(
                AuditTarget(url=parts[0], query=global_query or None, client=default_client, template="unspecified")
            )
        elif len(parts) == 2:
            targets.append(
                AuditTarget(
                    url=parts[0],
                    query=parts[1] or global_query or None,
                    client=default_client,
                    template="unspecified",
                )
            )
        elif len(parts) == 3:
            targets.append(
                AuditTarget(
                    url=parts[0],
                    query=parts[1] or global_query or None,
                    client=parts[2] or default_client,
                    template="unspecified",
                )
            )
        else:
            targets.append(
                AuditTarget(
                    url=parts[0],
                    query=parts[1] or global_query or None,
                    client=parts[2] or default_client,
                    template=parts[3] or "unspecified",
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


def zip_directory_to_file(folder: Path, zip_path: Path) -> Path:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for file_path in folder.rglob("*"):
            if file_path.is_file():
                zf.write(file_path, arcname=str(file_path.relative_to(folder)))
    return zip_path


def validate_zip_bytes(data: bytes) -> bool:
    try:
        with zipfile.ZipFile(io.BytesIO(data), mode="r") as zf:
            return zf.testzip() is None
    except zipfile.BadZipFile:
        return False


I18N = {
    "fr": {
        "app_title": "Multimodal GEO Auditor",
        "app_caption": "Audit batch d'URLs avec rapports, graphes et exports CSV par page.",
        "language_label": "Langue",
        "config_header": "Configuration",
        "vertex_location": "Region Vertex",
        "default_client": "Nom client par defaut",
        "output_root": "Dossier racine des sorties",
        "credentials_header": "Identifiants",
        "credential_source": "Source des identifiants",
        "auto_mode": "Auto (secrets/env)",
        "upload_mode": "Uploader un JSON d'identifiants",
        "paste_mode": "Coller un JSON d'identifiants",
        "upload_prompt": "Uploader le JSON du compte de service",
        "paste_prompt": "Coller le JSON du compte de service",
        "credentials_tip": "Astuce: le mode Auto lit d'abord Streamlit secrets puis GOOGLE_APPLICATION_CREDENTIALS + GOOGLE_CLOUD_PROJECT.",
        "targets_header": "Cibles",
        "targets_help": "Entrez une cible par ligne: url, url|query, url|query|client, ou url|query|client|template.",
        "batch_targets": "Cibles batch",
        "global_query": "Query globale (optionnel)",
        "run_button": "Lancer l'audit multimodal",
        "err_upload": "Veuillez uploader un fichier JSON d'identifiants.",
        "err_paste": "Veuillez coller votre JSON d'identifiants.",
        "err_credentials": "Identifiants non configures. Utilisez le mode Auto (secrets/env) ou fournissez un JSON en upload/collage.",
        "err_no_targets": "Aucune cible valide detectee.",
        "project": "Projet",
        "run_directory": "Dossier de run",
        "result_prefix": "Resultats",
        "global_metric": "Score GEO global",
        "output_folder": "Dossier de sortie",
        "png_warning": "Export PNG indisponible pour certains graphes (Kaleido/Chrome manquant). Les graphes HTML et CSV ont bien ete generes.",
        "batch_summary": "Synthese batch",
        "best_score": "Meilleur score global",
        "avg_score": "Score global moyen",
        "avg_assets": "Moyenne assets analyses / page",
        "command_center": "Multimodal Command Center",
        "download_summary": "Telecharger la synthese CSV",
        "download_zip": "Telecharger le run complet (ZIP)",
        "success_done": "Audit batch termine avec succes.",
        "saved_under": "Tous les fichiers sont enregistres sous",
        "chart_global_title": "Score GEO multimodal global",
        "chart_global_subtitle": "Score global + detail des ecarts par modalite",
        "chart_radar_title": "Empreinte multimodale par type",
        "chart_radar_subtitle": "Score moyen par type vs cible 0.75",
        "chart_flow_title": "Flux semantique des passages",
        "chart_flow_subtitle": "Passage indexing et tendance mobile",
        "chart_dist_title": "Distribution de pertinence semantique par type",
        "chart_dist_subtitle": "Lecture rapide des points forts et faibles",
        "chart_priority_title": "Matrice de priorisation",
        "chart_priority_subtitle": "Top actions a traiter en premier",
        "chart_modality_board_title": "Tableau des scores par modalite",
        "chart_modality_board_subtitle": "KPI principal: score moyen par type vs cible",
        "chart_footprint_title": "Empreinte multimodale",
        "chart_footprint_subtitle": "Taille = volume d'assets, couleur = qualite",
        "chart_heatmap_title": "Matrice template x modalite",
        "chart_heatmap_subtitle": "Identifier rapidement les modalites faibles par template",
        "chart_template_modality_title": "Template vs modalites",
        "chart_template_modality_subtitle": "Comparer home, category, blog, etc.",
    },
    "en": {
        "app_title": "Multimodal GEO Auditor",
        "app_caption": "Batch URL audit with per-page reports, charts, and CSV exports.",
        "language_label": "Language",
        "config_header": "Configuration",
        "vertex_location": "Vertex location",
        "default_client": "Default client name",
        "output_root": "Output root folder",
        "credentials_header": "Credentials",
        "credential_source": "Credential source",
        "auto_mode": "Auto (secrets/env)",
        "upload_mode": "Upload credentials JSON",
        "paste_mode": "Paste credentials JSON",
        "upload_prompt": "Upload service account JSON",
        "paste_prompt": "Paste service account JSON",
        "credentials_tip": "Tip: Auto mode checks Streamlit secrets first, then GOOGLE_APPLICATION_CREDENTIALS + GOOGLE_CLOUD_PROJECT.",
        "targets_header": "Targets",
        "targets_help": "Enter one target per line: url, url|query, url|query|client, or url|query|client|template.",
        "batch_targets": "Batch targets",
        "global_query": "Global query (optional)",
        "run_button": "Run multimodal audit",
        "err_upload": "Please upload a credentials JSON file.",
        "err_paste": "Please paste your credentials JSON.",
        "err_credentials": "Credentials not configured. Use Auto mode (secrets/env) or provide a JSON via upload/paste.",
        "err_no_targets": "No valid targets found.",
        "project": "Project",
        "run_directory": "Run directory",
        "result_prefix": "Results",
        "global_metric": "Global GEO score",
        "output_folder": "Output folder",
        "png_warning": "PNG export was skipped for some charts (Kaleido/Chrome runtime unavailable). HTML charts and CSV files were generated.",
        "batch_summary": "Batch Summary",
        "best_score": "Best global score",
        "avg_score": "Average global score",
        "avg_assets": "Average assets scored / page",
        "command_center": "Multimodal Command Center",
        "download_summary": "Download summary CSV",
        "download_zip": "Download full run (ZIP)",
        "success_done": "Batch audit completed successfully.",
        "saved_under": "All files saved under",
        "chart_global_title": "Global Multimodal GEO Score",
        "chart_global_subtitle": "Overall score with modality gap breakdown",
        "chart_radar_title": "Multimodal Footprint by Type",
        "chart_radar_subtitle": "Mean score by type vs 0.75 target",
        "chart_flow_title": "Semantic Passage Flow",
        "chart_flow_subtitle": "Passage indexing with rolling trend",
        "chart_dist_title": "Semantic Relevance Distribution by Type",
        "chart_dist_subtitle": "Quick read of strengths and weaknesses",
        "chart_priority_title": "Prioritization Matrix",
        "chart_priority_subtitle": "Top actions to address first",
        "chart_modality_board_title": "Per-Modality Scoreboard",
        "chart_modality_board_subtitle": "Primary KPI: mean score by modality vs target",
        "chart_footprint_title": "Multimodal Footprint",
        "chart_footprint_subtitle": "Tile size = asset volume, color = quality",
        "chart_heatmap_title": "Template x Modality Matrix",
        "chart_heatmap_subtitle": "Spot weak modalities by template quickly",
        "chart_template_modality_title": "Template vs Modalities",
        "chart_template_modality_subtitle": "Compare home, category, blog, etc.",
    },
    "ar": {
        "app_title": "مدقق GEO متعدد الوسائط",
        "app_caption": "تدقيق دفعي لعناوين URL مع تقارير ورسوم وملفات CSV لكل صفحة.",
        "language_label": "اللغة",
        "config_header": "الإعدادات",
        "vertex_location": "منطقة Vertex",
        "default_client": "اسم العميل الافتراضي",
        "output_root": "مجلد المخرجات",
        "credentials_header": "بيانات الاعتماد",
        "credential_source": "مصدر بيانات الاعتماد",
        "auto_mode": "تلقائي (secrets/env)",
        "upload_mode": "رفع ملف JSON",
        "paste_mode": "لصق JSON",
        "upload_prompt": "ارفع JSON لحساب الخدمة",
        "paste_prompt": "الصق JSON لحساب الخدمة",
        "credentials_tip": "معلومة: الوضع التلقائي يقرأ Streamlit secrets ثم GOOGLE_APPLICATION_CREDENTIALS + GOOGLE_CLOUD_PROJECT.",
        "targets_header": "الأهداف",
        "targets_help": "أدخل هدفا في كل سطر: url أو url|query أو url|query|client أو url|query|client|template.",
        "batch_targets": "الأهداف الدفعيّة",
        "global_query": "استعلام عام (اختياري)",
        "run_button": "تشغيل التدقيق متعدد الوسائط",
        "err_upload": "يرجى رفع ملف JSON لبيانات الاعتماد.",
        "err_paste": "يرجى لصق JSON لبيانات الاعتماد.",
        "err_credentials": "بيانات الاعتماد غير مهيأة. استخدم الوضع التلقائي أو ارفع/الصق JSON.",
        "err_no_targets": "لا توجد أهداف صالحة.",
        "project": "المشروع",
        "run_directory": "مجلد التشغيل",
        "result_prefix": "النتائج",
        "global_metric": "النتيجة العامة",
        "output_folder": "مجلد الإخراج",
        "png_warning": "تعذر تصدير PNG لبعض الرسوم (Kaleido/Chrome غير متوفر). تم توليد رسوم HTML وملفات CSV.",
        "batch_summary": "ملخص الدفعة",
        "best_score": "أفضل نتيجة عامة",
        "avg_score": "متوسط النتيجة العامة",
        "avg_assets": "متوسط الأصول المحللة لكل صفحة",
        "command_center": "مركز قيادة تعدد الوسائط",
        "download_summary": "تنزيل ملخص CSV",
        "download_zip": "تنزيل التشغيل الكامل (ZIP)",
        "success_done": "اكتمل التدقيق الدفعي بنجاح.",
        "saved_under": "تم حفظ كل الملفات في",
        "chart_global_title": "النتيجة العامة متعددة الوسائط",
        "chart_global_subtitle": "نتيجة شاملة مع تفكيك الفجوات حسب النوع",
        "chart_radar_title": "البصمة متعددة الوسائط حسب النوع",
        "chart_radar_subtitle": "متوسط النتيجة لكل نوع مقابل هدف 0.75",
        "chart_flow_title": "تدفق المقاطع الدلالي",
        "chart_flow_subtitle": "فهرسة المقاطع مع اتجاه متحرك",
        "chart_dist_title": "توزيع الصلة الدلالية حسب النوع",
        "chart_dist_subtitle": "قراءة سريعة لنقاط القوة والضعف",
        "chart_priority_title": "مصفوفة الأولويات",
        "chart_priority_subtitle": "أهم الإجراءات ذات الأولوية",
        "chart_modality_board_title": "لوحة النتائج حسب الوسيط",
        "chart_modality_board_subtitle": "المؤشر الرئيسي: متوسط النتيجة لكل وسيط مقابل الهدف",
        "chart_footprint_title": "البصمة متعددة الوسائط",
        "chart_footprint_subtitle": "الحجم = عدد الأصول، اللون = الجودة",
        "chart_heatmap_title": "مصفوفة القالب × الوسيط",
        "chart_heatmap_subtitle": "اكتشف نقاط الضعف حسب القالب بسرعة",
        "chart_template_modality_title": "القالب مقابل الوسائط",
        "chart_template_modality_subtitle": "قارن الرئيسية/التصنيف/المقال...",
    },
}


def tr(lang: str, key: str) -> str:
    return I18N.get(lang, I18N["en"]).get(key, I18N["en"].get(key, key))


PLOTLY_PREMIUM_TEMPLATE = {
    "layout": {
        "paper_bgcolor": "#FBFBFD",
        "plot_bgcolor": "#FBFBFD",
        "font": {"family": "Cairo, Tajawal, Inter, Segoe UI, sans-serif", "color": "#0F172A", "size": 14},
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
        "uniformtext": {"minsize": 9, "mode": "hide"},
    }
}
pio.templates["geo_premium"] = go.layout.Template(PLOTLY_PREMIUM_TEMPLATE)
# Enforce a light default for all Plotly figures, including exported HTML/PNG files.
pio.templates.default = "geo_premium"


def style_figure(fig: go.Figure, title: str | None = None, subtitle: str | None = None) -> go.Figure:
    if title:
        title_text = f"<b>{title}</b>"
        if subtitle:
            title_text += f"<br><span style='font-size:13px;color:#475569'>{subtitle}</span>"
        fig.update_layout(
            title=dict(
                text=title_text,
                x=0.02,
                xanchor="left",
                y=0.985,
                yanchor="top",
                pad=dict(t=8, b=24),
            )
        )
    fig.update_layout(
        template="geo_premium",
        height=600,
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font=dict(color="#0F172A"),
        hoverlabel=dict(bgcolor="#FFFFFF", font_color="#0F172A", bordercolor="#CBD5E1"),
        legend=dict(bgcolor="rgba(255,255,255,0.86)", bordercolor="#E2E8F0", borderwidth=1),
        polar=dict(
            bgcolor="#FFFFFF",
            radialaxis=dict(
                gridcolor="rgba(15, 23, 42, 0.10)",
                linecolor="rgba(15, 23, 42, 0.25)",
                tickfont=dict(color="#0F172A"),
            ),
            angularaxis=dict(
                gridcolor="rgba(15, 23, 42, 0.10)",
                linecolor="rgba(15, 23, 42, 0.25)",
                tickfont=dict(color="#0F172A"),
            ),
        ),
        margin=dict(l=55, r=35, t=140, b=70),
    )
    fig.update_xaxes(automargin=True)
    fig.update_yaxes(automargin=True)
    return fig


def build_notice_figure(title: str, subtitle: str, message: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        text=message,
        showarrow=False,
        font=dict(size=16, color="#334155"),
        align="center",
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig = style_figure(fig, title, subtitle)
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


def setup_credentials_from_streamlit_secrets() -> Optional[str]:
    """App-managed auth for Streamlit Cloud (no end-user upload needed)."""
    if "gcp_service_account" in st.secrets:
        secret_obj = dict(st.secrets["gcp_service_account"])
        return setup_credentials_from_json_bytes(json.dumps(secret_obj).encode("utf-8"))

    if "credentials_json" in st.secrets:
        raw = st.secrets["credentials_json"]
        if isinstance(raw, str):
            return setup_credentials_from_json_bytes(raw.encode("utf-8"))

    return None


def setup_credentials_from_environment() -> Optional[str]:
    project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
    credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if project_id and credentials_path:
        return project_id
    return None


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

    def generate_reports(self, df: pd.DataFrame, client_name: str, target_query: str, output_dir: str, lang: str = "en"):
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
        fig_global = style_figure(fig_global, tr(lang, "chart_global_title"), tr(lang, "chart_global_subtitle"))
        fig_global.update_layout(margin=dict(l=40, r=40, t=120, b=40))

        by_type = df.groupby("type", as_index=False).agg(
            mean_score=("score", "mean"),
            asset_count=("score", "size"),
            mean_gap=("gap", "mean"),
        )
        by_type["target"] = 0.75

        fig_type_breakdown = go.Figure()
        fig_type_breakdown.add_trace(
            go.Bar(
                x=by_type["mean_score"],
                y=by_type["type"],
                orientation="h",
                marker=dict(color=by_type["mean_score"], colorscale="RdYlGn", cmin=0, cmax=1),
                name="Mean score",
            )
        )
        fig_type_breakdown.add_trace(
            go.Scatter(
                x=by_type["target"],
                y=by_type["type"],
                mode="markers",
                marker=dict(color="#DC2626", symbol="line-ns-open", size=18),
                name="Target 0.75",
            )
        )
        fig_type_breakdown = style_figure(
            fig_type_breakdown,
            tr(lang, "chart_modality_board_title"),
            tr(lang, "chart_global_subtitle"),
        )
        fig_type_breakdown.update_layout(xaxis=dict(range=[0, 1], title="Score"), yaxis_title="Type")

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
        fig_radar = style_figure(fig_radar, tr(lang, "chart_radar_title"), tr(lang, "chart_radar_subtitle"))
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
            fig_flow = style_figure(fig_flow, tr(lang, "chart_flow_title"), tr(lang, "chart_flow_subtitle"))
            fig_flow.update_layout(xaxis_tickangle=-30)
        else:
            fig_flow = build_notice_figure(
                tr(lang, "chart_flow_title"),
                tr(lang, "chart_flow_subtitle"),
                "Not enough HTML passages to compute semantic flow for this URL.",
            )

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
        fig_box = style_figure(fig_box, tr(lang, "chart_dist_title"), tr(lang, "chart_dist_subtitle"))
        fig_box.update_layout(xaxis_tickangle=-20)

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
            fig_priority = style_figure(fig_priority, tr(lang, "chart_priority_title"), tr(lang, "chart_priority_subtitle"))
        else:
            fig_priority = build_notice_figure(
                tr(lang, "chart_priority_title"),
                tr(lang, "chart_priority_subtitle"),
                "No positive gaps found, so no priority points to display.",
            )

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

        fig_type_breakdown.write_html(str(prefix) + "_01b_modality_breakdown.html")
        safe_write_image(fig_type_breakdown, str(prefix) + "_01b_modality_breakdown.png", width=1000, height=600, scale=2)

        fig_flow.write_html(str(prefix) + "_02_semantic_flow.html")
        safe_write_image(fig_flow, str(prefix) + "_02_semantic_flow.png", width=1300, height=650, scale=2)

        fig_box.write_html(str(prefix) + "_03_multimodal_distribution.html")
        safe_write_image(fig_box, str(prefix) + "_03_multimodal_distribution.png", width=1100, height=650, scale=2)

        fig_priority.write_html(str(prefix) + "_05_priority_matrix.html")
        safe_write_image(fig_priority, str(prefix) + "_05_priority_matrix.png", width=1100, height=650, scale=2)

        backlog[["type", "label", "score", "gap", "weight", "priority", "content"]].to_csv(
            str(prefix) + "_04_action_backlog.csv", index=False
        )
        by_type.to_csv(str(prefix) + "_06_type_diagnostics.csv", index=False)
        df.to_csv(str(prefix) + "_07_all_scored_assets.csv", index=False)

        # Compatibility outputs to mirror historical livraison_multimodale_unique naming.
        fig_flow.write_html(str(prefix) + "_01_semantic_flow.html")
        safe_write_image(fig_flow, str(prefix) + "_01_semantic_flow.png", width=1300, height=650, scale=2)

        fig_box.write_html(str(prefix) + "_02_multimodal_distribution.html")
        safe_write_image(fig_box, str(prefix) + "_02_multimodal_distribution.png", width=1100, height=650, scale=2)

        # Alias kept for old runs where distribution used index 01.
        safe_write_image(fig_box, str(prefix) + "_01_multimodal_distribution.png", width=1100, height=650, scale=2)

        backlog[["type", "label", "score", "gap", "weight", "priority", "content"]].to_csv(
            str(prefix) + "_03_action_backlog.csv", index=False
        )

        return {
            "global_score": global_score,
            "by_type": by_type,
            "df": df,
            "fig_global": fig_global,
            "fig_radar": fig_radar,
            "fig_type_breakdown": fig_type_breakdown,
            "fig_flow": fig_flow,
            "fig_box": fig_box,
            "fig_priority": fig_priority,
            "backlog": backlog,
            "output_dir": str(out_path),
            "image_export_errors": image_export_errors,
        }


st.set_page_config(page_title="Multimodal GEO Auditor", page_icon="📊", layout="wide")

if "last_run_summary_csv" not in st.session_state:
    st.session_state["last_run_summary_csv"] = None
if "last_run_zip_bytes" not in st.session_state:
    st.session_state["last_run_zip_bytes"] = None
if "last_run_zip_name" not in st.session_state:
    st.session_state["last_run_zip_name"] = None
if "last_run_id" not in st.session_state:
    st.session_state["last_run_id"] = None
if "last_run_dir" not in st.session_state:
    st.session_state["last_run_dir"] = None

language_options = {"Francais": "fr", "English": "en", "العربية": "ar"}
with st.sidebar:
    chosen_label = st.selectbox("Language / Langue / اللغة", options=list(language_options.keys()), index=0)
lang = language_options[chosen_label]

if lang == "ar":
    st.markdown(
        """
        <style>
        [data-testid="stAppViewContainer"] * { font-family: Cairo, Tajawal, Segoe UI, sans-serif; }
        </style>
        """,
        unsafe_allow_html=True,
    )

st.title(tr(lang, "app_title"))
st.caption(tr(lang, "app_caption"))

with st.sidebar:
    st.header(tr(lang, "config_header"))
    location = st.text_input(tr(lang, "vertex_location"), value="us-central1")
    default_client = st.text_input(tr(lang, "default_client"), value="Client Demo")
    output_root = st.text_input(tr(lang, "output_root"), value="streamlit_outputs")
    flat_output_mode = st.checkbox("Flat output mode (save directly in output folder)", value=False)

    st.subheader(tr(lang, "credentials_header"))
    credential_mode = st.radio(
        tr(lang, "credential_source"),
        options=[tr(lang, "auto_mode"), tr(lang, "upload_mode"), tr(lang, "paste_mode")],
        index=0,
    )
    uploaded_credentials = None
    pasted_credentials = ""
    if credential_mode == tr(lang, "upload_mode"):
        uploaded_credentials = st.file_uploader(
            tr(lang, "upload_prompt"),
            type=["json"],
            accept_multiple_files=False,
        )
    elif credential_mode == tr(lang, "paste_mode"):
        pasted_credentials = st.text_area(
            tr(lang, "paste_prompt"),
            value="",
            height=180,
            placeholder='{"type":"service_account", ...}',
        )
    st.caption(tr(lang, "credentials_tip"))

st.markdown(f"### {tr(lang, 'targets_header')}")
st.markdown(tr(lang, "targets_help"))
default_targets = (
    "https://www.searchenginejournal.com/what-is-seo/|seo guide|SEJ|blog_article"
)
target_lines = st.text_area(tr(lang, "batch_targets"), value=default_targets, height=160)
global_query = st.text_input(tr(lang, "global_query"), value="")

run = st.button(tr(lang, "run_button"), type="primary")

if run:
    try:
        project_id = None
        if credential_mode == tr(lang, "upload_mode"):
            if uploaded_credentials is None:
                st.error(tr(lang, "err_upload"))
                st.stop()
            project_id = setup_credentials_from_json_bytes(uploaded_credentials.getvalue())
        elif credential_mode == tr(lang, "paste_mode"):
            if not pasted_credentials.strip():
                st.error(tr(lang, "err_paste"))
                st.stop()
            project_id = setup_credentials_from_json_bytes(pasted_credentials.encode("utf-8"))
        else:
            project_id = setup_credentials_from_streamlit_secrets() or setup_credentials_from_environment()

        if not project_id:
            st.error(tr(lang, "err_credentials"))
            st.stop()

        targets = parse_targets(target_lines, default_client=default_client, global_query=global_query)
        if not targets:
            st.error(tr(lang, "err_no_targets"))
            st.stop()

        run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")
        root_path = Path(output_root)
        run_dir = root_path if flat_output_mode else (root_path / run_id)
        run_dir.mkdir(parents=True, exist_ok=True)

        st.success(f"{tr(lang, 'project')}: {project_id}")
        st.info(f"{tr(lang, 'run_directory')}: {run_dir}")

        auditor = GeoMultimodalAuditor(project_id=project_id, location=location)
        summary_rows = []
        per_type_rows = []

        progress = st.progress(0)
        for idx, t in enumerate(targets, start=1):
            with st.status(f"Auditing {t.url}", expanded=False) as status:
                url_slug = slugify(urlparse(t.url).netloc + urlparse(t.url).path)
                client_name = t.client or default_client
                if flat_output_mode:
                    target_dir = run_dir
                else:
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
                    lang=lang,
                )

                summary_rows.append(
                    {
                        "url": t.url,
                        "page_title": page_title,
                        "client": client_name,
                        "template": t.template or "unspecified",
                        "resolved_query": resolved_query,
                        "assets_scored": len(df_results),
                        "global_score": round(reports["global_score"], 4),
                        "png_export_errors": len(reports["image_export_errors"]),
                        "output_dir": str(target_dir),
                    }
                )

                for row in reports["by_type"].to_dict("records"):
                    per_type_rows.append(
                        {
                            "url": t.url,
                            "template": t.template or "unspecified",
                            "type": row["type"],
                            "mean_score": row["mean_score"],
                            "asset_count": row["asset_count"],
                            "mean_gap": row["mean_gap"],
                        }
                    )

                status.update(label=f"Done: {t.url}", state="complete")

            st.markdown(f"#### Results: {t.url}")
            c1, c2 = st.columns(2)
            c1.metric(tr(lang, "global_metric"), f"{reports['global_score'] * 100:.2f}/100")
            c2.write(f"{tr(lang, 'output_folder')}: `{target_dir}`")
            if reports["image_export_errors"]:
                st.warning(tr(lang, "png_warning"))
            st.plotly_chart(reports["fig_global"], width="stretch", theme=None)
            st.plotly_chart(reports["fig_type_breakdown"], width="stretch", theme=None)
            st.plotly_chart(reports["fig_radar"], width="stretch", theme=None)
            st.plotly_chart(reports["fig_flow"], width="stretch", theme=None)
            st.plotly_chart(reports["fig_box"], width="stretch", theme=None)
            st.plotly_chart(reports["fig_priority"], width="stretch", theme=None)
            st.dataframe(reports["by_type"], width="stretch")
            st.dataframe(reports["backlog"].head(20), width="stretch")

            progress.progress(idx / len(targets))

        summary_df = pd.DataFrame(summary_rows).sort_values("global_score", ascending=False)
        summary_csv_path = run_dir / "run_summary.csv"
        summary_df.to_csv(summary_csv_path, index=False)

        per_type_df = pd.DataFrame(per_type_rows)

        st.subheader(tr(lang, "batch_summary"))
        top_score = float(summary_df["global_score"].max()) if not summary_df.empty else 0.0
        avg_score = float(summary_df["global_score"].mean()) if not summary_df.empty else 0.0
        avg_assets = float(summary_df["assets_scored"].mean()) if not summary_df.empty else 0.0
        c1, c2, c3 = st.columns(3)
        c1.metric(tr(lang, "best_score"), f"{top_score * 100:.1f}/100")
        c2.metric(tr(lang, "avg_score"), f"{avg_score * 100:.1f}/100")
        c3.metric(tr(lang, "avg_assets"), f"{avg_assets:.1f}")
        st.dataframe(summary_df, width="stretch")

        if not per_type_df.empty:
            st.subheader(tr(lang, "command_center"))

            modality_rank = (
                per_type_df.groupby("type", as_index=False)
                .agg(mean_score=("mean_score", "mean"), total_assets=("asset_count", "sum"), mean_gap=("mean_gap", "mean"))
                .sort_values("mean_score", ascending=False)
            )
            modality_rank["target"] = 0.75

            fig_modality_bar = go.Figure()
            fig_modality_bar.add_trace(
                go.Bar(
                    x=modality_rank["type"],
                    y=modality_rank["mean_score"],
                    marker=dict(color=modality_rank["mean_score"], colorscale="RdYlGn", cmin=0, cmax=1),
                    name="Mean score",
                )
            )
            fig_modality_bar.add_trace(
                go.Scatter(
                    x=modality_rank["type"],
                    y=modality_rank["target"],
                    mode="lines+markers",
                    name="Target 0.75",
                    line=dict(color="#DC2626", dash="dash", width=2),
                )
            )
            fig_modality_bar = style_figure(
                fig_modality_bar,
                tr(lang, "chart_modality_board_title"),
                tr(lang, "chart_modality_board_subtitle"),
            )
            fig_modality_bar.update_layout(yaxis=dict(range=[0, 1], title="Mean score"), xaxis_title="Modality")
            fig_modality_bar.update_layout(xaxis_tickangle=-20)

            fig_modality_mix = px.treemap(
                modality_rank,
                path=["type"],
                values="total_assets",
                color="mean_score",
                color_continuous_scale="RdYlGn",
                range_color=(0, 1),
                title=tr(lang, "chart_footprint_title"),
            )
            fig_modality_mix = style_figure(
                fig_modality_mix,
                tr(lang, "chart_footprint_title"),
                tr(lang, "chart_footprint_subtitle"),
            )

            c4, c5 = st.columns(2)
            c4.plotly_chart(fig_modality_bar, width="stretch", theme=None)
            c5.plotly_chart(fig_modality_mix, width="stretch", theme=None)

            template_type = (
                per_type_df.groupby(["template", "type"], as_index=False)
                .agg(mean_score=("mean_score", "mean"), total_assets=("asset_count", "sum"))
                .sort_values(["template", "type"])
            )
            heatmap_data = template_type.pivot(index="template", columns="type", values="mean_score").fillna(0)
            if not heatmap_data.empty:
                fig_heatmap = px.imshow(
                    heatmap_data,
                    color_continuous_scale="RdYlGn",
                    zmin=0,
                    zmax=1,
                    aspect="auto",
                    labels=dict(color="Score", x="Modality", y="Template"),
                    title="Template x Modality Heatmap",
                )
                fig_heatmap = style_figure(
                    fig_heatmap,
                    tr(lang, "chart_heatmap_title"),
                    tr(lang, "chart_heatmap_subtitle"),
                )
                st.plotly_chart(fig_heatmap, width="stretch", theme=None)

            fig_template_modality = px.bar(
                template_type,
                x="template",
                y="mean_score",
                color="type",
                barmode="group",
                hover_data=["total_assets"],
                title="Template vs Modality Scores",
            )
            fig_template_modality = style_figure(
                fig_template_modality,
                tr(lang, "chart_template_modality_title"),
                tr(lang, "chart_template_modality_subtitle"),
            )
            fig_template_modality.update_layout(yaxis=dict(range=[0, 1], title="Mean score"), xaxis_title="Template")
            fig_template_modality.update_layout(xaxis_tickangle=-20)
            st.plotly_chart(fig_template_modality, width="stretch", theme=None)

            st.dataframe(modality_rank, width="stretch")

        zip_bytes = zip_directory_bytes(run_dir)
        if not validate_zip_bytes(zip_bytes):
            raise RuntimeError("Generated ZIP is invalid. Please rerun the audit.")
        st.session_state["last_run_summary_csv"] = summary_df.to_csv(index=False).encode("utf-8")
        st.session_state["last_run_zip_bytes"] = zip_bytes
        st.session_state["last_run_zip_name"] = f"{run_id}.zip"
        st.session_state["last_run_id"] = run_id
        st.session_state["last_run_dir"] = str(run_dir)

        st.success(tr(lang, "success_done"))
        st.write(f"{tr(lang, 'saved_under')}: `{run_dir}`")

    except Exception as e:
        st.exception(e)

if st.session_state.get("last_run_summary_csv") is not None:
    st.download_button(
        tr(lang, "download_summary"),
        data=st.session_state["last_run_summary_csv"],
        file_name="run_summary.csv",
        mime="text/csv",
    )

if st.session_state.get("last_run_zip_bytes") and st.session_state.get("last_run_id"):
    zip_bytes = st.session_state["last_run_zip_bytes"]
    zip_name = st.session_state.get("last_run_zip_name") or f"{st.session_state['last_run_id']}.zip"
    zip_size_mb = len(zip_bytes) / (1024 * 1024)
    st.caption(f"ZIP ready: {zip_name} ({zip_size_mb:.2f} MB)")
    st.download_button(
        tr(lang, "download_zip"),
        data=zip_bytes,
        file_name=zip_name,
        mime="application/zip",
    )
