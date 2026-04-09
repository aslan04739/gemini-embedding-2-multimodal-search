# Multimodal GEO Auditor (Streamlit)

This project turns your notebook audit into a Streamlit tool that can process multiple URLs in one run.
Each URL gets its own folder with charts, CSV files, and diagnostics.

## What it does

- Batch audits multiple URLs in one click
- Uses Gemini embedding and optional reverse-intent detection
- Extracts and scores multimodal assets:
  - html passages
  - images
  - image metadata fallback
  - video metadata
  - audio metadata
  - pdf links
- Generates per-URL reports:
  - global score gauge (html + png)
  - multimodal radar (html + png)
  - semantic flow (html + png)
  - distribution boxplot (html + png)
  - priority matrix (html + png)
  - action backlog csv
  - type diagnostics csv
  - all scored assets csv
- Generates one run summary CSV for the whole batch

## Inputs format

Use one target per line in the app:

- `url`
- `url|query`
- `url|query|client`

Examples:

```text
https://www.searchenginejournal.com/what-is-seo/
https://www.riverly.com/bateaux/|boat rental no license|Riverly
```

## Local run

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run Streamlit:

```bash
streamlit run streamlit_app.py
```

4. In the app sidebar, configure credentials:
- Local file mode: set `credentials.json` path
- or Upload JSON mode: upload your service-account file

## Output structure

Each run creates a folder under `streamlit_outputs/`:

```text
streamlit_outputs/
  run_YYYYMMDD_HHMMSS/
    run_summary.csv
    client_slug__url_slug/
      client_slug_00_global_score.html
      client_slug_00_global_score.png
      ...
```

## Push to GitHub (step by step)

1. Initialize git (if not already):

```bash
git init
```

2. Check files:

```bash
git status
```

3. Add project files:

```bash
git add streamlit_app.py requirements.txt README.md .gitignore
```

4. Commit:

```bash
git commit -m "feat: add streamlit multimodal auditor with batch URL support"
```

5. Create a new GitHub repo (web UI), then connect remote:

```bash
git remote add origin https://github.com/<your-user>/<your-repo>.git
```

6. Push main branch:

```bash
git branch -M main
git push -u origin main
```

## Deploy options

### Streamlit Community Cloud

1. Push code to GitHub
2. Open Streamlit Community Cloud and connect your repo
3. Set main file to `streamlit_app.py`
4. Add secrets if needed (or use credentials upload mode in UI)

### Docker or VM

- Install Python + dependencies
- Run `streamlit run streamlit_app.py --server.port 8501`

## Security notes

- `credentials.json` is ignored by `.gitignore`
- generated outputs are ignored by default
- never commit service account keys
