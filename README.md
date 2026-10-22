# Dataset Analyser

A web app built in Python with Streamlit. You load any dataset, get an instant profile of it, then filter it, chart it and ask questions in plain language. Anyone with the link can use it in a browser. No Claude account is needed, and the AI is optional.

## What it does

| Area | Details |
|---|---|
| **Sources** | Upload CSV, TSV, TXT, Excel (every sheet), ODS, JSON, JSON Lines, XML or Parquet. You can also fetch any **URL or REST API** (with optional auth header, query parameters and a records path), a public **Google Sheet**, a **GitHub** file or a **Dropbox** link. Pasting text copied from Excel works too. |
| **Cleaning** | Detects number, date, category, yes/no, ID and text columns. Reads European and US number formats (`1.234,50`, `$1,200`, `(300)`, `15%`), day-first and month-first dates, quarters (`2024-Q1`) and Excel serial dates. Flattens nested JSON. |
| **Overview** | KPIs with a 3-month trend, automatic charts, a data quality score with issues, and a profile of every column. |
| **Filters** | Detects columns like country, region, product, category, channel and customer, plus date and number ranges. Every filter applies across the whole app. |
| **Explore** | Type “volume per category”, “sales by country top 5” or “monthly profit by region”, or use the chart builder. Chart types are bar, horizontal bar, line, area, pie, doughnut, scatter and histogram. The result table can be downloaded. |
| **Data** | Searchable, sortable table. Download the filtered rows as CSV or Excel. |
| **Insights** | Automatic findings with a recommendation for each: trend, concentration, fastest-growing and declining segments, margin, discount effect, correlations, outliers and data quality. With an API key, you also get a written executive review. |
| **Assistant** | Chat about the data. **Without a key**, a built-in engine answers “measure by column” questions, totals, averages, counts, rankings, shares and trends. It spots values like “Germany” or “2025” in the question and filters to them. It also handles summary, columns, data quality, correlations and insights. **With an Anthropic key**, Claude also answers open questions such as “why is Office declining and what should we do?” Claude runs calculations on the data and draws charts in the chat. |

## Run it on your own computer

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

The app opens at http://localhost:8501 with an example dataset loaded.

## Put it online for free (Streamlit Community Cloud)

1. Create a GitHub repository and upload everything in this folder. The `.streamlit` folder is included, but `secrets.toml` is not.
2. Go to **share.streamlit.io** and sign in with GitHub.
3. Click **Create app**. Pick the repository and branch, and set the main file to `app.py`. You can also choose a custom subdomain, such as `candy-dataset-analyser.streamlit.app`.
4. Optional, to switch on AI answers: open **Advanced settings → Secrets** and paste:
   ```toml
   ANTHROPIC_API_KEY = "sk-ant-..."
   ```
   Every visitor's AI questions are then billed to your key. If you leave it out, the app works without AI. Visitors can also paste their own key in the sidebar under **AI assistant**.
5. Click **Deploy**. After a minute or two you get a public link anyone can open.

Free apps go to sleep after a period without visitors and wake up on the next visit, which takes about 30 seconds.

## Other hosting options

- **Docker** (any cloud, or a company server): `docker build -t dataset-analyser .` then `docker run -p 8501:8501 -e ANTHROPIC_API_KEY=sk-ant-... dataset-analyser`. The app reads `ANTHROPIC_API_KEY` and `ANTHROPIC_MODEL` from environment variables or from `.streamlit/secrets.toml`.
- **Azure App Service / Azure Container Apps**: deploy the Docker image. This is a good fit for companies on Microsoft 365, and access can be restricted with Microsoft Entra ID sign-in.
- **Hugging Face Spaces**: create a Space with the Docker SDK and upload these files.
- **Render or Railway**: connect the GitHub repository and use the Dockerfile.

## Privacy

- Uploaded data is held in the server's memory for that visitor's session only. It is not stored, and other visitors cannot see it.
- With AI switched on, Claude receives the column list, a few example rows and the results of the calculations it requests. It never receives the whole file.
- URL fetching refuses private or internal network addresses.

## Project layout

```
app.py                  Streamlit interface (sidebar, Overview, Explore, Data, Insights, Assistant)
analyser/loaders.py     Reading files, URLs, APIs and pasted text
analyser/profiling.py   Type detection, cleaning, column profiles, quality score
analyser/engine.py      Filters, aggregation, plain-language parser, built-in assistant, insights
analyser/charts.py      Plotly charts
analyser/ai.py          Optional Claude analyst with calculation tools
sample_data/            Example dataset
```
