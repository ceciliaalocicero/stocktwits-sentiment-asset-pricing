# Data

No data files are committed. The notebook is saved **with its original outputs**, so all results can be read on GitHub without running it. To re-run it, place the files below in `data/raw/` and `data/cache/`; both folders are excluded by `.gitignore`.

## Raw inputs (`data/raw/`)

| File | Content | Source |
|------|---------|--------|
| `UnlabelledDataset.zip` | StockTwits posts for 20 tickers (one CSV per ticker) | Course-provided scrape (2020); not redistributable |
| `UnlabelledDataset_2.zip` | StockTwits posts for AAPL, AMZN, NFLX, QQQ, TSLA | Course-provided scrape (2020); not redistributable |
| `crsp_sp500_2007_2023.csv.gz` | CRSP daily file, S&P 500 constituents (CIZ format) | WRDS, licensed; see the companion CAPM repository for the field list |

Each StockTwits CSV has the columns `symbol`, `message`, `datetime` (UTC), `user` (numeric ID) and `message_id`.

Downloaded automatically by the notebook (internet required):
- **StockEmotions** labelled posts, from the authors' GitHub repository (`adlnlp/StockEmotions`);
- **daily prices** from Yahoo Finance via `yfinance`;
- **Fama–French 5 factors** (monthly) from the Ken French Data Library.

## Cached intermediate files (`data/cache/`)

These files let the notebook skip slow or rate-limited steps. Do not commit them: they contain StockTwits message IDs or derived text.

| File | What it stores | How it is produced |
|------|----------------|--------------------|
| `fintwitbert_scores.parquet` | FinTwitBERT probabilities for all 5,564,914 posts (`message_id`, `neg`, `pos`, `neu`, `score`) | Separate long-running scoring job, not part of the notebook; the scoring script is not included in this repository |
| `preds_stockemo_<model>.csv` | Benchmark predictions of the three BERT models on StockEmotions | Notebook, about 3 minutes per model |
| `preds_stockemo_llm.csv` | LLM scores on the 1,000-post StockEmotions test set | Notebook + Groq API key |
| `preds_events_llm.csv` | LLM scores for the event-study sample | Notebook + Groq API key |
| `fintwit_sample_10k.pkl` | FinTwitBERT results on the 10,000-post analysis sample | Notebook, about 10 minutes |
| `fintwit_emoji_reclass.pkl` | Re-classification of posts containing emojis | Notebook |
| `llm_zero_1k.pkl`, `llm_few_1k.pkl` | Zero- and few-shot LLM labels on 1,000 posts | Notebook + Groq API key |
| `task4_panel.pkl` | Merged tweet–ticker–sentiment panel | Notebook |

Note: the original notebook reads these files from its working directory. Adjust the paths in the setup cell if you keep them in `data/raw/` and `data/cache/`.
