# AI-Based Cybersecurity Monitoring & Attack Path Prediction (v3.1)

AI Mini-Capstone 25CS301, Vignan University. Author: Zogbedji Lewlesse Beni (251FA04139).

A Random Forest labels network flows (benign / attack). Its average attack probability becomes a *threat level* that changes the cost of the internet-facing nodes of a simulated 10-node network, and BFS, DFS and A* search the cheapest attack path. See `docs/Master_Project_Guide.pdf` for everything.

## Quick start
```
python -m venv .venv
.venv\Scripts\activate            # Windows   (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
python -m unittest discover -s tests -v     # expect: Ran 14 tests ... OK
python experiments.py search                # rewrites results/
streamlit run app.py                        # then upload e.g. Tuesday-WorkingHours.pcap_ISCX.csv (untick the synthetic demo)
python experiments.py real <Tuesday.csv> <Wednesday.csv> <Friday-DDos.csv>   # real metrics -> results/real_metrics.json
python experiments.py ablate <same three files>             # optional: with vs without the Destination Port feature
python apply_results.py                     # writes them into report, slides, poster and rebuilds
python make_submission.py                   # neat Submission_251FA04139.zip (no big CSV files)
```
`app.py` needs `core.py` in the same folder: always keep the whole project folder together and start Streamlit from inside it.

The CIC-IDS2017 CSV files (`MachineLearningCVE` folder: Tuesday, Wednesday, Friday-Afternoon-DDos; Monday is all benign and will not work) are **not included** (too large). Pass their full paths on the command line or in the app's path box.

## Layout
`core.py` logic | `app.py` Streamlit UI | `experiments.py` experiments | `tests/` unit tests + `ui_smoke_stub.py` | `notebooks/` earlier Kaggle reference notebook (IDS2018/2019, not the source of the results) | `results/` experiment outputs | `docs/` report, guide, diagrams | `slides/` PPTX | `poster/` poster (rebuild with `python poster/make_poster.py`).

## Status (honest)
* Verified by the assistant: search algorithms, 14 unit tests, search experiments, `tests/ui_smoke_stub.py` (runs app.py against a fake Streamlit).
* Author-run: the Streamlit app in a real browser with an uploaded Tuesday file, and `python experiments.py real` on three CIC-IDS2017 files (F1 0.9989 to 1.0000; see `results/real_metrics.json` and report section 7.2). The assistant transcribed the confusion counts from a screenshot and recomputed the metrics.
* Caveats: very high scores are typical for this dataset; the 70/30 split is random within one day; `Destination Port` is a feature. The ablation command tests the last point and has not been run on the real files yet.
* The built-in "synthetic demo" data is for testing only. Never report its numbers.
