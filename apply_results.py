"""Copies your REAL metrics into the report, the master guide, the slides and the poster, then rebuilds them.

  1.  python experiments.py real <file1.csv> <file2.csv> <file3.csv>    (writes results/real_metrics.json)
  2.  python experiments.py ablate <same files>                          (optional; writes results/ablation_metrics.json)
  3.  python apply_results.py                                            (rewrites report 7.2, guide G.5, slide 6, poster; rebuilds them)

Needs: Playwright + Chromium (report PDF, guide PDF, poster) and Node + pptxgenjs (slides). A missing tool is skipped with a message.
Numbers are only ever read from the JSON files in results/, never typed or invented.
"""
import json, os, re, subprocess, sys

here = os.path.dirname(os.path.abspath(__file__))
rp = os.path.join(here, "results", "real_metrics.json")
ap = os.path.join(here, "results", "ablation_metrics.json")
if not os.path.exists(rp):
    sys.exit("results/real_metrics.json not found. Run first: python experiments.py real <file1.csv> <file2.csv> <file3.csv>")
data = json.load(open(rp)); meta = data.pop("_meta", {})
ablation = json.load(open(ap)) if os.path.exists(ap) else None
ab_meta = ablation.pop("_meta", {}) if ablation else {}


def friendly(name):
    n = name.lower()
    if "tuesday" in n:
        return "Tuesday (FTP/SSH-Patator)"
    if "wednesday" in n:
        return "Wednesday (DoS)"
    if "friday" in n and "ddos" in n:
        return "Friday afternoon (DDoS)"
    return name


def results_block():
    head = ("| File | Train / test flows | Attack flows in test set | Accuracy | Precision | Recall | F1 | Missed attacks (FN) | False alarms (FP) |\n"
            "|---|---|---|---|---|---|---|---|---|")
    rows = []
    for k, m in data.items():
        att = m["tp"] + m["fn"]
        rows.append(f"| {friendly(k)} | {m['n_train']:,} / {m['n_test']:,} | {att:,} ({att / m['n_test']:.1%}) | {m['accuracy']:.4f} | "
                    f"{m['precision']:.4f} | {m['recall']:.4f} | {m['f1']:.4f} | {m['fn']} | {m['fp']} |")
    src = f"\n\n*How these values entered this document: {meta['source']}. Running `apply_results.py` on your own `results/real_metrics.json` regenerates this section from your file.*" if meta.get("source") else ""
    text = (f"\n## 7.2 Machine-learning results (*Measured by the author*)\n\nProduced by `python experiments.py real ...` on {meta.get('date', 'the run date')}: Random Forest (100 trees, depth 20, balanced class weights), "
            f"at most {meta.get('max_rows', 60000):,} rows sampled per file, {meta.get('split', '70/30 stratified')} split, seed {meta.get('seed', 42)}. "
            f"Every metric is computed on the 30 % of flows the model never saw. Raw values: `results/real_metrics.json`.{src}\n\n"
            + head + "\n" + "\n".join(rows) + "\n\n"
            "**How to read it.** FN = attack flows the model called benign (missed attacks); FP = benign flows it flagged as attacks (false alarms). "
            "The share of attack flows differs per file, so compare FN and FP counts as well as accuracy.\n\n"
            "**Caveats (please state them when you present).**\n\n"
            "* Scores this high are typical for CIC-IDS2017 and are weak evidence that the model would work on other traffic or other days.\n"
            "* The 70/30 split is random over flows of the same day. Near-identical flows from one attack burst can land on both sides, which makes the result optimistic. A split by time, or training on one day and testing on another, would be stricter (proposed, not done).\n"
            "* `Destination Port` is one of the features. The Tuesday attacks target fixed ports (FTP 21, SSH 22), so a perfect score there may mean the model learned the port rather than the attack behaviour. The port-ablation check below tests this.\n"
            "* The Tuesday perfect score rests on a small number of attack flows in the test set (see the table), so it carries more uncertainty than the others.\n"
            "* Each 50,000-row chunk of a file contributes the same number of random rows (at most 60,000 per file), so class shares in the sample differ from the full file.\n")
    if ablation:
        keys = list(ablation)
        rows2 = []
        worst = (0.0, "")
        for k in keys:
            a, b = ablation[k]["with_port"], ablation[k]["without_port"]
            for nm, m in (("with port", a), ("without port", b)):
                rows2.append(f"| {friendly(k)} | {nm} | {m['accuracy']:.4f} | {m['recall']:.4f} | {m['f1']:.4f} | {m['fn']} | {m['fp']} |")
            if a["f1"] - b["f1"] > worst[0]:
                worst = (a["f1"] - b["f1"], friendly(k))
        text += ("\n**Port ablation (measured).** Same pipeline and seed, with and without the `Destination Port` feature "
                 f"(run on {ab_meta.get('date', 'the run date')}).\n\n| File | Setting | Accuracy | Recall | F1 | FN | FP |\n|---|---|---|---|---|---|---|\n"
                 + "\n".join(rows2) + "\n\n"
                 + (f"Largest F1 drop when the port is removed: {worst[0]:.4f} ({worst[1]}). " if worst[0] > 0 else "Removing the port did not lower F1 on any file. ")
                 + "A small change means the model also uses other features; a large drop means the port was doing most of the work.\n")
    else:
        text += ("\n**Port ablation: not run yet.** Run `python experiments.py ablate <the same files>`, then `python apply_results.py`, "
                 "and this paragraph is replaced by a with/without-port table.\n")
    return text


def abstract_sentence():
    f1s = [m["f1"] for m in data.values()]
    return (f"On {len(data)} CIC-IDS2017 files the Random Forest reached an F1 score between {min(f1s):.4f} and {max(f1s):.4f} on flows it had not seen "
            "(section 7.2). Scores this high are typical for this dataset and are weak evidence of general performance.")


def swap(text, tag, new):
    pat = rf"<!--{tag}-->.*?<!--/{tag}-->"
    if not re.search(pat, text, flags=re.S):
        raise KeyError(tag)
    return re.sub(pat, lambda m: f"<!--{tag}-->{new}<!--/{tag}-->", text, flags=re.S)


# ---- report ----
p = os.path.join(here, "docs", "Project_Report.md")
t = open(p, encoding="utf-8").read()
t = swap(t, "ML_RESULTS", results_block())
t = swap(t, "ML_ABSTRACT", abstract_sentence())
t = swap(t, "ML_LIMIT", "* The reported metrics come from one random sample and one seed per file; no repeated-seed, cross-day or time-based test is included.")
open(p, "w", encoding="utf-8").write(t)
print("report text updated")

# ---- guide ----
g = os.path.join(here, "docs", "Master_Project_Guide.md")
if os.path.exists(g):
    t = open(g, encoding="utf-8").read()
    try:
        t = swap(t, "ML_RESULTS_G", results_block().replace("## 7.2 Machine-learning results (*Measured by the author*)\n\n", "", 1))
        open(g, "w", encoding="utf-8").write(t)
        print("guide text updated")
    except KeyError:
        print("guide marker ML_RESULTS_G not found: guide text left unchanged")


def run(cmd, cwd, label):
    try:
        subprocess.run(cmd, cwd=os.path.join(here, cwd), check=True)
        print("rebuilt:", label)
    except Exception as e:
        print("SKIPPED", label, "->", e)


run([sys.executable, "build_pdf.py", "Project_Report.md", "Project_Report.pdf"], "docs", "docs/Project_Report.pdf")
run([sys.executable, "build_pdf.py", "Master_Project_Guide.md", "Master_Project_Guide.pdf"], "docs", "docs/Master_Project_Guide.pdf")
run(["node", "build_deck.js"], "slides", "slides/AI_Cybersecurity_Capstone_Slides.pptx")
run([sys.executable, "make_poster.py"], "poster", "poster/poster.png and poster.pdf")
print("\nStill do by hand: add your own app screenshot (screenshots/ folder), then run python make_submission.py")
