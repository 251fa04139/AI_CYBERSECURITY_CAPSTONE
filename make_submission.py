"""Builds a neat submission folder and zip (no large CSV data files).
   python make_submission.py            -> Submission_251FA04139/ and Submission_251FA04139.zip
Put your screenshots (PNG/JPG) in a folder called  screenshots/  next to this file first; they are included.
"""
import os, shutil, zipfile
here = os.path.dirname(os.path.abspath(__file__))
name = "Submission_251FA04139"
out = os.path.join(here, name)
if os.path.exists(out):
    shutil.rmtree(out)
plan = {
    "1_Report": ["docs/Project_Report.pdf"],
    "2_Presentation": ["slides/AI_Cybersecurity_Capstone_Slides.pptx", "slides/AI_Cybersecurity_Capstone_Slides.pdf"],
    "3_Poster": ["poster/poster.png", "poster/poster.pdf"],
    "4_Master_Guide": ["docs/Master_Project_Guide.pdf"],
}
missing = []
for folder, files in plan.items():
    os.makedirs(os.path.join(out, folder), exist_ok=True)
    for f in files:
        src = os.path.join(here, f)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(out, folder, os.path.basename(f)))
        else:
            missing.append(f)
code = os.path.join(out, "5_Source_code")
for f in ["README.md", "requirements.txt", "core.py", "app.py", "experiments.py", "apply_results.py", ".gitignore"]:
    if os.path.exists(os.path.join(here, f)):
        os.makedirs(code, exist_ok=True); shutil.copy2(os.path.join(here, f), code)
for d in ["tests", "notebooks", "results", "docs/diagrams"]:
    s = os.path.join(here, d)
    if os.path.isdir(s):
        shutil.copytree(s, os.path.join(code, d), ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "make_diagrams.py" if False else "~"))
shots = os.path.join(here, "screenshots")
if os.path.isdir(shots):
    shutil.copytree(shots, os.path.join(out, "6_Screenshots"))
big = [os.path.join(r, f) for r, _, fs in os.walk(out) for f in fs if f.lower().endswith(".csv") and os.path.getsize(os.path.join(r, f)) > 5_000_000]
assert not big, f"large data file found, remove it: {big}"
zp = os.path.join(here, name + ".zip")
with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
    for r, _, fs in os.walk(out):
        for f in fs:
            p = os.path.join(r, f); z.write(p, os.path.relpath(p, here))
print("created", zp, f"({os.path.getsize(zp)/1e6:.1f} MB)")
if missing: print("WARNING missing:", missing)
print("\nContents:")
for r, _, fs in sorted(os.walk(out)):
    for f in sorted(fs): print("  ", os.path.relpath(os.path.join(r, f), out))
