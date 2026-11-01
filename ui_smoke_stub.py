"""Executes app.py top to bottom with a FAKE streamlit (MagicMock) to catch NameError/TypeError/KeyError
in the interface code. This is NOT a real browser test: it cannot check layout or widgets.
Run:  python tests/ui_smoke_stub.py"""
import os, sys, types
from unittest.mock import MagicMock
import matplotlib
matplotlib.use("Agg")
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root)

st = MagicMock(name="streamlit")
st.cache_resource = lambda *a, **k: (lambda f: f)
st.sidebar.file_uploader.return_value = None
st.sidebar.text_input.return_value = ""
st.sidebar.checkbox.return_value = True            # use synthetic demo
def make_col():
    c = MagicMock()
    c.selectbox.side_effect = lambda label, options, index=0, *a, **k: list(options)[index]
    return c
st.columns.side_effect = lambda n, *a, **k: [make_col() for _ in range(n if isinstance(n, int) else len(n))]
st.slider.return_value = 0
st.radio.side_effect = lambda label, options, *a, **k: list(options)[int(os.environ.get("RADIO_IDX", 0)) % len(list(options))]
st.selectbox.side_effect = lambda label, options, index=0, *a, **k: list(options)[index]
st.stop.side_effect = SystemExit
st.expander.return_value.__enter__ = lambda s: s
st.expander.return_value.__exit__ = lambda s, *a: False
sys.modules["streamlit"] = st
src = open(os.path.join(root, "app.py"), encoding="utf-8").read()
for idx in (0, 1, 2):                                # mixed / benign / attack traffic windows
    os.environ["RADIO_IDX"] = str(idx)
    st.reset_mock(side_effect=False)
    exec(compile(src, "app.py", "exec"), {"__name__": "__main__"})
    calls = [c[0] for c in st.method_calls]
    assert any(c.startswith("pyplot") for c in calls), "figure was not drawn"
    assert any(c.startswith("table") for c in calls), "comparison table was not drawn"
print("UI smoke test (fake streamlit) passed for 3 traffic windows")
