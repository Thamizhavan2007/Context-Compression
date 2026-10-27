# Context Compression Lab UI — revised

This version keeps four review-focused pages:
1. Overview
2. Results Explorer
3. Experiment Inspector
4. Side-by-side Comparison

The unused Semantic Graph and Methodology & Evidence navigation pages have been removed. The core method equation and concise explanation remain inside an Overview expander.

## Update your existing project

1. Back up your current `app.py` if you made any manual edits.
2. Copy the revised `app.py` and `requirements-ui.txt` into the root of `Context_Compression`, replacing the previous UI files only.
3. Do not replace `src/`, `data/`, or the files inside `results/`.
4. In PowerShell at the project root:

   ```powershell
   .\.venv\Scripts\Activate.ps1
   python -m pip install -r requirements-ui.txt
   streamlit run app.py
   ```

5. If Streamlit is already running, save `app.py`; the browser usually reloads automatically. Otherwise stop it with Ctrl+C and launch again.

## Improvements
- Cleaner, more compact layout.
- Overview chart filters to one dataset at a time so the chart is not clipped horizontally.
- Improved CSV aliases for `refs`, `qa_f1_score`, `f1_score`, and generated-answer columns.
- Side-by-side display uses compact chunk IDs instead of tall lists.
- Four focused pages; no separate Semantic Graph or Methodology navigation items.
- All scores and answers still come from saved results. No values are invented.
