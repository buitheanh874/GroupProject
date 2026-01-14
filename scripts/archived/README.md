# Archived Scripts

This directory contains scripts that are no longer actively used in production but are preserved for reference.

## Archived on: 2026-01-10

### Debug Tools (No longer used)

**`run_sumo_episode.py`** (2,114 bytes)
- **Purpose**: Simple debug tool to run N cycles with fixed action
- **Reason archived**: Basic single-TLS only, not used in recent development
- **Replacement**: Use `smoke_baseline.py` or `eval.py` for testing

**`diagnose_episode.py`** (3,917 bytes)
- **Purpose**: More complete debug tool with KPI tracking and diagnostics
- **Reason archived**: Not used in recent development
- **Replacement**: Use `scripts/doctor.py` for diagnostics or `eval.py` for full evaluation

### Demo/Example Scripts

**`plot_results.py`** (3,641 bytes)
- **Purpose**: Demo plotting script with hardcoded CSV data
- **Reason archived**: Contains fake/example data, not for production use
- **Replacement**: Use `plot_eval.py` or `plot_kpis.py` for real data visualization



---

## Note

These files are kept in git history and can be restored if needed. They were archived during a cleanup session to reduce clutter in the main scripts directory.

If you need to use any of these scripts:
1. They should still work (no breaking changes to dependencies)
2. You may need to update config paths
3. Consider whether newer tools are more appropriate

For questions, see cleanup reports in `.gemini/antigravity/brain/` artifacts.
