# Archived: legacy FastAPI runtime

This directory holds the **retired** FastAPI application (`main.py` + `app/`).

## Why it's archived
Production has always run the **Django** app (`gunicorn ai_study_project.wsgi`
→ `ai_services/views/bridge.py`), managed by PM2 (`deploy/ecosystem.config.js`).
The FastAPI app under `main.py`/`app/` was a parallel re-implementation of the
same AI services and was **not** part of the production request path. Keeping
two runtimes was the main source of fork drift, so it was retired during the
coaching+school vertical unification.

## What moved out before archiving
The only `app/` modules used by production (imported by the Django doubt
endpoint) were relocated, not archived:
- `app/scientific_solver.py`  → `ai_services/solver/scientific_solver.py`
- `app/formula_retriever.py`  → `ai_services/solver/formula_retriever.py`

## Status
- Not imported by any live code path.
- Not copied into the Docker image.
- Retained here (and in git history) for reference only. Safe to delete later.
