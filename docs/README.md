# Project Documentation

This folder stores engineering decisions, incident investigations, and experiment notes.

## Structure

- `adr/` - Architecture Decision Records (why a technical decision was made).
- `incidents/` - Debugging investigations and postmortems (what failed, root cause, fix).
- `experiments/` - Short run notes and outcomes (what was tried, what worked, what did not).

## Writing Rules

- Write in English.
- Be precise and concise.
- Prefer facts over opinions.
- Link to evidence when possible (W&B run, log file, commit SHA).
- Include exact commands/config values for reproducibility.
- Do not add generic "all tests passed" statements unless they are directly useful.
- If a step needs manual follow-up, add explicit TODO entries:
  - `TODO:: <what to verify>`
  - `TODO:: <plot/screenshot/link to add>`

## Suggested Workflow

1. Add/update an incident note when a failure is investigated.
2. If the solution changes design or training policy, add an ADR.
3. Record follow-up experiments in `experiments/` with outcome and next step.
4. Leave explicit `TODO::` placeholders for manual checks and missing artifacts.

## Minimal Entry Template

Use this structure in new notes:

- Context
- Evidence
- Root cause
- Decision or fix
- Validation
- Open risks / next actions
