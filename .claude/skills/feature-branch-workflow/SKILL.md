---
name: feature-branch-workflow
description: Use when committing code, pushing to GitHub, or opening a pull request — especially when deciding which branch to work on, how to stage and commit changes, or how to verify a deploy before opening a PR to main
---

# Feature Branch Workflow

## Overview

Never commit directly to `dev` or `main`. All work lives on a feature branch. Changes flow through `dev` first to verify the deploy, then a PR is opened from the feature branch to `main`. **You never merge into `main` yourself — the PR is opened and left for review.**

## Branch Safety Rule

**Before touching any file, confirm your current branch:**

```bash
git branch --show-current
```

**If the output is `dev` or `main` — STOP.** Create or switch to a feature branch first:

```bash
git checkout -b feature/your-feature-name
# or switch to an existing one
git checkout feature/your-feature-name
```

**Never commit to `dev` or `main` directly. No exceptions.**

## Conflict Rule

**Never resolve merge conflicts yourself.**

When a conflict occurs:
1. Show the conflicting files and the conflicting lines
2. Suggest how to resolve each one
3. Stop — wait for the user to apply the fix

Do not edit conflict markers (`<<<<<<<`, `=======`, `>>>>>>>`) yourself.

## One-Time Setup

Before using `gh` commands, authenticate once in WSL:

```bash
gh auth login
```

Choose: **GitHub.com → HTTPS → Login with a web browser**, copy the one-time code, open https://github.com/login/device in your browser, and paste it.

## The Workflow

### Step 1 — Stage changes

Stage all changed files:
```bash
git add .
```

Or stage only specific files (preferred when changes are unrelated):
```bash
git add path/to/file1 path/to/file2
```

### Step 2 — Commit

```bash
git commit -m "short description of what changed and why"
```

Good commit message: `"fix: skip empty cart before checkout"` — not `"update"` or `"changes"`.

### Step 3 — Merge into dev and push

Switch to `dev`, merge your feature branch into it, then push:

```bash
git checkout dev
git merge feature/your-feature-name
git push origin dev
```

⚠️ **Conflict? STOP.** Do not resolve it. Show the files, suggest the fix, wait for the user.

### Step 4 — Wait for GitHub Actions to pass

Run this to check the status of the latest `dev` workflow:

```bash
wsl -e gh run list --branch dev --limit 3
```

Read the output:
- `completed  success` ✅ → proceed to Step 5
- `in_progress` → wait and re-run the command
- `completed  failure` ❌ → fix the issue on your feature branch, repeat Steps 1–4

Do not open a PR until the `dev` workflow shows `completed  success`.

### Step 5 — Push the feature branch and open PR

```bash
git checkout feature/your-feature-name
git push origin feature/your-feature-name
```

Then open the PR using the `gh` CLI:

```bash
wsl -e gh pr create --title "short description of the change" --body "what changed and why" --base main --head feature/your-feature-name
```

The command will print the PR URL when done.

**Stop here. Do not merge the PR yourself.** Leave it open for review. Merging into `main` is done through the PR — never via `git merge main` or a direct push.

## Quick Reference

| Action | Command |
|---|---|
| Check current branch | `git branch --show-current` |
| Create feature branch | `git checkout -b feature/name` |
| Stage all changes | `git add .` |
| Stage specific files | `git add file1 file2` |
| Commit | `git commit -m "message"` |
| Switch to dev | `git checkout dev` |
| Merge feature → dev | `git merge feature/name` |
| Push dev | `git push origin dev` |
| Push feature branch | `git push origin feature/name` |
| Check Actions status | `wsl -e gh run list --branch dev --limit 3` |
| Open PR | `wsl -e gh pr create --title "..." --body "..." --base main --head feature/name` |

## Common Mistakes

| Mistake | Fix |
|---|---|
| Committing directly on `dev` | `git checkout -b feature/name`, cherry-pick or redo |
| Opening PR before workflow passes | Close it, fix `dev`, re-check Actions |
| Pushing feature branch before `dev` is green | Wait — a failing deploy on `dev` means the code isn't ready |
| Forgetting to merge feature into `dev` before pushing `dev` | `git merge feature/name` while on `dev` |
| Merging `main` into `dev` instead of feature into `dev` | Check `git log --oneline -5` to verify direction |
| Resolving merge conflicts yourself | Show the conflict and suggest the fix — let the user apply it |
| Merging the PR yourself | Leave it open — only merge through the GitHub PR review process |
| Pushing directly to `main` | Never push to `main` directly; always go through a PR |
