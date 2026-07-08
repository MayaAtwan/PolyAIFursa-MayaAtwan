---
name: clean-code
description: Use when writing, reviewing, or refactoring code — especially when deciding how much to change, how to name things, how to structure logic, or when code feels hard to understand or modify
---

# Clean Code

## Overview

Code is read far more than it is written. Write code that is correct, simple, short, readable, and easy to maintain. Prefer the smallest safe change that solves the actual problem.

## Core Rules

**Correctness first.** Code that is elegant but wrong is worse than code that is ugly but right.

**Smallest safe change.** Don't refactor what you don't need to touch. A bug fix is not a licence to restructure surrounding code.

**No speculative generality.** Don't add abstractions for future requirements that don't exist yet. Three similar lines beats a premature helper.

**Name things honestly.** A name should say what a thing *is*, not how it was implemented or who calls it.

**No half-finished implementations.** Either the feature works or don't ship it. Partial code left in with `// TODO` is a trap for the next person.

## What to Check Before You Commit

| Question | If "no", fix it |
|---|---|
| Does it do exactly what was asked — no more, no less? | Scope creep hidden in "improvements" |
| Can you delete any line and nothing breaks? | Unnecessary code |
| Would a new reader understand each name without context? | Rename |
| Is any condition or loop hard to hold in your head at once? | Extract or flatten |
| Did you add error handling for something that can't happen? | Delete it |
| Did you add a comment explaining *what* (not *why*)? | Delete it — good names do that |

## Naming

```python
# ❌ Cryptic, encodes type in name, describes impl
def proc_usr_lst_for_db(u):
    ...

# ✅ Says what it does
def save_users(users):
    ...
```

- Verbs for functions (`fetch`, `save`, `compute`, `render`)
- Nouns for data (`user`, `invoice`, `result`)
- No type suffixes (`userList`, `strName`) — the type system or context handles that
- No filler (`doProcess`, `handleStuff`, `manageData`)

## Structure

**Flat beats nested.** Return early or use guard clauses rather than deep `if/else` chains.

```python
# ❌ Deeply nested
def process(order):
    if order:
        if order.items:
            if order.paid:
                ship(order)

# ✅ Guard clauses
def process(order):
    if not order or not order.items or not order.paid:
        return
    ship(order)
```

**One level of abstraction per function.** Don't mix business logic with string formatting with database calls in the same block.

**Delete dead code.** Don't comment it out and leave it. Git has history.

## Comments

Write no comments by default.

Write one when the **why** is non-obvious: a hidden constraint, a subtle invariant, a workaround for a specific external bug.

```python
# ❌ Narrates what the code says
# Increment counter by 1
count += 1

# ✅ Explains a non-obvious constraint
# Stripe requires idempotency keys ≤ 255 chars; truncate before sending
key = key[:255]
```

## Tests

Clean tests are code too — the same rules apply.

- **One assertion per test** (or one logical scenario). If the test name needs "and", split it.
- **Name the scenario, not the implementation.** `test_checkout_fails_when_cart_is_empty` beats `test_process_returns_false`.
- **No logic in tests.** No loops, no conditionals. If you need them, the test is doing too much.
- **Arrange / Act / Assert** — one block each, in that order, with a blank line between.
- **Don't test implementation details.** Test observable behaviour. If a refactor breaks your test but not your feature, the test was wrong.

```python
# ❌ Tests internals, has logic, unclear name
def test_order():
    for status in ["pending", "paid"]:
        o = Order(status=status)
        assert o._state_machine.current == status  # private

# ✅ Tests behaviour, single scenario, clear name
def test_paid_order_can_be_shipped():
    order = Order(status="paid")
    assert order.shippable()
```

## Existing Style

Match the surrounding code's style — even if you'd do it differently from scratch.

- Indentation, quote style, spacing around operators: copy what's already there.
- If the file uses `snake_case`, don't introduce `camelCase` in your addition.
- If the codebase has no type hints, don't add them only to your new function.

**Why:** A consistent file is easier to read than a correct-but-jarring addition. Style disagreements belong in a linter config, not in a single-function change.

The exception: if there's a project formatter (Black, Prettier, gofmt), run it and let it decide — don't hand-match.

## Public API Safety

Changing a public interface is a breaking change. Treat it with care.

- **Don't rename, remove, or reorder parameters** of any function that callers outside your module depend on.
- **Don't change a return type** in a way that silently breaks callers (e.g. `None` → `[]` may seem safe but changes truthiness).
- **Don't add a required parameter** — add an optional one with a sensible default, or introduce a new function.
- If a change must break the API, make the breakage loud: a different name, a deprecation warning, a major version bump.

| Unsafe (silent breakage) | Safe alternative |
|---|---|
| Rename `get_user(id)` → `fetch_user(id)` | Keep both; deprecate old |
| Remove optional `timeout` param | Keep it; ignore it if unused |
| Return `{}` instead of `None` on miss | Document and version-bump |

## Response Format

Be consistent in how functions communicate results back to callers.

- **Pick one pattern per function family** and stick to it: return value, raise exception, or return `(value, error)` — don't mix.
- **Don't return `None` to signal both "not found" and "error"** — callers can't tell them apart.
- **Empty collection beats `None`** for "no results" (`[]`, `{}`) — callers iterate without a null check.
- **Raise exceptions for unexpected failures; return values for expected absence.**

```python
# ❌ Caller can't distinguish "not found" from "fetch failed"
def get_order(id):
    try:
        return db.query(id)
    except Exception:
        return None

# ✅ Distinguishable outcomes
def get_order(id) -> Order | None:
    return db.query(id)          # None = not found; exception propagates on error
```

## Common Mistakes

| Mistake | Fix |
|---|---|
| Fixing a bug AND refactoring in the same change | Two separate commits |
| Adding a parameter "in case we need it later" | Add it when it's needed |
| Wrapping a working function in a new abstraction "to be safe" | Use the function directly |
| Guarding against `None` inside a function that receives only validated data | Remove the guard |
| Leaving commented-out code | Delete it |
| Writing a 10-line docstring for a 3-line function | Delete the docstring; improve the name |
