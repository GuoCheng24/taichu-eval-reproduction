# Redaction of a machine name from the sealed pre-registrations

**2026-09-23.** One machine name was removed from the pre-registration(s) below,
after they were published. This file exists because the alternative — changing a
sealed document quietly — would destroy the only thing a seal is for.

## What was changed

A single six-character token naming a compute node was replaced by a
six-character neutral label. **Nothing else.** The replacement is the same length
as what it replaced, so every other byte in each file is at the byte offset it
was at before, and the number of differing bytes is exactly the token length
times the number of occurrences:

| file | occurrences | bytes changed | expected | length before = after |
|---|---|---|---|---|
| `prereg/PREREG_budget.md` | 1 | 6 | 1 × 6 = 6 | yes |

## Hashes

| file | sealed as | now |
|---|---|---|
| `prereg/PREREG_budget.md` | `3f7d0abbb296f6f6b8e716cefdf57acc3a82c431478ac17fce19ecf90a1baafd` | `d54906b32e08ed10be64a2eff1381dcfe82bce103b6f1df7583c2f92db10974c` |

Anyone holding the original bytes can confirm the first column. The analysis
scripts assert the second.

## What was not changed

No sentence, number, threshold, stopping rule, pre-stated interpretation or
date. The scientific content of a pre-registration is what makes it worth
sealing, and none of it is a machine name.

## What this does not undo

The pre-redaction bytes remain in this repository's **git history**, as they do
for any file that has ever been committed. Removing them there would mean
rewriting history, which is a separate decision with separate consequences and
is not done here silently.
