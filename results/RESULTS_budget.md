# Budget arms — the ambiguity was a token budget, and it closes

Pre-registered in [`prereg/PREREG_budget.md`](../prereg/PREREG_budget.md),
committed before any generation. Numbers in
[`results/metrics_budget.json`](metrics_budget.json).

## The question this answers

The MathVista thinking-on arm reported 82.0% on a fixed 100-item subsample
against a card number of 84.50 — but **11 of those 100 items never closed their
reasoning block**, all at the 3,072-token cap, and the answer extractor credited
**5 of the 11** from a half-written trace. Scored as no-answer instead, three of
four estimator intervals stopped containing 84.50.

The README said what would settle it: *a larger token budget, not a larger
sample* — and added that the machine available was one shared GPU. That second
part stopped being true.

## What an 8,192-token budget does

Same 100 items, same decoding, same extractor. The two arms differ only in
`--max_new`.

| arm | budget | accuracy | unclosed | of those, credited | accuracy if truncated scored wrong |
|---|---|---|---|---|---|
| **T-A** | 3,072 | 79.0% | **15** | 5 | **74.0%** |
| **T-B** | 8,192 | **85.0%** | **2** | **0** | **85.0%** |

**The ambiguity closes.** At 3,072 the verdict depended on how 15 truncated
items were counted — a 5-point spread between the two scorings. At 8,192 only
two items truncate, neither is credited, and **the two scorings give the same
number**. There is one column instead of two.

The median token count is unchanged at 1,044: the budget only ever bound the
tail.

## The control did not reproduce, so the published numbers are not restated

`PREREG_budget.md` made arm T-A the control — the published configuration re-run
on the new accelerator in the new environment — and pre-committed to what
happens if it fails:

> If it does not, T-B is still reported in full, the disagreement is reported
> with it, and **only the T-A vs T-B contrast run tonight is used**.

It did not reproduce item for item:

| T-A against the published run, same budget | |
|---|---|
| outcome identical | **95 / 100** |
| token count identical | **47 / 100** |
| unclosed | 15 against 11 |

So this page compares **T-A with T-B** and does not claim the published numbers
were reproduced. What moved between them is an accelerator model and a
software environment, and a companion measurement in a sister repository found
the same weights and the same seed scoring **1.48 points apart on two different
cards** under greedy decoding, with a same-machine repeat byte-identical on 541
of 541 prompts. This is the same phenomenon, one level up.

The contrast that is licensed: T-B against T-A, **+6.0 points**, outcomes
identical on 94 of 100, token counts identical on 81 of 100 — the 19 that differ
are the items the 3,072 cap was binding on.

## What this does not settle

85.0% is a **subsample** figure. The four estimators in the README exist because
the subsample was a compute limitation, and a subsample number cannot be read
against the card's full-set number without them.

That limitation is now also gone. The full sets are running under
[`prereg/PREREG_fullset.md`](../prereg/PREREG_fullset.md) — all 1,000 MathVista
testmini items and all 2,638 CV-Bench items, thinking on, at 8,192 — and they
will replace the estimators with the quantity the estimators estimate. **This
page will be superseded by that measurement**, which is the point of running it.

## CV-Bench is not re-run at a larger budget

Its thinking-on arm has **1 of 150** items unclosed, so its verdict never
depended on truncation. It is included in the full-set arms for the subsample
question, not for the budget question.
