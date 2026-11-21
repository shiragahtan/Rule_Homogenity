---
name: instructor-update
description: Write the meeting summary or an update email for the project's instructor (Brit) - decisions, status, questions, next steps. Use when the user prepares a meeting, asks for a summary of decisions, or wants an email to the instructor.
---

# Summary or email for the instructor

The summary page lives outside the repo, in the parent folder: `summary_for_brit_rule_selection.html` (short) and
`summary_for_brit_details.html` (long). Update both when results change.

## Style (from the user's feedback)
- English, plain words, bottom lines first. Middle length: not a wall of text, but each point has one line of
  context or method, so it can be understood alone.
- Every question says what was actually done (e.g. which binning method), so the instructor can answer.
- Say clearly which results were computed before a data or code change.
- Do not mention names of outside sources where the user asked not to: the RW weights are "taken from an outside
  source, not calculated by us" (not "Yarden's weights").
- Keep only what the reader needs: no speed or accuracy numbers unless asked, no point repeated twice.

## Email structure
1. Opening: these are the main points we discussed in the previous meeting; please confirm. (If the user cannot
   attend a meeting, say so and that a status update follows by mail.)
2. Numbered decisions, one short paragraph each.
3. Next steps at the end; open questions for the next meeting go there (e.g. computing the RW weights ourselves).
No status section unless the user asks for one.

## Decisions so far (Nov 2026)
- ε per dataset: both homogeneous and non-homogeneous rules, approximately half and half (not exactly); the ε
  closest to Table 3.
- 10 equal-width bins for continuous attributes only; German amount stays binary (0 = below 4,000 DM, 1 = 4,000+).
- RW: 8 predicates drawn by weight, then the largest deviation (`weighted8`); check its effect on the results.
- Size filter: each treatment arm ≥ δ/10 rows, also for the rule's overall effect (asked by Sean, agreed).
- Rule selection procedure: approved (skill `select-rules`).
- Open: how to compute the RW attribute weights.
