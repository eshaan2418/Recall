# M10 questionnaires (paper or offline form; transcribe into `study/data/questionnaires.csv`)

Administer after **each** block. NASA-TLX after all three blocks; SUS, trust and usefulness only after the two
assistant blocks (`recall`, `b2`). Refer to the assistants as "Assistant 1" / "Assistant 2" in the order the
participant meets them (never "Recall" or "baseline"), so the label does not bias ratings.

## A. NASA-TLX (raw, unweighted) — all blocks

Mark a line with 21 ticks (0, 5, …, 100). Column → `tlx_<scale>`.

| column | scale | question | anchors |
|---|---|---|---|
| `tlx_mental` | Mental demand | How mentally demanding was finding the items in this block? | very low – very high |
| `tlx_physical` | Physical demand | How physically demanding was it? | very low – very high |
| `tlx_temporal` | Temporal demand | How hurried or rushed was the pace? | very low – very high |
| `tlx_performance` | Performance | How successful were you in finding the items? | **perfect (0) – failure (100)** |
| `tlx_effort` | Effort | How hard did you have to work to reach your level of performance? | very low – very high |
| `tlx_frustration` | Frustration | How insecure, discouraged, irritated, stressed or annoyed were you? | very low – very high |

Raw TLX (mean of the six) is analysed; the pairwise-weighting step is skipped (Hart, 2006).

## B. System Usability Scale (SUS) — assistant blocks only

1 = strongly disagree … 5 = strongly agree. "the assistant" = the one used in this block. (Brooke, 1996; wording
unchanged apart from "system" → "assistant".)

| column | item |
|---|---|
| `sus_1` | I think that I would like to use this assistant frequently. |
| `sus_2` | I found the assistant unnecessarily complex. |
| `sus_3` | I thought the assistant was easy to use. |
| `sus_4` | I think that I would need the support of a technical person to be able to use this assistant. |
| `sus_5` | I found the various functions in this assistant were well integrated. |
| `sus_6` | I thought there was too much inconsistency in this assistant. |
| `sus_7` | I would imagine that most people would learn to use this assistant very quickly. |
| `sus_8` | I found the assistant very cumbersome to use. |
| `sus_9` | I felt very confident using the assistant. |
| `sus_10` | I needed to learn a lot of things before I could get going with this assistant. |

## C. Trust (3 items) and usefulness (2 items) — assistant blocks only

1 = strongly disagree … 7 = strongly agree.

| column | item |
|---|---|
| `trust_1` | I could rely on the assistant's answers. |
| `trust_2` | I could tell when I should double-check what the assistant told me. |
| `trust_3` | I would trust this assistant to tell me where my things are at home. |
| `useful_1` | The assistant helped me find the items faster than I would have on my own. |
| `useful_2` | I would want an assistant like this on my phone. |

Trust and usefulness are each averaged over their items (short ad-hoc scales; report Cronbach's alpha
descriptively — do not over-interpret with n ≈ 12).

## D. End of session (not analysed statistically; optional)

"Which of the three ways of finding things did you prefer?" — tick one of: Assistant 1 / Assistant 2 / no
assistant / no preference. Record as a tally on the run sheet only. **No free-text comments are recorded**
(free text is a re-identification risk); the experimenter may summarise themes in aggregate in their notebook
without any participant ID.
