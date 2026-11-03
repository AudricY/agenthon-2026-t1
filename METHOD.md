# Method

n11 gives the model one call to choose library functions and one request for a whole script, which it keeps as a
fallback. It then runs a loop of up to 20 requests. In each turn the model writes Python files, the agent runs them,
and the agent answers with bounded output and checks on the output files. The code is under `agent/n11/`, with
`agent/common/` for the code it shares with the other agents I built and `agent/model.py` for the model client.

## The task

A Track 1 unit is a quantitative-finance coding task: an instruction, a task card and data files. The platform runs the
image as `solve --task-dir /input --out /app/output` on a read-only copy of the unit with its checks removed. The agent
must write the files the instruction names. Hidden pytest checks then grade them, and a unit passes only if every
check passes. The score is the share of units passed.

The limits shaped the design more than anything else:

- The only model is `nvidia/nemotron-3-super-120b-a12b`, reached through the organisers' proxy. The container has no
  other network.
- A unit may make 25 requests. Each reply is capped at 4,000 tokens, reasoning included. The simplest agent asked for
  a whole script in one reply, and the cap cut that first reply on 54% of 374 unit runs.
- Each card sets its own timeout, from 1,200 to 5,400 s. The organisers' kit holds 86 public units. The Final used
  private units whose number was not published.

## One unit, in order

1. **Reading the unit.** The agent reads the instruction, with its canary lines removed, plus the card and the unit's
   `environment/Dockerfile`. It maps each path the instruction names to the mounted file, builds a digest of the
   data of at most 24,000 bytes, and lists the output files the instruction names (`agent/common/`).
2. **The selection call.** One request shows the task, the input files described and an index of the library. The
   reply names up to 8 library functions to use. Thinking is set low and the reply is capped at 2,000 tokens.
3. **The draft.** One request asks for a complete script, as my first agent `n0` does (`agent/n0/`). The
   script runs for at most 60 s. Its files are held aside, and they are written out at the end only if the loop leaves
   nothing gradable.
4. **The loop.** The system prompt gives the reply format, the working rules, the runtime's package versions and the
   library section. The first message gives the task, the paths, the digest, the input files described, the output
   files and the budget. The model replies with fenced files (```` ```python file=work/main.py ````), `run` lines and
   `finish`. The agent saves each file whole and runs it for at most 90 s. If a reply writes a file and does not
   run it, the agent runs `main.py`. If the reply ends in `finish`, the agent runs the file before taking the finish.
5. **What the model sees back.** A run's observation shows the first and last 1,000 characters of stdout, the last
   2,000 of stderr and at most 8 traceback frames, within 7,000 bytes. After each clean run it also shows the output
   checks and the findings of 14 output rules drawn from the task's stated properties. Observations older than the
   last three turns are cut to their first line. A reply the output cap cut mid-file is continued from the cut.
6. **The finish.** The checks are: the named files exist, they parse, their numbers are finite, no canary appears,
   and the script reports no failure of its own. A finish after a failed run is turned away up to three times while a
   request and 45 s remain. A finish with rule findings is turned away once.

The loop also ends when the 20 requests are spent, when less than 45 s remain, or after four replies in a row with no
command. The clock is the card's timeout less 200 s, and at most 600 s. Every request uses temperature 0, a fixed
seed and at most 4,000 output tokens. Thinking is off everywhere except the selection call.

## The library

`agent/qf/` holds 316 functions, written and tested for this entry (`tests/`). They use numpy and scipy, and one
volatility function also uses `arch`. The library covers option pricing, rates and curves, credit, risk, volatility,
econometrics, Monte Carlo and backtest metrics, among others. Each function's docstring is a card: the quantity,
its conventions and units, and a reference.

The model never calls a lookup tool. The system prompt carries the whole cards of the functions the selection call
picked and of up to five more whose aliases appear in the task's own words. It lists every function name in the
modules those come from, and the other modules by name. If the call picks nothing, the prompt carries the full index
instead. The generated scripts `import qf` like any other package.

## What was tried

I built 16 agents in all. The table gives each one's best count on the kit's units: all 86, or the 58 of them I
used as development units.

| Agent | Idea | Solved |
|---|---|---|
| n0 | One request for a whole script, after a data digest and a path map | 17 of 86 |
| n2 | n0 with low-effort reasoning | 13 of 86 |
| n10 | n0, then anchored edits to one persistent script | 15 of 86 |
| n11 | A loop of small files and runs over a documented library | 25 of 86; 18 of 58 |
| pi | The pi coding agent off the shelf, behind a capping proxy | 11 of 58 |
| w1 | Three drafts at temperature 0.6, filtered by running them, then voted | 10 of 58 |
| n7, n8 | The script in parts, stitched, with patch repairs | 8 of 58 |
| n9 | n0 with a small finance module | 8 of 58 |
| n1 | Compact code, with a cut reply continued | 7 of 58 |
| n3 | An inspection request before the script | 6 of 58 |
| l1 | A contract and a script per reply, the contract checked in code | 5 of 58 |

Four more never ran on a full set. n4 (a plan first) and w2 (three framings at temperature 0) solved 5 and 7 of a
24-unit panel, and n5 and n6 (reasoning budgets) failed their first probes. In the first comparison on the 58
development units, the loop solved 18 against a control's 7.

I then tested changes to n11 in pairs, each against the unchanged agent on a panel of units. Each result below
reads as solved with the change against solved without it. A difference of four on a 16-unit panel counts as noise
here, because a run with nothing changed once moved that panel by four.

Kept:

| Change | Result |
|---|---|
| The selection call | 15 against 12 of 24 |
| No inspection tool; the input files described in the first message | 5 against 1 of 28 |
| An unchanged re-sent file read as the finish | 18 against 15 of 40 |
| Asking again after a reply with no command, with picks limited to named functions | 13 against 11 of 24 |
| One file by default | 12 of 21 one-file runs solved against 5 of 37 split runs |
| A reserved last call, and time kept for the final run | units with no output fell from 13 to 6; solved 17 against 18 |
| The system prompt after an audit removed its contradictory and stale lines | 135 against 121 over 200 paired runs |

Dropped:

| Change | Result |
|---|---|
| A plan step before the loop | 9 against 18 of 58 |
| A scored index in place of the cards | 11 against 15 of 58 |
| A state rebuilt every turn in place of the transcript | 11 against 14 of 24 |
| A verify turn | helped none of the 36 times it ran, and cost 5 solves |
| Shorter system prompts | lost in four reads, by 1 to 11 units |
| Thinking on the first write | 6 against 6 of 16, with six more first scripts cut at the output cap |
| A longer clock | 900 s solved none of the 13 units that ran past 600 s |

## The scores

The image solved 25 of the kit's 86 public units in the last full run before the upload. That run made 551
requests and cost $1.36 on a hosted route.

Its rehearsals solved about 50 of 86, but on a different set: a mixed set I built later, of 58 units converted from
open task pull requests to `QF-Bench/QuantitativeFinance-Bench` plus 28 of the kit's units. The converted units turned
out easier. They carry fewer hidden checks, a median of 8 against the kit's 28. Among units with ten or fewer checks,
the image solved 27 of 31 converted units and 4 of 18 of the kit's. Compare the kit's figure, 25 of 86, not the mixed
set's.

In the Final, Codabench scored the upload at 0.1071. The organisers' verification rerun scored it at 0.1786.
