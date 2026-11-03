"""N11: a short loop of native tool calls over small files and the method library. `python3 -m n11.solve solve
--task-dir /input --out /app/output`.

The design is docs/research/weak-model-agents.md, section 3, as lane n11's brief changed it (the library shown by an
index and `lib`, no retriever), and the lanes since, each named where its part sits; docs/agent.md's N11 section is
the whole account, its summary keys among it. The House works through five commands (agent/n11/commands.py): `write`
a file under `work/` (any length that arrives whole), `run` a `.py` file there, `inspect` an input file, `lib` the
library's documentation, `finish`. Each answers with an observation built by code under a byte bound, at no request:
a run's exit, the ends of its output, its traceback's frames in `work/`, the output's files, after an exit 0 the
output's digest and the findings (the instruction's own properties and keys and columns it breaks or lacks,
agent/n11/props.py; the sanity checks', agent/n11/sanity.py), the checks (every output file the instruction names
present, parsing, its numbers finite; the output's self-report, common.selfreport) and the budget line (requests and
seconds left, the named files still missing). Several calls in a reply run in order; after a reply's last `write` of
a `.py` file that no `run` follows, the harness runs `work/main.py` where it changed since its last run, or where a
module it imports was written and its last run was not clean (nothing while a module it imports is nowhere), or the
file written where it is not main.py, has a `__main__` block and main.py does not import it, said so (`after_write`).
The harness does every step that needs no model: the path map, the digest, the input files' description, the named
files, the checks, the clock.

- **The forms** (`FORMS`, read once at import by `form_of`: `AGENTHON_N11_FORM`, else the one word of
  `agent/n11/FORM`, else `loop`; any other name refused, `loop` run and the forms named; logged at the unit's start,
  on the summary line and on the usage line, common.house.Client.form). `loop`, the default since lane prune1 (the
  owner's in322 and in326, agenthon-t2: the plan form and its machinery removed, docs/research/plan-failures-1010.md):
  every unit through the loop below, from request 1 (`band=loop`). `review` (lane loop1): under `LOOP_BYTES` of
  instruction, request 1 is N0's byte for byte, its system prompt and user turn both, and N0's reply form; its script
  becomes `work/main.py` whole and runs as N0 runs its first script (`first`, `ran_first`). A reply the cap cut is
  continued (`continued`, lane n11e): kept to its last complete top-level statement before the definition the cut fell
  in, the next request (N0's system prompt, request 1's user turn, the script so far) asks the rest, code only; the
  join (`runner.stitch`) that compiles runs as request 1's script would, its entry checked; one cut again is
  continued while `CONTINUATIONS` allow, else its complete definitions are kept as `work/draft.py` and the rest asked
  in as few files as fit (lane files1). A clean first run (exit 0, the checks passing, no finding) meets the library
  pass where the pointer named functions (`library_pass`, lane n11c: one request carrying their cards and the script,
  the same script asked back with the pointed quantities computed by `qf`; adopted only where it runs as cleanly and
  writes the same files, else the first run's output put back), and then, instead of the unit's end, the review
  (`open_review`): the loop opens on an observation built by code (`review_text`: the run's exit, digest, checks and
  findings, every property the instruction states as the output holds it, the named files, keys and columns present
  or absent, and an ask to check the script against each stated convention and each value against the task), the
  first output copied aside, under `REVIEW_REQUESTS`; at the unit's end the first output is put back unless the
  review's last run that wrote to the output exited 0, passed the checks, met no finding and wrote exactly the first
  run's files (`review_end`). A failed or flagged first run is the loop's first observation, as if the House had
  written and run it there; at `LOOP_BYTES` or more the unit runs the loop from request 1. With the selection call
  armed (lane review2, `review_loop`) every unit runs the loop from request 1, form `loop`'s bodies under the call, and
  the loop's first clean finish opens the review instead of ending the unit (`review_at_finish`); either way the
  review opens only with two requests and two `ASK_S` left (`review_room`, `review_skipped`). `state` (lane state1;
  agent/n11/state.py): every request rebuilt from the harness's state, never a transcript (`state_loop`): the task,
  the paths, the input files described, the cards, the House's files under work/ as they stand, the last run's result
  (`STATE_RESULT`: `full`, `lean` or `minimal`, lane state2), a ledger of one line a turn and the budget; the reply is
  fenced files and the directives `lib` and `finish`; a saved file runs (`state_run`); an identical file resent a
  second time is refused asking another method, a third time ends the unit on the last clean run's output. `sketch`
  (lane plan4; the owner's in316): form `loop` with one closing ask on its first message (`sketch_ask`, `SKETCH_ASK`,
  in shapes, before `CLAUSE` where armed): a short paragraph on how the task will be solved, in what order, and which
  library functions where the task names their methods; `SKETCH` says `turn` (a reply of its own, kept as the House's first turn and answered by
  `SKETCH_NEXT`) or `inline` (the first reply's text before its call). Its library names alone are read
  (`sketch_names`), the cards of those the pointer missed carried by the next message (`sketch_cards`); under form
  `state` with the switch named, the first state closes on the ask and the sketch is every later state's SKETCH
  section (`state_sketched`). `SKETCH_THINKING=low` asks low-effort thinking on the request carrying the ask alone.
- **The loop.** Each request carries the system prompt (`SYSTEM`: the job, the commands, five steps, the size rule in
  shapes, never a count (lane state3; the owner's in281): one work/main.py where one reply holds it, a split into as
  few modules as fit only where a reply would be cut (lane files1), the library rule (`LIBRARY_USE`, lane lib4), the
  runtime card (`api_card`, lane n11e), the conventions (`CONVENTION_LINES`), N0's rules, an example, the library's
  index (`INDEX_MODE`: `full`, or `scored`, lane index2) and, where the pointer names functions, their cards), the
  first user turn (N0's task, paths and digest; every input file described, `INPUTS`: `standard` or `rich`, lane
  inputs1; the output files the instruction names; the time; the budget; the conventions' sentence; the pointer's
  line) and the turns since, the tool results older than the last `KEEP_TURNS` turns cut to their first line.
  Thinking off (but form `review`'s review request under `REVIEW_THINKING=low`, lane inputs1, and the write replies
  under lane thinkwrite1's `WRITE_THINKING`, below), temperature 0, the seed, 4,000 tokens, as N0.
- **The selection call** (lane select2, agent/n11/selection.py; `SELECT`, `AGENTHON_N11_SELECT` or agent/n11/SELECT:
  `off`, today's, or `call`, armed in forms `loop`, `sketch`, `state` and `review`; the owner's in494). Armed, the unit's first
  request is one bounded request naming the library functions a correct solution would call (`selected`): the task,
  its input files described, the scored index and an ask, at the harness's `max_tokens`, thinking `SELECT_THINKING`
  (`off` or `low`). Where it picks, the system prompt carries the cards of the pointer's functions and then its picks' in the
  pointer's cards' place, and in the index's the lines of their shelves with every function by name and the other
  shelves' names; no index. A call that draws no reply or picks nothing leaves today's prompt. `given`, the desk's
  alone (lane given1; never an image's, refused at the start, `select_of`): no request, the unit's list in the picks
  file the harness names read as the call's reply and shaping the prompt the same way. Form `state` (lane state4)
  takes the same section in its system prompt and the same cards as its CARDS section's base (`state_base`).
- **The reply forms** (`MODE`): native `tool_calls` (`tools`), or the fenced fallback (`fenced`). A write the output
  cap cut is never run: its complete definitions are kept as `work/<name>.draft` (`salvage`), and the
  `SHORT_STREAK`-th cut reply in a row asks for one small file; under `CUT=continue` (lane review2; the owner's in675;
  `AGENTHON_N11_CUT` or agent/n11/CUT, `split` today's) a `.py` write's head, to its last complete line, is kept as
  `work/<name>.py.part` instead and the rest asked in `CONTINUE_ASK`'s shape (`hold`, `CUT_HELD`: its last lines
  quoted, no count), the next reply's code stitched under it and written whole as a write of the file (`resumed`; lane
  join1: the head's last lines a rest repeats taken once as the rest has them, a join that does not compile said with
  its junction, `JUNCTION`, and the head held for the rest again), the split returning at the `SHORT_STREAK`-th cut in
  a row (under `JOINASK=change` the asks after a failed join, a rest cut again or the same text are lane joinask1's,
  below); a malformed call is said and asked again, and a second reply in a row with nothing runnable ends the loop; a
  reply with no call is answered `NO_CALL`, never taken for a finish. A route that refuses the tools turns the
  conversation to the fenced form; `MODE=fenced` starts the loop's forms in it (lane fenced1, below).
- **Repeats** (`written`): a text the file already holds is refused, naming the turn that wrote it and what its run
  gave, and the third return to an earlier version among at most three is named a cycle; a run failing on the very
  source line the run before it failed on is answered quoting that line (`same_frame`).
- **The pointer and the gate.** Where the instruction holds an alias a library function's docstring names, the
  loop's first user turn ends with one line naming `qf.<fn>` for that alias, to be called for its quantity (at most
  five functions, agent/n11/library.py); after a run that exits 0 and is flagged or failing, where no file under
  `work/` calls a pointed function, the observation says once that the task names the method (`gate_fired`).
- **The stops:** a `finish` alone ends the unit by the House's choice: the checks failing, the first with a request
  left shows them; the checks passing with a finding standing, one is refused, and the next ends the unit; the checks
  passing and no finding, the first ends it (under lane finishfail1's `FINISHFAIL=on`, below, a finish on a failing
  run with the checks failing is none of these while a request and `ASK_S` remain; under lane finishcheck1's
  `FINISHCHECK=on`, below, one these rules would end the unit on meets the contract first). Then the request cap (`CAP`:
  `REQUESTS` unless `AGENTHON_N11_REQUESTS` or
  agent/n11/REQUESTS names another within `REQUESTS_BOUNDS`, lane inputs1) counted at admission, and the clock: no
  request with under `ASK_S` left, each run at most `RUN_S` and the time left less `ASK_S` while a request could
  follow. Lane clock1's three: a run's limit the whole of the time left where the cap keeps a request's reserve below
  `RUN_S` (`run_limit`, `run_uncapped`); the last call where no named output exists and the budget buys one more
  reply and not two (`LAST_CALL`); the forced ask before the `NO_CALL_CAP`-th reply with no call ends the unit
  (`FORCED`). The output is never cleared: the last run's files stand. The pacing's waits after a 429 come off the
  clock under the desk's `AGENTHON_MODEL_RATE_UNTIMED` (`moved`). The clock itself is the client's (`client.deadline`:
  `model.unit_deadline`, the card's clock less the platform's allowance under the roster cap, 400 s unless the desk's
  `AGENTHON_MODEL_ROSTER_CAP_S`, which a registration's `unit_cap` sets, or an image's /opt/agent/ROSTER_CAP names
  another, lane finalclock1), named with its source on the unit's `unit` log line; the rules above are seconds, the
  same at any clock, and the cap the same 20 unless set (tests/test_n11_clock.py's cases at 1,000 and 1,700 s).
- **The switches**, each read once at import and named on the unit's `settings` or `library` log line: `STATE_RESULT`,
  `SKETCH`, `SKETCH_THINKING`, `INDEX_MODE`, `INPUTS`, `REVIEW_THINKING`, `CAP`, and lane cap1's two (the
  failure-mode read of 10-10): `NOCALL` (`ask`: a call asked after every reply with none, `ask_call`) and
  `SKETCH_PICKS` (`named`: the sketch's picks from `qf.` names and code spans alone), each an `AGENTHON_N11_*`
  variable (src/agenthon_t1/harness.py hands them to an --agent run's container, `FORM_SWITCHES`, and refuses a word
  that is no setting, `SWITCH_WORDS`) or, but `REVIEW_THINKING`, a one-word file under agent/n11/: the image's
  switches are its files, the checkout's the candidate's tree (lane bake1; docs/agent.md, "The image's switches").
- **Closed levers, gone** (lane prune2; the owner's in326): the verify turn (lanes n11e, cap1 and verify1: no solve
  helped, docs/research/loop-wrong-outputs-1011.md, E105 ROW 249), the scaffold and the compact-code rule (lane
  scaffold1: neither moved a unit, docs/research/scaffold-1012.md, E096 ROW 244) and the library's frame (lane
  libframe1: recall lost, docs/research/library-design-1011.md, E102 ROW 245). The finish path is the one the verify
  turn's `off` ran. No file of their names is read; the desk refuses their variables (src/agenthon_t1/harness.py
  `RETIRED`) and agent/Dockerfile a build carrying their files.
- **The platform's pace** (lane pace1; `PACE`, `AGENTHON_N11_PACE` or agent/n11/PACE: `off`, today's, or `adaptive`,
  which acts in forms `loop` and `sketch`). Every loop request's seconds on the clock are kept and their median is the
  pace (`pace`); the requests the clock still buys are the time left over the pace and a run's expected seconds
  (`bought`). Under `adaptive`: the first message, where the clock at the platform's published pace (`PACE_PRIOR_S`,
  no reply yet) buys fewer than `SHORT_CLOCK_REQUESTS`, appends `SHORT_FIRST` before an armed clause (the whole script first, nothing
  looked up first) and form `sketch`'s turn is skipped (`short_start`); after a reply whose measured pace leaves under
  that bound the short-clock mode is entered and never left (`paced_turn`), each turn's observations then closing on
  `SHORT_ASK` (finish where the last run was clean with every named output present, else the script or the fix and
  its run in one reply: `short_ask`); the last call and the forced ask read two more replies by the pace
  (`two_replies`). Nothing about the pace or the requests is said to the model as a number.
- **The last resort** (lane fallback1, agent/n11/fallback.py; `FALLBACK`, `AGENTHON_N11_FALLBACK` or agent/n11/FALLBACK:
  `off`, today's, or `oneshot`, armed in forms `loop` and `sketch`). Armed, a reserve of the clock and one request is
  held off the loop (`left`, `requests_left`) until a run leaves an output the harness could grade (`ran`); where the
  loop stops with nothing to grade (`stop`'s kind: its requests, its clock, replies with no call, a finish), N0's
  one-shot request is sent with what is left, by agent/n0's own code, and its script run once (`last_resort`). While
  the reserve is held, each loop request is bounded by the time left before it (lane fallback2), a request cut there
  ending the loop on its clock, and the reserve is sized after each reply from the unit's observed rate (`ask`).
- **The conventions card** (lane convcard1, agent/n11/conventions.py; `CONVCARD`, `AGENTHON_N11_CONVENTIONS` or
  agent/n11/CONVENTIONS: `off`, today's, or `on`, armed in forms `loop`, `review` and `sketch`; the owner's in616 and
  in618). Under `on` the loop's first message carries, after the output files and before the clock, the instruction's
  convention words (a fixed table: scale, time, windows, returns, estimators, rounding and ordering) each quoted and
  put as a question its code must answer, never an answer, and the parameters files' keys named as given
  (`card_part`); within a bound in the model's tokens never said; absent where nothing fires. Always on, apart from
  the switch: after a run that exits 0, the span finding (an output series' dates short of the inputs' by more than
  the windows the instruction names, or past them) and the scale finding (an output said in percent, basis points or
  as a decimal whose values read in another scale), one line each in the run's observation beside the findings and
  apart from them (commands.Workspace.against): neither blocks a finish nor makes a run unclean.
- **The first message's three switches** (lane first1; the owner's in693, the error ledger's time class), each off
  by default: `CARDS` (`AGENTHON_N11_CARDS` or agent/n11/CARDS: `cut`, today's, or `whole`: the cards the call shaped,
  or the pointer's, whole within `library.WHOLE_TOKENS` with the index, the index trimmed for them first, `made_whole`),
  `INSPECT` (`AGENTHON_N11_INSPECT` or agent/n11/INSPECT: `open`, today's, or `described`: the inspect tool withheld
  while every input is described and nothing is written, `offered`, and a described file's inspect answered from its
  description, `inspected`) and the selection call's bound (`AGENTHON_N11_SELECT_BOUND`, agent/n11/selection.py:
  `short` asks `selection.MAX_TOKENS` with thinking off where the clock is short, `short_clock`).
- **Lane clock1's three of 10-13** (the owner's in707 and in709; reader `time`, docs/research/time-1013.md), each off
  by default and changing no text the model reads before it acts: `LASTCALL` (`AGENTHON_N11_LASTCALL` or
  agent/n11/LASTCALL: `late`, today's, or `early`, the last call due by its send: `by_send`, `send_room`), `SHIM`
  (`AGENTHON_N11_SHIM` or agent/n11/SHIM: `on` runs each script with agent/n11/preload.py before it, agent/n11/shim.py,
  each pre-empted crash said in a line of the run's observation) and `LINT` (`AGENTHON_N11_LINT` or agent/n11/LINT: `on`
  adds a saved file's findings to the write's answer, agent/n11/lint.py). Lane lint2's `LINT2` beside it, apart
  (`AGENTHON_N11_LINT2` or agent/n11/LINT2: `on` adds two narrow findings after the lint's, an imported name shadowed
  and a configuration value never read, agent/n11/lint.py `check2`; keys `LINT2_KEYS`).
- **A wasting loop redirected** (lane redirect1; the owner's in703 and in705, docs/research/requests-1013.md section 6):
  `REDIRECT` (`AGENTHON_N11_REDIRECT` or agent/n11/REDIRECT: `off`, today's, or `write`, armed in forms `loop`,
  `review` and `sketch`). Under `write` the waste state is declared once (`redirected`: nothing written, at least
  `REDIRECT_LOOKS` inspect-only replies so far, any reply at request `REDIRECT_AT` or later); until a file is
  saved `inspect` and `lib` are put away (`offered`, `put_away`), each observation closes on `REDIRECT_ASK`
  (`redirect_ask`) and a look still asked gets `REDIRECT_HELD` in one line; and a reply at the cap is a cut whatever
  finish the route said (`at_cap`); each reply at the cap counted by its shape (`cut_shape`) and each reply to a
  continuation ask by what it held (`continuation_read`; under `write` a re-sent head's repeat dropped, `past_head`).
  Keys `REDIRECT_KEYS`.
- **The system prompt follows the tools** (lane putaway1; the owner's in739 and in743). Under `INSPECT=described` a
  request that withholds `inspect` sends the system prompt without its line (`said`, `said_tools`, `told`: its line in
  `TOOL_LINES`, the tools form's opening `TOOLS_OPEN`, How to work's `STEP_ONE`); under `REDIRECT=write`, from the
  declaration until a file is saved, the lines of the tools put away become one sentence saying so (`PUT_AWAY`), and
  where both act it names both; a third word, `INSPECT=never`, withholds `inspect` from every request in either form,
  a look still asked answered as under `described`. With both switches unset every body is today's. Keys
  `SYSTEM_KEYS`.
- **The prompt as a switch** (lane prompt1; the owner's in90 and in91, RULING ROW 605; the cold reader's designs A,
  B and C, runs/lanes/cold-prompt1017.md): `PROMPT` (`AGENTHON_N11_PROMPT` or agent/n11/PROMPT: `full`, today's,
  `lean`, `taskfirst`, `constants`, `norules`, `rulesafter`, `minimal`, `leanconst` or `clean`, armed in form `loop`
  where the loop starts in the fenced form, `prompt_word`; agent/n11/prompts.py). `lean` sends design A's system prompt,
  its paths the path map's, the runtime's versions (`installed`) and design A's cards of the functions the first
  message cards (`prompts.lean_card`), no index, their form lane lean2's `LEAN_CARDS` (`AGENTHON_N11_LEAN_CARDS`
  or agent/n11/LEAN_CARDS: `cut`, today's, `whole`, each card as today's full prompt gives it, or a number of bytes;
  `prompts.lean_card_as`); `leanconst` (lane
  leanconst1) sends `lean`'s with design C's parts in its prose and library paragraph (`prompts.LEANCONST`,
  `prompts.LEANCONST_LIBRARY`) and its whole cards' conventions as stubs (`prompts.constants_cards`);
  `taskfirst` keeps today's head, reply form, cut, must list and example in the system prompt and moves How to work
  and the library after the task, the first message in design B's order (`task_part`, `prompts.TASKFIRST_ORDER`)
  and closing on `prompts.TASKFIRST_CLOSE`; `constants` is today's text with design C's edits (`worded`); `norules`
  leaves today's runtime paragraph and defaults list out of the system prompt, and `rulesafter` says them in the
  first message after the task's sections (`prompts.rules_after`): the rules probe's arms. `minimal` (lane
  prompt2) sends `prompts.minimal`, the one-shot's head and must list around today's fenced reply form, the cut and
  four working sentences, and no library: the pointer's functions dropped at the opening (no pointer line, no card,
  no gate), and no selection call whatever `SELECT` says (`selected`). `clean` (lane clean1; the static audit,
  runs/lanes/codex-audit-full-prompt-1019.md) is today's text with the audit's edits, every guidance sentence kept
  (`prompts.clean_parts`, `clean_text`): the system prompt reordered, its reply form said so in `told` too
  (`prompts.clean_tools`); the first message without its clock, its closing conventions' sentence, its Paths tail
  and a repeated input portrait (`first_turn`, `cleaned`); the selection call's index without the lookup
  (`selected`). The others read
  today's rendered prompt, every other switch's replacement in it (`control`), and runs today's where it is not in
  its paragraphs. No text states a count; under `full` and unset every body is today's. Keys `PROMPT_KEYS`, the
  first loop request's bytes as sent among them (`prompt_bytes_first`, lane clean1's).
- **No library lookup** (lane libnever1; the owner's in751): `LIB` (`AGENTHON_N11_LIB` or agent/n11/LIB: `open`,
  today's, or `never`, armed in forms `loop`, `review` and `sketch`, `lib_never`). Under `never` the `lib` tool is
  neither offered (`offered`) nor said (`said_tools`: its line, the tools form's count; `LIBRARY_USE_NEVER`; the index's
  and the selection call's section's heads, `library_unsaid`; a cut card's last line, `cards_unsaid`; the pointer's
  line, `POINTER_LOOKUP`; the short first move's, `LOOKS_SAID`; a malformed call's list, `malformed`; the sketch's
  line, `SKETCH_LEFT_NEVER`), and the picked or pointed cards go in the first message whole as under `CARDS=whole`
  whatever `CARDS` says (`whole_armed`); a lookup still asked is answered in one line (`LIB_ASKED`); the waste state
  puts away `inspect` alone, or nothing (`put_away`), its texts naming what it puts away (`LIB_NEVER_WHY`,
  `LIB_NEVER_HELD`). The selection call's own request is today's. Keys `LIB_KEYS`.
- **The fenced form as a switch** (lane fenced1; the owner's in727, in731 and in752): `MODE` (`AGENTHON_N11_MODE` or
  agent/n11/MODE: `tools`, today's, or `fenced`, armed in forms `loop`, `review` and `sketch`; `mode_of`). Under
  `fenced` the loop's forms run in the fenced form from the loop's first request, as `N11(mode="fenced")` does: no
  tools sent, the system prompt's fenced paragraph, every observation in the fenced form's heads; so a reply the output
  cap cuts arrives as text on every serving, and the cut paths read it (`cut_shape` fenced: its head held under
  `CUT=continue`, its definitions drafted under `split`; prose alone, `empty`, asks a named block under the explicit
  switch (`CUT_FENCED`), while the refusal fallback retains the tool form's empty-cut words).
  The selection call's own request is today's; form `state` is fenced whatever it says. A route that refuses the tools
  under `tools` turns fenced as before (`mode_refused`). Key `mode_source` beside `mode` (`MODE_KEYS`).
- **A fix by definition as a switch** (lane fixdef1; the owner's in727 and in767, docs/research/large-1013.md lever
  e2): `WRITE` (`AGENTHON_N11_WRITE` or agent/n11/WRITE: `whole`, today's, or `defs`, armed in forms `loop`, `review`
  and `sketch`). Under `defs` the House's write of a `.py` file that exists, given a text holding only top-level
  definitions one of whose functions or classes the file has, replaces those by name and keeps the rest
  (commands.splice), a splice that does not compile refused; the write tool's words, the tools paragraph and the
  fenced paragraph say so (commands.TOOLS_DEFS, `PART_DEFS`); the repeat refusal hashes the file the splice leaves
  (`written`), and a join is the file whole. Keys `WRITE_KEYS`.
- **A library function written by hand, said** (lane called1; the owner's in767 and in749): `CALLED`
  (`AGENTHON_N11_CALLED` or agent/n11/CALLED: `off`, today's, or `on`, armed in forms `loop`, `review` and `sketch`,
  `called_armed`). Under `on` the functions whose card reached the House are gathered (`card_reached`: a sent
  request's card, by its typed signature's line, or `lib`'s answer; never the conventions shelf's), and after each run,
  whatever its exit, and at the finish a def under work/ computing one by hand (lane lib4's rule) is one finding line
  naming the def, its file and the function (agent/n11/commands.py `Workspace.called`, `called_said`), said beside the
  findings and joining the finish's (`called_finish`), so the one finish refusal fires on it. Keys `CALLED_KEYS`.
- **The output rules and one refusal at the finish** (lane rules1; the owner's in767, in765 and in749,
  docs/research/wrong-1013.md): `RULES` (`AGENTHON_N11_RULES` or agent/n11/RULES: `off`, today's, or `on`, armed in
  forms `loop`, `review` and `sketch`, `rules_armed`). Under `on` the workspace's sanity step says the eight rules'
  findings (agent/n11/commands.py `RULES`, `Workspace.ruled`) after a run's exit 0 and at the finish alike, so
  `finish`'s one refusal a unit fires on them; at the finish the unread parameters are a finding too, said as a broken
  rule (`commands.unread_finding`); and where the unit ends with no request left (the cap, the clock, the last call) on
  an output breaking a `formula`, `row_sum` or `fraction` property the last clean run's output held, that output is put
  back (`put_back`, from the copy `ran` keeps of each clean run's output within `KEPT_BYTES`). Where no rule fires,
  every body is today's. Keys `RULES_KEYS`. Under `wide` (lane rules2; the owner's in767 and in749,
  docs/research/wrong-1014.md) all of `on` and six rules more after the eight (agent/n11/commands.py `WIDE_RULES`,
  `rules_wide`), their findings said after each run and among the finish's one refusal as the eight's are; the unread
  line and the put-back as under `on`.
- **An unchanged re-send read as the finish** (lane resend1; the owner's in767, docs/research/finish-1014.md): `RESEND`
  (`AGENTHON_N11_RESEND` or agent/n11/RESEND: `refuse`, today's, or `finish`, armed in forms `loop`, `review` and
  `sketch`, `resend_armed`). Under `finish` a reply's last write (`implicit`'s) refused as unchanged from turn N, where
  turn N's run exited clean (exit 0, no traceback: `resend_ran`) with no check failing, and every output file the task
  names is present, is read as the finish: `finish` runs, its observation the write's under one line saying so
  (`RESEND_READ`), and the unit ends, or the finish is refused as a written one is (its findings, or the checks over
  the output as it stands); the next such re-send is the finish again. Where those do not hold, the refusal's run line
  says the file is unchanged since turn N (`RESEND_SINCE`), and after a clean exit with named outputs missing names
  them (`RESEND_MISSING`); `HELD`, `CYCLE` and a refused join's line are today's. Under `refuse` every body is today's.
  Keys `RESEND_KEYS`.
- **A finish after a write runs the file first** (lane finishrun1; E162 ROW 402, the owner's in767): `FINISHRUN`
  (`AGENTHON_N11_FINISHRUN` or agent/n11/FINISHRUN: `off`, today's, or `on`, armed in forms `loop`, `review` and
  `sketch`, `finishrun_armed`). Under `on`, where a reply's `finish` follows a `.py` write with no `run` between
  (`finish_writes`), the harness runs the file before the finish exactly as `after_write` runs a reply's last write
  (`finishing`): its conditions, its observation's run text, `implicit_runs`; the write's observation carries the run,
  the finish's follows it and reads the run's output. Where `after_write` would run nothing (the write refused or not
  compiling, work/main.py missing or unchanged, too little time), the write's observation is today's and the finish
  judges the output as it stands. Under `off` every body is today's. Keys `FINISHRUN_KEYS`.
- **The ask after a failed continuation changes** (lane joinask1; E168 ROW 421, SPINE 414, the owner's in13):
  `JOINASK` (`AGENTHON_N11_JOINASK` or agent/n11/JOINASK: `same`, today's, or `change`, armed in forms `loop`, `review`
  and `sketch` under `CUT=continue`, `joinask_armed`; agent/n11/joinask.py keeps each file's junction). Under `change` a
  failure on a junction (a rest that does not compile with the head, a rest cut again, the same join refused or a head
  held before in the unit) is answered by an ask that is never the one before it (`join_failed`): the statement shape
  (`JOIN_STATEMENT`: the head's last statement named, the statement the cut fell in named by its first line, the rest
  asked from there; `join_statement`) or the module shape (`JOIN_MODULE`: what was kept through its last complete
  top-level statement saved as `work/<stem>_kept.py` and the file asked anew importing it; `join_module`). The third
  failure on one junction asks no rest: what stands runs, the last clean script or what was kept (`head_run`), and the
  fix is asked as definitions alone (`JOIN_RAN`; the next write spliced, `commands.splice`'s `adding`, or statements put
  after what was kept, `appended`). A whole rewrite of a file cut before, cut after a failing run, holds nothing and
  asks the fix as definitions (`JOIN_DEFS`, `defs_ask`). A file asked for definitions takes them by name until it is
  cut again or written whole (`defs_files`), and the first fragment that is neither its definitions nor it whole is
  refused, the file kept (`JOIN_KEPT`, once per definitions ask). Unchanged definitions get today's refusal, then
  end the attempt on their next re-send (`JOIN_STOP`), keeping its file and output. A head broken above its end is
  kept to what compiles, never dropped, and the streak's split does not fire on a held head. No ask states a count; under `same` every body is
  today's. Keys `JOINASK_KEYS`.
- **A finish on a failing run not taken** (lane finishfail1; the reader pairs, docs/research/pairs-1015.md section 6
  rank 2; the owner's in767 and in15): `FINISHFAIL` (`AGENTHON_N11_FINISHFAIL` or agent/n11/FINISHFAIL: `off`, today's,
  or `on`, armed in forms `loop`, `review` and `sketch`, `finishfail_armed`). Under `on` a `finish` whose run did not
  exit 0 (`shown_run`, `saw`: the last run the House was shown, `FINISHRUN`'s before it included; one stopped at its
  limit too) is answered with that run's observation under `FINISHFAIL_SAID`, its old budget line dropped, and is not
  counted among the finishes while a request and `ASK_S` remain (`turns_away`), only where the checks fail over the
  current output; at most `FINISHFAIL_MAX` times a unit, the next counted as today's (`finishfail_ended` where it ends the unit), and with no
  request or time left one ends the unit as today. `RESEND=finish`'s read finish is read before it and never turned
  away. No text states a count; under `off` every body is today's. Keys `FINISHFAIL_KEYS`.
- **The finish contract** (lane finishcheck1; the cold reader's form 5, runs/lanes/cold-velocity-proposals.toml;
  agent/n11/finishcheck.py): `FINISHCHECK` (`AGENTHON_N11_FINISHCHECK` or agent/n11/FINISHCHECK: `off`, today's, or
  `on`, armed in forms `loop`, `review` and `sketch`, `finishcheck_armed`). Under `on` a finish today's rules (and
  `FINISHFAIL`'s) would end the unit on, `counted`'s clean end or its second finish on failing checks, is checked
  with no request (`finishcheck_turn`, `finishcheck_findings`): the instruction read once for the names its own
  sentences tie to an output, against the output as it stands; where a finding of `finishcheck.ARMED` stands, a
  request and `ASK_S` remain and none was turned before (`FINISHCHECK_MAX`), the finish is answered with the findings
  under `FINISHCHECK_SAID` and not counted, the next counted as today's. A reader's fault is logged and the finish
  stands. No text states a count; under `off` every body is today's. Keys `FINISHCHECK_KEYS`.
- **The first output first** (lane firstout1; the reader pairs, docs/research/pairs-1015.md section 6 rank 5; ASK-12
  on the owner's in723 and in725): `FIRSTOUT` (`AGENTHON_N11_FIRSTOUT` or agent/n11/FIRSTOUT: `off`, today's, or `on`,
  armed in forms `loop`, `review` and `sketch`, `firstout_armed`). Under `on` the words the model reads ask the first
  script, work/main.py, to read the inputs and write the task's first output file alone, and each later output file
  in a reply of its own by a script with its own `__main__` block that imports what it needs (`FIRSTOUT_LATER`, which
  `after_write` runs as it runs any such file): the system prompt's second and third steps, its paragraph on the
  output limit, its rule on what every script writes and its example in either reply form (`FIRSTOUT_SAID`), the
  first message's output files (`FIRSTOUT_FILES`, the first file the task names), the redirect's demand
  (`REDIRECT_ASK_FIRSTOUT`), form `review`'s cut notes (`CUT_NOTE_FIRSTOUT`, `CUT_EMPTY_FIRSTOUT`) and the last clause
  of a continuation's ask for work/main.py's rest (`through`, `CUT_THROUGH_FIRSTOUT`: through its output, whatever
  it writes). The first message explains that missing output files keep the checks failing until written. A cut's
  split asks for modules of that output's script (`split_words`, `FIRSTOUT_SPLIT`), including a later script;
  repeated cuts ask for a complete module without stating a count (`CUT_SHORT_FIRSTOUT`). One shape for every
  task: no size read, no count stated, and a first script writing every output is
  taken as today's. The loop's rules do not change; under `off` every body is today's. Keys `FIRSTOUT_KEYS`.
- **A clause on task fidelity** (lane clauses1; the Codex brainstorm runs/lanes/codex-velocity-levers.md section 1,
  the reader pairs docs/research/pairs-1015.md section 2; the owner's in25 and in26): `CLAUSE` (`AGENTHON_N11_CLAUSE`
  or agent/n11/CLAUSE: `off`, today's, `fit`, `card` or `reuse`, armed in forms `loop`, `review` and `sketch`,
  `clause_armed`). Under a word the loop's first message closes on that word's sentence (`CLAUSE_SAID`) as a paragraph
  of its own, after the conventions' sentence, the pointer's line and any pace or sketch ask: `fit` asks
  the task's own steps where a library function's model differs from them, `card` asks the returned quantity mapped
  to the one asked with the card's lag, sign and unit rules applied, `reuse` asks a matching function called rather
  than its calculation rewritten. One sentence for every task, no count; the system prompt and every other text are
  today's, and under `off` every body is today's. Keys `CLAUSE_KEYS`.
- **The card after a failing library call** (lane cardfail1; the Codex reads runs/lanes/codex-levers-fresh.md rank 4
  and codex-errors-time.md, docs/research/levers-1015.md; the owner's in27): `CARDFAIL` (`AGENTHON_N11_CARDFAIL` or
  agent/n11/CARDFAIL: `off`, today's, `on` or `again`, armed in forms `loop`, `review` and `sketch`, `cardfail_armed`).
  Under `on` a run the House is shown (`execute`'s, `after_write`'s, a head run's, request 1's) whose last traceback ends in
  a library function's frame, or whose error names one (commands.failed_in: the guard's `qf.<name>`, a TypeError
  refusing the call's arguments, a bare call requiring an explicit, unshadowed `qf` import in its work script),
  carries that function's card, the text `lib name=` gives within its bound
  (library.answer), under `CARDFAIL_SAID`, after the run's own words and before its budget line (`carded_after`). One
  card a run, the most recent frame's; never a function carded already in the unit, by an earlier failure, a `lib`
  answer or the first message's cards (`cardfail_carded`, `cardfail_looked`, `cards`). Under `again` the first
  message's cards and a `lib` answer do not count, an earlier failure's card still does: once a function a unit. No
  text states a count; under `off` every body is today's. Keys `CARDFAIL_KEYS`.
- **A repeated failing program state answered** (lane repeat1, agent/n11/repeat.py; the Codex read
  runs/lanes/codex-errors-time.md rank 2, docs/research/levers-1015.md; the owner's in27): `REPEAT`
  (`AGENTHON_N11_REPEAT` or agent/n11/REPEAT: `off`, today's, or `change`, armed in forms `loop`, `review` and
  `sketch`, `repeat_armed`). Under either setting the unit's program states are kept (`repeat.Ledger`, `repeat_met`):
  after each run the House is shown, the `.py` files under work/ by content and the failure's signature (the exit and
  the exception's line, or after an exit 0 the first failing check's, folded: `repeat.signature`); a write refused as
  unchanged re-sends the program as it stands with its last run's failure; a state met before is a recurrence, counted
  in the unit over every state; a clean run clears the re-send's failure, but retains that history. Under `change` a
  turn ending on the unit's second recurrence or a later one closes its last observation on `REPEAT_ASK` before the
  next request (`repeat_turn`, `repeat_ask`; never after a cut nor with the
  last call's ask). At the third, once a unit, lane fallback1's one-shot is sent mid-loop where `FALLBACK=oneshot`
  arms it (`repeat_handoff`: its run a loop run's at most, the loop's output set aside and put back, the one-shot's
  kept beside work/ where it passes the checks); where the unit then ends on an output holding nothing to grade or
  failing the checks, the one-shot's is put back (`repeat_end`, the watchdog's close too). The rule never ends the
  unit, and `RESEND=finish`'s read comes first. No text states a count; under `off` every body is today's. Keys
  `REPEAT_KEYS`.
- **A worked example in the first message** (lane example1, agent/n11/example.py; the cold reader's form 4, the
  owner's in25 and in767): `EXAMPLE` (`AGENTHON_N11_EXAMPLE` or agent/n11/EXAMPLE: `off`, today's, or `nearest`, armed
  in forms `loop`, `review` and `sketch`, `example_armed`). Under `nearest` the loop's first message carries, after the
  output files and the conventions card and before the clock (`example_part`), the corpus's entry
  (agent/n11/examples.jsonl, `EXAMPLES`) nearest this task by its instruction's words (`example.nearest`): never this
  task, its twin or its source by the card's id, never one at `example.SAME` or over or with the same nonempty named
  output files (this task by its content), never one below `example.MIN_OVERLAP`, never
  one `AGENTHON_N11_EXAMPLE_EXCLUDE` names (`EXAMPLE_EXCLUDE`, the desk's alone); its instruction, its input files'
  headers and its passing script within a budget of its own (`example.shown`), no count stated. Under `off` every body
  is today's. Keys `EXAMPLE_KEYS`.
- **The state at a failure in the run's answer** (lane snapshot1, agent/n11/snapshot.py; the mid-loop trace read
  runs/lanes/codex-traces-midloop.md, "Failure snapshot"; the owner's in49): `SNAPSHOT` (`AGENTHON_N11_SNAPSHOT` or
  agent/n11/SNAPSHOT: `off`, today's, or `on`, armed in forms `loop`, `review` and `sketch`, `snapshot_armed`). Under
  `on` each run's script starts with agent/n11/preload.py's hook (commands.Workspace's `snapshotted`), which at an
  uncaught exception writes the state at the failure beside the run, never to its streams: the error; the innermost
  frame under work/ with the source around its line; the locals its failing statement reads and the frame's others,
  each by its type, shape or length, dtype, index, columns or keys, missing and non-finite cells and a few values; a
  library call's actual arguments. A run that raised says it after the traceback's tail and before the checks, whole
  even where the error is said in one line as an earlier run's, within a budget in the model's tokens never said; the
  runs the House is shown count it said or dropped (`snapshot_seen`). No text states a count; under `off` every body
  is today's. Keys `SNAPSHOT_KEYS`.
- **The continuation's join repaired** (lane join2 of 10-16, agent/n11/rejoin.py; the cut trace read
  runs/lanes/codex-traces-cut.md, "Join or write-state loss"; the owner's in49): `JOIN` (`AGENTHON_N11_JOIN` or
  agent/n11/JOIN: `today`, today's, or `repair`, armed in forms `loop`, `review` and `sketch`, `join_armed`). Under
  `repair` a rest whose join today fails, where the head's cut fell inside a statement and the rest goes on from the
  cut line, is stitched under that statement's received lines (`held_tail`, `rejoined`; `rejoin.repair`, taken only
  where it compiles); a rest that starts again like the program replaces the head; and a write over a saved file that
  compiles, shorter and unable to run in its place (`rejoin.staged`), is saved beside it as `<name>.staged`, answered
  `JOIN_STAGED` and work/main.py run where it changed (`stage`, `after_write`). No other text the model reads changes;
  under `today` every body is today's. Keys `REJOIN_KEYS`.
- **A bounded thinking budget on the write replies** (lane thinkwrite1, agent/n11/thinking.py; the wrong clean answer
  finished early, docs/research/errors.md; the owner's in53 and in55): `WRITE_THINKING` (`AGENTHON_N11_WRITE_THINKING`
  or agent/n11/WRITE_THINKING: `off`, today's, `first` or `all`, armed in forms `loop`, `review` and `sketch`) and its
  budget (`AGENTHON_N11_WRITE_THINKING_BUDGET` or agent/n11/WRITE_THINKING_BUDGET: a whole number of reasoning tokens
  within `thinking.BOUNDS`, `thinking.BUDGET` unset, or `low`, the House's low effort). Each loop request is read by
  the harness's state at its send (`request_kind`): a held head's rest, the answer to an unchanged re-send, the answer
  to a finish not taken, else the short clock's `state_kind`. Under `first` the requests before the unit's first reply
  that wrote or was cut (`first_write`; form `review`'s request 1 among them) ask the budget (`write_think`: the
  client's `thinking=`, model.thinking_template); under `all` every write and fix request does; a continuation, the
  finish, the request after a re-send, the selection call, the library pass and a request whose own switch asks
  (the sketch's turn, the review's request) never do. Each reply that asked is counted (`write_thought`), and one the
  cap cut with most of it reasoning stands the switch down for the unit. No text the model reads changes; under `off`
  every body is today's. Keys `WRITE_THINKING_KEYS`.
- **The dated derivation and the harness's own checks** (lane derive1, agent/n11/derive.py and agent/n11/witness.py;
  the reasoning read runs/lanes/codex-redesign-reasoning.md item 2, the wrong-answer trace read
  runs/lanes/codex-traces-wrong.md): `DERIVE` (`AGENTHON_N11_DERIVE` or agent/n11/DERIVE: `off`, today's, or `on`,
  armed in forms `loop`, `review` and `sketch`, `derive_armed`). Under `on` the loop's first message closes on
  `derive.ASK` (after any clause): the first reply opens with a short derivation naming each quantity's information
  date, window endpoints, payment or activation date and state, and in the same reply the script implements those
  steps as small functions keeping labelled indexes, every reported figure computed from the objects they build, and
  ends on its own checks as assertions that run; no request is added. After each run that exits 0 the checks of
  `witness.KEPT`, read from the instruction and the input files alone, are said in the run's observation
  (commands.Workspace's `witnessed`), advisory. Under either setting the loop's first reply is read for a derivation
  and the unit's first `.py` write for its assertions and functions (`derive_read`); the runs shown with findings
  are counted (`derive_seen`). No text states a count; under `off` every body is today's. Keys `DERIVE_KEYS`.
- **The output contract** (lane contract1, agent/n11/contract.py and agent/n11/props.py `terms`; the conventions
  synthesis docs/research/conventions-1017.md change A, the checks reads runs/lanes/codex-checks-{a,b,c}.md change 1):
  `CONTRACT` (`AGENTHON_N11_CONTRACT` or agent/n11/CONTRACT: `off`, today's, `on` or `refuse`, armed in forms `loop`,
  `review` and `sketch`, `contract_armed`). The instruction is read once a unit for the contract it states (each named
  output file's keys or columns and whether they are exact or ordered, roundings, the word for a missing cell, native
  types, value labels, nested paths, row order), under any setting, its terms counted. Under `on` and `refuse` the
  loop's first message says `stated_contract.ASK` after its output files, and after each run that exits 0 the findings
  of `stated_contract.KEPT` are said in the run's observation (commands.Workspace's `contracted`), advisory; under
  `refuse` a finish the checks pass while a finding stands is turned away once a unit with the findings quoted
  (`contract_turn`), and the next is today's. No text states a count; under `off` every body is today's. Keys
  `CONTRACT_KEYS`.
- **The conventions stated once** (lane conventions1; the conventions synthesis docs/research/conventions-1017.md
  items 1, 4 and 5 and change B, the checks reads runs/lanes/codex-checks-{a,b,c}.md): `STATED` (`AGENTHON_N11_STATED`
  or agent/n11/STATED: `off`, today's, or `on`, armed in forms `loop`, `review` and `sketch`, `stated_armed`). Under
  `on` the system prompt alone changes, in two places: its conventions say row eligibility and clocks operation by
  operation (`ELIGIBILITY_LINES`: eligibility per calculation, duplicates resolved as stated, grids and windows by the
  clock, warm-up and the first return, thresholds strict or inclusive as worded) in the place of today's blanket
  missing-row lines (`MISSING_LINES`), the others kept (`STATED_LINES`); and How to work's first step asks each
  convention the task pins stated once at the script's top as a named constant or function and used by that name,
  a silent one's choice said once in a comment (`SETTLE_STATED` for `SETTLE`). No default is stated: there is none
  (the synthesis's item 1). No text states a count; under `off` every body is today's. Keys `STATED_KEYS`.
- **The reconciliation after a clean run** (lane recon1, agent/n11/recon.py; the failure reads'
  cause 1, docs/research/failures-1017.md, and the relitigation's lesson, docs/research/relitigation-1017.md): `RECON`
  (`AGENTHON_N11_RECON` or agent/n11/RECON: `off`, today's, `on` or `refuse`, armed in forms `loop`, `review` and
  `sketch`, `recon_armed`). Under `on` and `refuse` each clean run (exit 0, the checks passing) is read against the
  identities the harness computes from the output, the input files and the task's words (`recon.KEPT`: the accounting
  replay, the repricing identity, the date and window comparison) and lane derive1's checks (`witness.KEPT`, one
  finding an object), and while one stands the run's verdict line says the numbers do not reconcile yet, the findings
  after it as the observation's last block (commands.Workspace's `reconciled`), the short clock's finish ask withheld
  (`last_clean`), and a finding standing unchanged after the scripts changed said in the same-frame form
  (`recon_same`); under `refuse` a finish while one stands is turned away once a unit with the findings quoted
  (`recon_turn`), sharing one refusal with the contract's and finishcheck's (`guards_turn`), and the next is today's.
  A finding said at a run and absent at the next run read is counted acted (`recon_seen`). No text states a count;
  under `off` every body is today's. Keys `RECON_KEYS`.
- **The audit on the first clean run** (lane audit1, agent/n11/audit.py; the cold reader's design D,
  runs/lanes/cold-prompt1017.md; the failure reads' cause 1, docs/research/failures-1017.md): `AUDIT`
  (`AGENTHON_N11_AUDIT` or agent/n11/AUDIT: `off`, today's, or `on`, armed in forms `loop`, `review` and `sketch`,
  `audit_armed`). Under `on` the system prompt's fifth step is said the audit's way (`audit.STEP` for `audit.TODAY`, the
  template's text unchanged), and the unit's first clean run in the loop (exit 0, no error, the checks passing: every
  named output present) is answered with the audit after its observation (`audited`, `audit.section`: the outputs and
  the method, then the items the task's words switch on), the budget line kept last; where under `audit.FLOOR_S` of
  the loop's clock (`left`) or no request remain it is skipped, logged and traced, never said. A finish in the reply
  whose run carried it is held once (`audit_turn`, `audit.HELD`), sharing one refusal with the contract's, the
  reconciliation's and finishcheck's (`guards_turn`); the next is today's. The reply after it is read for its answers
  (`audit_read`, `audit.answers`). No text states a count; under `off` every body is today's. Keys `AUDIT_KEYS`.
- **The first drafts** (lane drafts1, agent/n11/drafts.py; the refuter's accepted form, ACCEPT ROW 600; the one-shot
  floor, the coordinator's scope addition after REJECT ROW 603): `DRAFTS` (`AGENTHON_N11_DRAFTS` or agent/n11/DRAFTS:
  `off`, today's, `one` or `three`, the first drafts sent, armed in forms `loop` and `sketch`, `drafts_armed`). Armed,
  before the loop's first request (after the selection call), N0's request is drawn on a client of its own a draft (the
  unit's one count, its own pacing floor), within the round's request allowance and time: under `one` once, as N0
  sends it (its system prompt, temperature 0, the seed); under `three` at once under agent/w1's first preambles, their
  counts reworded to none (`drafts.SYSTEMS`), at w1's temperature, the seed pinned. Each script runs in turn under a
  cap, its files stored apart; the clean drafts are compared (`drafts.compare`: the same files, keys and shapes, every
  number within rtol 1e-4). The floor (the agreed files, else the first clean draft's) is held aside and written at the
  unit's close only where the output holds nothing to grade (`drafts_close`), so the loop's bodies are today's but for
  the requests and seconds the round spent, and but for, where two or more clean drafts match none, the first
  message's closing block of the scripts and the field table with the ask to settle each field by the task's sentence
  (`drafts_first`). A failure of the round loses a draft, never the unit. No text states a count; under `off` every
  body is today's. Keys `DRAFTS_KEYS`.
- **The repair pair** (lane repairpair1, agent/n11/repair.py; the cold reader's second concurrency form, COLD ROW 669):
  `REPAIR` (`AGENTHON_N11_REPAIR` or agent/n11/REPAIR: `off`, today's, or `pair`, armed in forms `loop` and `sketch`).
  Under `pair` the loop's request after a turn whose run failed its own observation, or whose finish was not taken on
  a failing output (`repair_due`), goes beside B, a rewrite on a sibling client (`repair_start`: A's system prompt and
  tools, the write tool required, the task as the first message frames it, `framing`, the failed run's scripts and
  observation, the ask for work/main.py whole and plain enough to fit one reply; a rewrite the output cap cuts all
  the same continued with B's second request as request 1's is, `rest_asked`); B's script runs on its thread in a
  directory of its own, its output root rewritten there, and the loop waits on B at most a request's median after
  A's turn. Before the next request, or at the stop, the keep rule (`repair.keep`: clean, then more named
  files, then fewer findings, then A) settles the pair (`repair_settle`): A kept changes nothing; B kept puts back the
  state A's turn changed (`TURN_STATE`, the files under work/), moves B's files to the output, saves its script as
  work/main.py and carries its reply in the conversation as one write answered by its run (`repair_kept`). B's
  requests come from the House's room above the loop's cap (`requests_left` today's). No text states a count or says
  the pair; under `off` every body is today's. Keys `REPAIR_KEYS`.
Rules 8 and 9 as N0 keeps them: nothing names a unit (but the worked example's corpus, whose entries are other public
units' and never shown to their own unit); `write` reaches `work/` alone, so every deliverable comes from a
run of the House's scripts; the canaries are stripped from the instruction, marked `<canary>` in every observation and
scanned at the close. The watchdog and the close are N0's (common.guard), and the exit is 0 whatever happens. One
`summary:` line ends the log with the counts the registered read takes (scripts/run_sweep_read.py `N11_KEYS`).
"""
from __future__ import annotations

import argparse
import ast
import codeop
import copy
import difflib
import functools
import hashlib
import importlib.metadata
import io
import json
import os
import platform
import re
import shutil
import statistics
import sys
import tempfile
import time
import tokenize
import traceback
import warnings
from collections import Counter
from pathlib import Path

from common import canary, contract, digest, guard, house, paths, runner, trace, unit
from common.house import model
from common.qfuse import qf_use
from n0 import solve as n0
from n11 import audit, commands, conventions, derive, drafts, example, fallback, finishcheck, inputs, joinask, \
    library, measure, props, recon, rejoin, repair, repeat, sanity, selection, snapshot, state, thinking, witness
from n11 import contract as stated_contract
from n11.commands import Call, fit
from n11 import prompts

AGENT = "n11"
MODES = ("tools", "fenced")     # the reply forms: native tool calls, or the fenced fallback (3.2), the first the
#                                 default (`MODE`). The probe on the free route read tools: 16 of 17 replies
#                                 well-formed, a median of 247 tokens (lane n11's report)
MODE_ENV = "AGENTHON_N11_MODE"  # lane fenced1: the loop's forms in the fenced form from their first request (`fenced`,
MODE_FILE = "MODE"              # so a cut write's text arrives on every serving) or today's tools form; the desk's
#                                 switch, or agent/n11/MODE, one word, an image's
MODE_FORMS = ("loop", "review", "sketch")   # armed in the loop's forms (form `state` is fenced whatever it says)
MODE_KEYS = ("mode_source",)    # where the reply form the unit ended in came from, beside today's `mode`
MODE_REFUSED = "refused"        # `mode_source` where the route refused the tools and the loop turned fenced
PROMPT_ENV, PROMPT_FILE = prompts.ENV, prompts.FILE     # lane prompt1: the loop's prompt, `full` (today's),
PROMPT_SETTINGS = prompts.SETTINGS  # `lean`, `taskfirst`, `constants`, `norules`, `rulesafter`, `minimal` or
#                                     `leanconst` (agent/n11/prompts.py); the desk's switch, or agent/n11/PROMPT, one
#                                     word, an image's
PROMPT_FORMS = prompts.FORMS        # armed in form `loop` where the loop starts fenced (`prompts.MODE`)
PROMPT_KEYS = prompts.KEYS
LEAN_CARDS_ENV, LEAN_CARDS_FILE = prompts.LEAN_CARDS_ENV, prompts.LEAN_CARDS_FILE   # lane lean2: `lean`'s cards' form,
LEAN_CARDS_SETTINGS = prompts.lean_cards_words()    # `cut` (today's), `whole` or a number of bytes; armed where
#                                                     `lean` or `leanconst` is (`prompts.LEAN_WORDS`)
REQUESTS = 20               # counted at admission, of the House's 25 (3.5)
LOOP_BYTES = 7_500          # form `review`: request 1 is the loop's own at this many bytes of instruction or more
RUN_SHARE = 0.5             # N0's: request 1's script runs under half the time left
RUN_RESERVE_S = 20.0        # kept from the deadline after the last run: the close, before the watchdog's 10 s
MIN_RUN_S = 5.0             # a run is not started with less time than this
RUN_S = 90.0                # a loop run's limit
ASK_S = 45.0                # no request with less time than this left; and a run leaves it where a request may follow
KEEP_TURNS = 3              # tool results older than the last three turns are cut to their first line
GATE_HOLDS_CLEAN = False    # the gate acts on flagged or failed runs alone: 3.6's bar is unread while no qf docstring
#                             names an alias (scripts/n11_offline.py measures it once the library lanes land)
NO_CALL_CAP = 3             # replies in a row with no tool call before the unit ends as it stands (lane n11e)
LAST_CALL_S = 2 * ASK_S     # lane clock1: under this many seconds left the time buys one more reply and not two
CONTINUATIONS = 2           # requests asking the rest of request 1's cut script, at most (lane n11e)
FORMS = ("loop", "review", "state", "sketch")   # the agent's shape, chosen without editing the code (`form_of`); the
#                                                 first the default (lane prune1)
FORM_ENV = "AGENTHON_N11_FORM"          # the desk's: src/agenthon_t1/harness.py hands it to an --agent run's container
FORM_FILE = "FORM"                      # agent/n11/FORM, one word: written by whoever builds an image of a form
REVIEW_REQUESTS = 4                     # form `review`: the review's requests, at most
REVIEW_FLOOR = 2                        # and it opens only with this many requests and this many ASK_S left
REVIEW_BYTES = 8_000                    # the review observation's bound, its ask and the budget line kept whole
STATE_RESULT_ENV = "AGENTHON_N11_STATE_RESULT"      # form `state`'s LAST RUN setting (lane state2): the desk's switch,
STATE_RESULT_FILE = "STATE_RESULT"                  # or agent/n11/STATE_RESULT, one word, written by an image's builder
SKETCH_ENV = "AGENTHON_N11_SKETCH"      # form `sketch`'s switch (lane plan4): the desk's, handed to an --agent run's
SKETCH_FILE = "SKETCH"                  # container, or agent/n11/SKETCH, one word, written by an image's builder
SKETCHES = ("turn", "inline")           # the sketch a reply of its own before any code, or the first reply's text
#                                         before its call
SKETCH_THINKING_ENV = "AGENTHON_N11_SKETCH_THINKING"    # addendum 4 (the owner's in344): the sketch's request asks
SKETCH_THINKING_FILE = "SKETCH_THINKING"                # low-effort thinking where this says `low`; the desk's switch,
SKETCH_THINKINGS = ("off", "low")                       # or agent/n11/SKETCH_THINKING, an image's
SKETCH_FLOOR = 2                        # `turn` asks it only with this many requests and this many ASK_S left
SKETCH_COUNTS = ("sketch_asked", "sketch_tokens", "sketch_cut", "sketch_names_lib", "sketch_skipped",
                 "sketch_inline_bytes", "sketch_picks", "sketch_picks_unknown", "sketch_cards_added",
                 "sketch_reasoning_tokens")     # lane plan4's and its addenda's, on the summary line as counted
INPUTS_ENV = "AGENTHON_N11_INPUTS"      # the input files described (lane inputs1; the owner's in346): the desk's switch,
INPUTS_FILE = "INPUTS"                  # or agent/n11/INPUTS, one word, written by an image's builder
INPUTS_SETTINGS = ("standard", "rich")  # `standard`, the descriptions as they were; `rich`, agent/n11/inputs.py's
REVIEW_THINKING_ENV = "AGENTHON_N11_REVIEW_THINKING"    # the desk's switch: `low` asks form `review`'s review request
#                                                         thinking at low effort (lane inputs1); unset, off
REQUESTS_ENV = "AGENTHON_N11_REQUESTS"  # the loop's request cap (lane inputs1): the desk's switch, or agent/n11/REQUESTS,
REQUESTS_FILE = "REQUESTS"              # one number, written by an image's builder; unset, `REQUESTS`
REQUESTS_BOUNDS = (4, model.REQUESTS_PER_UNIT - 1)  # a cap outside them refused: the House admits 25, one slot kept for
#                                                     a charged retry
REPLY_KINDS = ("request1", "loop", "review", "libpass")     # the kinds the summary splits replies by
CLOCK_COUNTS = ("last_call_asked", "last_call_output", "last_call_clean", "run_uncapped",
                "prose_forced")     # lane clock1's, on the summary line as counted
FENCE_COUNTS = ("blocks_path_line", "blocks_unsaved", "blocks_twice", "finish_asked", "blocks_named",
                "blocks_named_outside")   # lane fence1's, on the summary line: the blocks named on the line after the
#                                           fence, those never saved, the files a later block replaced, the asks to
#                                           finish; lane fencedpath1's: the whole blocks a labelled line named another
#                                           file by (commands.labelled), accepted or refused by the workspace path check
INDEX_KEYS = ("index_bytes", "index_functions")     # lane index1's: the index's bytes and the functions it shows
#                                                     with their card whole (library.Library.whole), on the summary
#                                                     line and the usage line
INDEX_ENV = "AGENTHON_N11_INDEX"        # lane index2: the index's setting (library.INDEX_MODES), the desk's switch,
INDEX_SWITCH = "INDEX"                  # or agent/n11/INDEX, one word, written by an image's builder
SCORED_KEYS = ("index_mode", "index_scored_hits")   # lane index2's, on the summary line after lane index1's
MEASURE_KEY = "measure"     # lane tokens1: the unit the bounds counted in (agent/n11/measure.py), `tokens` or `bytes`,
#                             on the summary line and the usage line; why bytes, on the log's `settings` line
NOCALL_ENV = "AGENTHON_N11_NOCALL"      # lane cap1: the loop's answer to a reply with no call, `stop` (today's: the
NOCALL_FILE = "NOCALL"                  # third in a row ends the unit) or `ask` (`NOCALL_ASK` after every one); the
NOCALLS = ("stop", "ask")               # desk's switch, or agent/n11/NOCALL, one word, written by an image's builder
PICKS_ENV = "AGENTHON_N11_SKETCH_PICKS"     # lane cap1: the sketch's picks read from every word (`words`, today's)
PICKS_FILE = "SKETCH_PICKS"                 # or from `qf.` names and code spans alone (`named`); the desk's switch,
PICKINGS = ("words", "named")               # or agent/n11/SKETCH_PICKS, an image's
CAP1_SETTINGS = ("nocall_mode", "picks_mode")      # lane cap1's settings and counts, on the summary line
CAP1_COUNTS = ("nocall_asked", "nocall_recovered", "sketch_picks_named", "sketch_picks_dropped")
PACE_ENV = "AGENTHON_N11_PACE"          # lane pace1: the loop at the platform's pace, `off` (today's) or `adaptive` (the
PACE_FILE = "PACE"                      # measured pace, the short-clock mode, the clock rules by the pace); the desk's
PACE_SETTINGS = ("off", "adaptive")     # switch, or agent/n11/PACE, one word, written by an image's builder
PACE_PRIOR_S = 44.0         # the pace before any reply: the platform's published one (the organisers' record, 37 to 46 s
#                             a request at the median in two full runs of 10-07, 44 s over 240 requests on 10-02); it
#                             shapes the first message alone, the mode waiting on a measured pace
RUN_PRIOR_S = 1.0           # a run's expected seconds before any run: E076's loop runs, 0.8 s at the median of the
#                             units' medians and 1.1 s at their 75th percentile
SHORT_CLOCK_REQUESTS = 10   # the short-clock mode under this many requests bought: E076's 15 solved units reached their
#                             first clean run within 7 requests, the sketch's turn among them, and spent up to 3 more (the
#                             finish, the verify turn's rewrite and its finish); on the 400 s clock the platform's slow
#                             regime (37 to 46 s a request) buys 8.1 to 10 at the start, under it from 38 s, and its
#                             15.6 s about 23
PACE_COUNTS = ("short_first", "short_clock", "short_clock_at", "short_finish_asked",
               "last_call_by_pace")     # lane pace1's, on the summary line after its setting, `pace_s` and
#                                         `requests_bought`
CONVCARD_ENV = "AGENTHON_N11_CONVENTIONS"     # lane convcard1: the conventions card, `off` (today's) or `on` (the
CONVCARD_FILE = "CONVENTIONS"               # instruction's convention words as questions in the loop's first message,
CONVCARD_SETTINGS = ("off", "on")           # agent/n11/conventions.py); the desk's switch, or agent/n11/CONVENTIONS,
CONVCARD_FORMS = ("loop", "review", "sketch")   # one word, written by an image's builder; armed in the loop's forms
CONVCARD_KEYS = ("conventions_mode", "conventions_terms", "conventions_tokens", "span_finding", "scale_finding")
FIRST_WRITE_KEYS = ("first_write_tokens", "first_write_cut")    # lane scaffold1's addendum: the unit's first reply
#                                                 that wrote a file, its tokens and its cut, on the summary line after
#                                                 the pace's
CUT_ENV = "AGENTHON_N11_CUT"        # lane review2: a .py write the output cap cut, `split` (today's: its definitions
CUT_FILE = "CUT"                    # kept as a draft and the split asked) or `continue` (its head kept, the rest asked
CUT_MODES = ("split", "continue")   # and stitched under it); the desk's switch, or agent/n11/CUT, one word, an image's
CUT_FORMS = ("loop", "review", "sketch")    # armed in the loop's forms (form `state` keeps its own cut path)
CUT_KEYS = ("cut_mode", "cut_continued", "cut_stitched", "cut_streak_max")
JOIN_KEYS = ("cut_overlap", "cut_join_failed", "cut_join_refused", "cut_join_reread", "directive_in_block")
#   lane join1: the joins that took the head's end a rest repeated once, and the joins (a restart aside) that did not
#   compile, saved or refused; lane join2: those refused as the file already saved, the joins taken under a reading
#   past `_readings` (`runner.REREAD`), and the fenced blocks read as the one directive they held
WRITE_ENV = "AGENTHON_N11_WRITE"    # lane fixdef1: a write of a .py file that exists, `whole` (today's: replaced whole)
WRITE_FILE = "WRITE"                # or `defs` (a text holding only top-level definitions replaces those by name and
WRITE_MODES = ("whole", "defs")     # keeps the rest, commands.splice); the desk's switch, or agent/n11/WRITE, one word,
#                                     an image's
WRITE_FORMS = ("loop", "review", "sketch")      # armed in the loop's forms (form `state` keeps its own write)
WRITE_KEYS = ("write_mode", *commands.SPLICE_KEYS)
PART = ".part"                      # the head's file beside the named one, work/<name>.py.part, never run
HEAD_LINES = 4                      # the head's last lines the ask quotes, at most: the harness's bound, never said
HEAD_BYTES = 600                    # and their bytes (a longer last line quoted by its end)
HELD_BYTES = 2_000                  # the ask's bound, its budget line kept whole
REVIEW_KEYS = ("review_skipped",)   # lane review2: the review due and not opened for the requests or the clock
CARDS_ENV = "AGENTHON_N11_CARDS"    # lane first1: the first message's cards, `cut` (today's: the pointer's within
CARDS_FILE = "CARDS"                # library.CARD_TOKENS, the call's within selection.CARD_TOKENS, each cut by rank) or
CARDS_MODES = ("cut", "whole")      # `whole` (each whole, the index trimmed for them: `made_whole`); the desk's
#                                     switch, or agent/n11/CARDS, one word, an image's
CARDS_FORMS = ("loop", "review", "sketch")      # armed in the loop's forms (form `state` keeps its CARDS section's)
CARDS_KEYS = ("cards_mode", "cards_whole", "index_trimmed_for_cards")   # beside today's `cards_cut`
INSPECT_ENV = "AGENTHON_N11_INSPECT"    # lane first1: an inspect of an input file the first message describes, `open`
INSPECT_FILE = "INSPECT"                # (today's) or `described` (answered from that description, `inspected`, and
INSPECT_MODES = ("open", "described", "never")  # the tool withheld while it could show nothing the first message
#                                         lacks, `offered`), or lane putaway1's `never` (withheld from every request,
#                                         a look still asked answered as under `described`); the desk's switch, or
#                                         agent/n11/INSPECT, one word, an image's
INSPECT_FORMS = ("loop", "review", "sketch")    # armed in the loop's forms (form `state` has no inspect)
INSPECT_KEYS = ("inspect_mode", "inspect_local", "inspect_sent", "inspect_withheld")
LASTCALL_ENV = "AGENTHON_N11_LASTCALL"  # lane clock1 of 10-13 (reader time, docs/research/time-1013.md): the last
LASTCALL_FILE = "LASTCALL"              # call due by its ask (`late`, today's) or by its send (`early`: `by_send`);
LASTCALL_MODES = ("late", "early")      # the desk's switch, or agent/n11/LASTCALL, one word, an image's
LASTCALL_REPLY_S = 40.0     # a whole reply's room after its send: 4,000 tokens at about 105 a second and its start
#                             (reader time: 38.7 s and 38.2 s at the median on the free variant and NVIDIA); never said
LASTCALL_RUN_S = 4.0        # and its run's after it (reader time: the runs take 1 to 4 s a unit-run); never said
LASTCALL_KEYS = ("lastcall_mode", "lastcall_by_send", "lastcall_asked_at_s", "lastcall_sent_at_s", "lastcall_left_s",
                 "lastcall_outcome")
SHIM_ENV = "AGENTHON_N11_SHIM"          # lane clock1 of 10-13: the run's environment pre-empting the mechanical
SHIM_FILE = "SHIM"                      # crashes (agent/n11/shim.py), `off` (today's) or `on`; the desk's switch, or
SHIM_MODES = ("off", "on")              # agent/n11/SHIM, one word, an image's
SHIM_KEYS = ("shim_mode", "shim_fired", "shim_runs")
LINT_ENV = "AGENTHON_N11_LINT"          # lane clock1 of 10-13: a static check at each saved write of a Python file
LINT_FILE = "LINT"                      # (agent/n11/lint.py), `off` (today's) or `on`, its findings advisory lines of
LINT_MODES = ("off", "on")              # the write's observation; the desk's switch, or agent/n11/LINT, an image's
LINT_KEYS = ("lint_mode", "lint_findings", "lint_writes")
CLOCK1_KEYS = LASTCALL_KEYS + SHIM_KEYS + LINT_KEYS     # the three's keys, before lane redirect1's
REDIRECT_ENV = "AGENTHON_N11_REDIRECT"  # lane redirect1: a loop wasting its requests on looks redirected to its script,
REDIRECT_FILE = "REDIRECT"              # `off` (today's) or `write` (the waste state declared, `redirected`; then, until
REDIRECT_MODES = ("off", "write")       # a file is saved, `inspect` and `lib` put away and the script asked, `offered`,
#                                         `redirect_ask`; and a reply at the cap a cut, `at_cap`); the desk's switch, or
#                                         agent/n11/REDIRECT, one word, an image's
REDIRECT_FORMS = ("loop", "review", "sketch")   # armed in the loop's forms (form `state` has no inspect)
REDIRECT_LOOKS = 2      # inspect-only replies accumulated before any qualifying reply declares the waste state
#                         one (docs/research/requests-1013.md, I2@4: 5 of 91 solved unit-runs touched); 3 is the
#                         variant under 5 percent (I3)
REDIRECT_AT = 4         # and the request it answers, at the least, the selection call's place counted whether the call
#                         was sent or not (where it is off, the loop's third reply)
CUT_READ_KEYS = ("cut_shape_calls", "cut_shape_length", "cut_shape_fenced", "cut_shape_empty",  # each reply at the
                 "cut_tail", "cut_resent", "cut_prose")     # cap by what arrived, each reply to a continuation ask by
#                                                             what it held: under either setting
REDIRECT_KEYS = ("redirect_mode", "redirect_at", "redirect_left_s", "redirect_after", "redirect_writes",
                 "redirect_cut", "redirect_looks", "redirect_nocall", "redirect_first_saved_s", *CUT_READ_KEYS)
SYSTEM_KEYS = ("system_tools", "system_changed")    # lane putaway1: the tools the last loop request's system prompt
#                                                     described, and the loop's requests whose prompt differed from
#                                                     its first's
LIB_ENV = "AGENTHON_N11_LIB"            # lane libnever1: the library lookup, `open` (today's) or `never` (the `lib`
LIB_FILE = "LIB"                        # tool neither offered nor said on any request, a lookup still asked answered in
LIB_MODES = ("open", "never")           # one line, and the cards in the first message whole as under `CARDS=whole`:
#                                         `lib_never`); the desk's switch, or agent/n11/LIB, one word, an image's
LIB_FORMS = ("loop", "review", "sketch")    # armed in the loop's forms (form `state` keeps its `lib` directive)
LIB_KEYS = ("lib_mode", "lib_asked", "lib_blind")   # the setting, the lookups answered in one line, and the functions
#                                                     the House's files call whose card the first message did not carry
CALLED_ENV = "AGENTHON_N11_CALLED"      # lane called1: a def computing a library function whose card reached the House
CALLED_FILE = "CALLED"                  # said as a finding after each run and at the finish, `off` (today's) or `on`
CALLED_MODES = ("off", "on")            # (`called_armed`; agent/n11/commands.py `Workspace.called`); the desk's
#                                         switch, or agent/n11/CALLED, one word, an image's
CALLED_FORMS = ("loop", "review", "sketch")     # armed in the loop's forms (form `state` runs its files its own way)
CALLED_KEYS = ("called_mode", "called_findings", "called_refused", "called_fixed", "called_kept")
RULES_ENV = "AGENTHON_N11_RULES"        # lane rules1: the output rules after each run and one refusal at the finish,
RULES_FILE = "RULES"                    # `off` (today's), `on` or `wide` (lane rules2: `on` and six rules more); the
RULES_SETTINGS = ("off", "on", "wide")  # desk's switch, or agent/n11/RULES, one word, an image's
RULES_FORMS = ("loop", "review", "sketch")      # armed in the loop's forms (form `state` keeps its own finish)
RULES_KEYS = ("rules_mode", *(f"rule_{name}" for name in (*commands.RULES, *commands.WIDE_RULES)), "rules_findings",
              "rules_refused", "unread_refused", "putback", "rules_fixed")     # lane rules2's six after the eight
PUTBACK_RULES = ("formula", "row_sum", "fraction")  # the properties an earlier clean output is put back for
RESEND_ENV = "AGENTHON_N11_RESEND"      # lane resend1: a reply's last write refused as unchanged since a clean run with
RESEND_FILE = "RESEND"                  # every named output present, `refuse` (today's) or `finish` (read as the
RESEND_SETTINGS = ("refuse", "finish")  # finish); the desk's switch, or agent/n11/RESEND, one word, an image's
RESEND_FORMS = ("loop", "review", "sketch")     # armed in the loop's forms (form `state` answers its own re-sends)
RESEND_KEYS = ("resend_mode", "resend_finished")
FINISHRUN_ENV = "AGENTHON_N11_FINISHRUN"    # lane finishrun1: a `finish` after a `.py` write in one reply, `off`
FINISHRUN_FILE = "FINISHRUN"                # (today's: the finish judges the output as it stands) or `on` (the file
FINISHRUN_SETTINGS = ("off", "on")          # run first, as `after_write` runs a reply's last write); the desk's
#                                             switch, or agent/n11/FINISHRUN, one word, an image's
FINISHRUN_FORMS = ("loop", "review", "sketch")  # armed in the loop's forms (form `state` runs a saved file itself)
FINISHRUN_KEYS = ("finishrun_mode", "finish_ran")
FINISHFAIL_ENV = "AGENTHON_N11_FINISHFAIL"  # lane finishfail1: a `finish` whose run did not exit 0, `off` (today's: it
FINISHFAIL_FILE = "FINISHFAIL"              # counts among the finishes) or `on` (answered with that run's failure and
FINISHFAIL_SETTINGS = ("off", "on")         # not counted while a request and `ASK_S` remain); the desk's switch, or
#                                             agent/n11/FINISHFAIL, one word, an image's
FINISHFAIL_FORMS = ("loop", "review", "sketch")     # armed in the loop's forms (form `state` keeps its own finish)
FINISHFAIL_KEYS = ("finishfail_mode", "finishfail_turned", "finishfail_ended")
FINISHFAIL_MAX = 3              # the finishes on a failing run turned away a unit; the next is counted as today's
FINISHCHECK_ENV, FINISHCHECK_FILE = finishcheck.ENV, finishcheck.FILE   # lane finishcheck1: a finish today's rules take
FINISHCHECK_SETTINGS = finishcheck.SETTINGS     # checked against the instruction's contract with no request,
#                                                 `off` (today's) or `on` (its findings answer it, once a unit, while
#                                                 a request and `ASK_S` remain); the desk's switch, or
#                                                 agent/n11/FINISHCHECK, one word, an image's
FINISHCHECK_FORMS = finishcheck.FORMS           # armed in the loop's forms (form `state` keeps its own finish)
FINISHCHECK_KEYS = finishcheck.KEYS
FINISHCHECK_MAX = finishcheck.MAX               # the finishes the contract turns away a unit; the next is today's
FIRSTOUT_ENV = "AGENTHON_N11_FIRSTOUT"      # lane firstout1: the words the model reads ask the whole script first
FIRSTOUT_FILE = "FIRSTOUT"                  # (`off`, today's) or the task's first output file alone and each later one
FIRSTOUT_SETTINGS = ("off", "on")           # in a reply of its own (`on`); the desk's switch, or agent/n11/FIRSTOUT,
#                                             one word, an image's
FIRSTOUT_FORMS = ("loop", "review", "sketch")   # armed in the loop's forms (form `state` keeps its own words)
FIRSTOUT_KEYS = ("firstout_mode", "first_outputs", "outputs_named")
CLAUSE_ENV = "AGENTHON_N11_CLAUSE"      # lane clauses1: a sentence on task fidelity closing the loop's first message,
CLAUSE_FILE = "CLAUSE"                  # `off` (today's: none) or `fit`, `card` or `reuse` (`CLAUSE_SAID`'s); the
CLAUSE_SETTINGS = ("off", "fit", "card", "reuse")   # desk's switch, or agent/n11/CLAUSE, one word, an image's
CLAUSE_FORMS = ("loop", "review", "sketch")     # armed in the loop's forms (form `state` keeps its own words)
CLAUSE_KEYS = ("clause_mode",)
CARDFAIL_ENV = "AGENTHON_N11_CARDFAIL"      # lane cardfail1: a run failing in a library function, `off` (today's: the
CARDFAIL_FILE = "CARDFAIL"                  # run's observation alone), `on` (that function's card after it, once a
CARDFAIL_SETTINGS = ("off", "on", "again")  # function a unit, never one the House has a card of) or `again` (the card
#                                             even where the first message or a `lib` answer carried it); the desk's
#                                             switch, or agent/n11/CARDFAIL, one word, an image's
CARDFAIL_FORMS = ("loop", "review", "sketch")   # armed in the loop's forms (form `state` keeps its own observations)
CARDFAIL_KEYS = ("cardfail_mode", "cardfail_shown")
JOINASK_ENV, JOINASK_FILE = joinask.ENV, joinask.FILE   # lane joinask1: the ask after a failed continuation, `same`
JOINASK_SETTINGS = joinask.SETTINGS     # (today's) or `change` (never the ask before it after a failure, at most
#                                         `joinask.ASKS` asks on one junction, the head never dropped for a rest's
#                                         fault); the desk's switch, or agent/n11/JOINASK, one word, an image's
JOINASK_FORMS = joinask.FORMS           # armed in the loop's forms, under `CUT=continue`
JOINASK_KEYS = joinask.KEYS
REPEAT_ENV, REPEAT_FILE = repeat.ENV, repeat.FILE     # lane repeat1: the same program failing the same way again, `off`
REPEAT_SETTINGS = repeat.SETTINGS       # (today's: the states kept and counted, nothing said) or `change` (the unit's
#                                         second recurrence and each later one closes its turn on `REPEAT_ASK`, the third
#                                         hands the unit to lane fallback1's one-shot where it is armed); the desk's
#                                         switch, or agent/n11/REPEAT, one word, an image's
REPEAT_FORMS = repeat.FORMS             # the loop's forms (form `state` answers its own re-sends)
REPEAT_KEYS = repeat.KEYS
REPEAT_ASIDE = "repeat-loop"            # the loop's output, set aside beside work/ while the one-shot runs
REPEAT_FLOOR = "repeat-oneshot"         # and the one-shot's, kept beside work/ where it passed the checks
LINT2_ENV = "AGENTHON_N11_LINT2"        # lane lint2: the write's two narrow findings (agent/n11/lint.py `check2`: an
LINT2_FILE = "LINT2"                    # imported name shadowed, a configuration value never read), `off` (today's) or
LINT2_MODES = ("off", "on")             # `on`, advisory lines of the write's answer after the lint's, apart from `LINT`;
#                                         the desk's switch, or agent/n11/LINT2, one word, an image's
LINT2_KEYS = ("lint2_mode", "lint2_shadowed", "lint2_unused")
EXAMPLE_ENV, EXAMPLE_FILE = example.ENV, example.FILE   # lane example1: the nearest solved public task in the loop's
EXAMPLE_SETTINGS = example.SETTINGS     # first message, `off` (today's: none) or `nearest` (agent/n11/example.py);
#                                         the desk's switch, or agent/n11/EXAMPLE, one word, an image's
EXAMPLE_FORMS = example.FORMS           # armed in the loop's forms (form `state` builds its own message)
EXAMPLE_KEYS = example.KEYS
SNAPSHOT_ENV, SNAPSHOT_FILE = snapshot.ENV, snapshot.FILE   # lane snapshot1: the state at an uncaught exception in
SNAPSHOT_SETTINGS = snapshot.SETTINGS   # the run's answer, `off` (today's) or `on` (agent/n11/snapshot.py); the desk's
#                                         switch, or agent/n11/SNAPSHOT, one word, an image's
SNAPSHOT_FORMS = snapshot.FORMS         # armed in the loop's forms (form `state` builds its own run text)
SNAPSHOT_KEYS = snapshot.KEYS
JOIN_ENV, JOIN_FILE = rejoin.ENV, rejoin.FILE   # lane join2 of 10-16: a continuation's join and a write over a saved
JOIN_SETTINGS = rejoin.SETTINGS     # file, `today` (today's) or `repair` (agent/n11/rejoin.py's repairs); the desk's
#                                     switch, or agent/n11/JOIN, one word, an image's
JOIN_FORMS = rejoin.FORMS           # armed in the loop's forms (form `state` keeps its own join and write)
REJOIN_KEYS = rejoin.KEYS           # beside lane join1's and join2's `JOIN_KEYS`
WRITE_THINKING_ENV, WRITE_THINKING_FILE = thinking.ENV, thinking.FILE   # lane thinkwrite1: a bounded thinking budget
WRITE_THINKING_SETTINGS = thinking.SETTINGS     # on the loop's write replies, `off` (today's), `first` or `all`
#                                                 (agent/n11/thinking.py); the desk's switch, or
#                                                 agent/n11/WRITE_THINKING, one word, an image's
WRITE_THINKING_BUDGET_ENV, WRITE_THINKING_BUDGET_FILE = thinking.BUDGET_ENV, thinking.BUDGET_FILE   # and its budget
WRITE_THINKING_FORMS = thinking.FORMS           # armed in the loop's forms (form `state` builds its own requests)
WRITE_THINKING_KEYS = thinking.KEYS
DERIVE_ENV, DERIVE_FILE = derive.ENV, derive.FILE   # lane derive1: the dated derivation asked in the loop's first
DERIVE_SETTINGS = derive.SETTINGS   # message and the harness's own checks said after each clean run, `off` (today's)
#                                     or `on` (agent/n11/derive.py, agent/n11/witness.py); the desk's switch, or
#                                     agent/n11/DERIVE, one word, an image's
DERIVE_FORMS = derive.FORMS         # armed in the loop's forms (form `state` builds its own message and run text)
DERIVE_KEYS = derive.KEYS
CONTRACT_ENV, CONTRACT_FILE = stated_contract.ENV, stated_contract.FILE     # lane contract1: the output contract the
CONTRACT_SETTINGS = stated_contract.SETTINGS    # instruction states, `off` (today's), `on` (named in the first message
#                                                 and said after each clean run) or `refuse` (and a finish turned away
#                                                 once while a finding stands; agent/n11/contract.py); the desk's
#                                                 switch, or agent/n11/CONTRACT, one word, an image's
CONTRACT_FORMS = stated_contract.FORMS          # armed in the loop's forms (form `state` builds its own message)
CONTRACT_KEYS = stated_contract.KEYS
CONTRACT_MAX = stated_contract.MAX              # the finishes the contract turns away a unit; the next is today's
CONTRACT_SAID = stated_contract.REFUSED         # the refusal's text, the findings after it
STATED_ENV = "AGENTHON_N11_STATED"      # lane conventions1: the system prompt's conventions, `off` (today's) or `on`
STATED_FILE = "STATED"                  # (row eligibility and clocks said operation by operation, `ELIGIBILITY_LINES`,
STATED_SETTINGS = ("off", "on")         # and each convention the task pins stated once in the script, `SETTLE_STATED`);
#                                         the desk's switch, or agent/n11/STATED, one word, an image's
STATED_FORMS = ("loop", "review", "sketch")     # armed in the loop's forms (form `state` keeps its own prompt)
STATED_KEYS = ("stated_mode",)
RECON_ENV, RECON_FILE = recon.ENV, recon.FILE   # lane recon1: the reconciliation after a clean run, `off` (today's),
RECON_SETTINGS = recon.SETTINGS     # `on` (said after each clean run) or `refuse` (and a finish turned away once while
#                                     a finding stands; agent/n11/recon.py); the desk's switch, or agent/n11/RECON,
#                                     one word, an image's
RECON_FORMS = recon.FORMS           # armed in the loop's forms (form `state` builds its own run text and finish)
RECON_KEYS = recon.KEYS
RECON_MAX = recon.MAX               # the finishes the reconciliation turns away a unit; the next is today's
RECON_SAID = recon.REFUSED          # the refusal's text, the findings after it
AUDIT_ENV, AUDIT_FILE = audit.ENV, audit.FILE   # lane audit1: the audit on the first clean run, `off` (today's) or
AUDIT_SETTINGS = audit.SETTINGS     # `on` (the system prompt's fifth step, the audit after the unit's first clean run
#                                     and a finish in its reply held once; agent/n11/audit.py); the desk's switch, or
#                                     agent/n11/AUDIT, one word, an image's
AUDIT_FORMS = audit.FORMS           # armed in the loop's forms (form `state` builds its own run text and finish)
AUDIT_KEYS = audit.KEYS
AUDIT_MAX = audit.MAX               # the finishes the audit holds a unit; the next is today's
DRAFTS_ENV, DRAFTS_FILE = drafts.ENV, drafts.FILE   # lane drafts1: the first drafts before the loop, `off` (today's),
DRAFTS_SETTINGS = drafts.SETTINGS   # `one` (the one-shot's floor) or `three` (the floor laid, and the table said where
#                                     they disagree; agent/n11/drafts.py); the desk's switch, or agent/n11/DRAFTS, an image's
DRAFTS_FORMS = drafts.FORMS         # armed in the forms whose loop opens on its own first message
DRAFTS_KEYS = drafts.KEYS
REPAIR_ENV, REPAIR_FILE = repair.ENV, repair.FILE   # lane repairpair1: the repair pair, `off` (today's) or `pair` (a
REPAIR_SETTINGS = repair.SETTINGS   # rewrite sent beside each repair, kept by execution; agent/n11/repair.py); the
REPAIR_FORMS = repair.FORMS         # desk's switch, or agent/n11/REPAIR, an image's; armed in forms `loop` and `sketch`
REPAIR_KEYS = repair.KEYS
OUTPUT_KEYS = ("output_env",)       # lane outguard1's: the loop's runs with OUTPUT_DIR naming the output
EXAMPLE_EXCLUDE_ENV = example.EXCLUDE_ENV   # the entries retrieval skips (unit and split names, commas): the desk's
KEPT = "clean-output"               # the last clean run's output, copied beside work/ (`keep_clean`)
KEPT_BYTES = 8 * 1024 * 1024        # an output larger than this in all is not copied, and none is kept then
CUT_SHAPES = ("calls", "length", "fenced", "empty")     # a reply at the cap: a call the route said finished, a call
#                                                         said cut (`length`), a fenced block, or nothing of either
TEXT_CALL = ("<tool_call>", '"arguments"', "<parameter=")  # a call written into the reply's text (`cut_from_text`)
RESENT_LINES = 3        # a continuation whose first lines repeat the kept head's first lines, this many at the least
#                         (or the whole head where shorter), after whitespace normalisation, re-sent it
HERE = Path(__file__).resolve().parent

LIBRARIES = ("numpy, pandas, scipy, statsmodels, scikit-learn, arch, polars, pyarrow, openpyxl, matplotlib (the Agg "
             "backend), plotly, seaborn, TA-Lib (imported as talib) and backtrader")

# form `review`'s request 1: N0's own, by import (agent/n0/solve.py; tests/test_n11.py pins it to the request N0 sends)
N0_SYSTEM = n0.SYSTEM
n0_prompt = n0.prompt


def release(version: str) -> tuple[int, int]:
    """A version's major and minor numbers, (0, 0) where it names none."""
    found = re.match(r"(\d+)(?:\.(\d+))?", version or "")
    return (int(found.group(1)), int(found.group(2) or 0)) if found else (0, 0)


def installed() -> dict[str, str]:
    """The versions of `API_DISTRIBUTIONS` this process runs under, read from the installed distributions (in the
    image, the image's own: never guessed), a distribution not installed left out: `api_card`'s, read once a process
    there, and lane prompt1's `lean` runtime line (`prompts.versions_said`), once a unit."""
    versions: dict[str, str] = {}
    for name in API_DISTRIBUTIONS:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return versions


@functools.lru_cache(maxsize=1)
def api_card() -> str:
    """The runtime paragraph of the loop's system prompt (lane n11e): the Python and library versions this process
    runs under (`installed`), and for each library the calls its version removed or changed (`API_NOTES`, the newest
    note its version reaches)."""
    versions = installed()
    notes, said = [], set()
    for name, least, note in API_NOTES:
        if name in versions and name not in said and release(versions[name]) >= least:
            notes.append(note)
            said.add(name)
    return (f"The runtime: Python {platform.python_version()}"
            + "".join(f", {name} {version}" for name, version in versions.items()) + ". " + " ".join(notes)).strip()


def said_of(environ, here: Path, env: str, file: str | None) -> tuple[str, str]:
    """A setting as said and where: the variable `env` where it is set, else the one word of the file `file` under
    `here` where that file is (no file where `file` is None), else ""."""
    environ = os.environ if environ is None else environ
    said, source = (environ.get(env) or "").strip(), env
    if not said and file is not None:
        try:
            said, source = (here / file).read_text(encoding="utf-8").strip(), file
        except (OSError, ValueError):
            said = ""
    return said, source


def chosen(environ, here: Path, env: str, file: str | None, choices, default: str, what: str) -> tuple[str, str]:
    """A choice this process runs and where it was read: the variable `env` where it is set, else the one word of the
    file `file` under `here` where that file is, else `default`. A word that is none of `choices` runs `default`, and
    the source says so."""
    said, source = said_of(environ, here, env, file)
    if not said:
        return default, f"no {what} named"
    if said not in choices:
        return default, f"{source} names {said[:40]!r}, no {what}"
    return said, source


def form_of(environ=None, here: Path = HERE) -> tuple[str, str]:
    """The form this process runs (lane loop1) and where it was read: `AGENTHON_N11_FORM` where it is set, else the
    one word of `agent/n11/FORM` where that file is, else `loop` (lane prune1). A name that is no form (`FORMS`) is
    refused: `loop` runs, and the source says so with the forms' names."""
    form, source = chosen(environ, here, FORM_ENV, FORM_FILE, FORMS, FORMS[0], "form")
    return form, f"{source}: refused; the forms are {', '.join(FORMS)}" if source.endswith(", no form") else source


FORM, FORM_SOURCE = form_of()           # read once at import


def mode_of(environ=None, here: Path = HERE) -> tuple[str, str]:
    """The reply form the loop's forms start in (lane fenced1) and where the setting was read: `AGENTHON_N11_MODE`
    where it is set, else the one word of `agent/n11/MODE` where that file is, else `tools`. A word that is no setting
    (`MODES`) runs `tools`, and the source says so."""
    return chosen(environ, here, MODE_ENV, MODE_FILE, MODES, MODES[0], "setting")


MODE, MODE_SOURCE = mode_of()           # read once at import


def prompt_of(environ=None, here: Path = HERE) -> tuple[str, str]:
    """The loop's prompt (lane prompt1, agent/n11/prompts.py) and where the setting was read: `AGENTHON_N11_PROMPT`
    where it is set, else the one word of `agent/n11/PROMPT`, else `full`. A word that is no setting
    (`PROMPT_SETTINGS`) runs `full`, and the source says so."""
    return chosen(environ, here, PROMPT_ENV, PROMPT_FILE, PROMPT_SETTINGS, "full", "setting")


PROMPT, PROMPT_SOURCE = prompt_of()     # read once at import


def lean_cards_of(environ=None, here: Path = HERE) -> tuple[str, str]:
    """The lean prompt's cards' form (lane lean2, `prompts.lean_card_as`) and where the setting was read:
    `AGENTHON_N11_LEAN_CARDS` where it is set, else the one word of `agent/n11/LEAN_CARDS`, else `cut`. A word that is
    no setting (`LEAN_CARDS_SETTINGS`: `cut`, `whole` or a whole number within `prompts.LEAN_CARD_BOUNDS`) runs `cut`,
    and the source says so."""
    return chosen(environ, here, LEAN_CARDS_ENV, LEAN_CARDS_FILE, LEAN_CARDS_SETTINGS, "cut", "setting")


LEAN_CARDS, LEAN_CARDS_SOURCE = lean_cards_of()     # read once at import


def rules_of(environ=None, here: Path = HERE) -> tuple[str, str]:
    """The output rules and the finish's refusal on them (lane rules1) and where the setting was read:
    `AGENTHON_N11_RULES` where it is set, else the one word of `agent/n11/RULES`, else `off` (`wide`: lane rules2's).
    A word that is no setting (`RULES_SETTINGS`) runs `off`, and the source says so."""
    return chosen(environ, here, RULES_ENV, RULES_FILE, RULES_SETTINGS, "off", "setting")


RULES, RULES_SOURCE = rules_of()        # read once at import


def resend_of(environ=None, here: Path = HERE) -> tuple[str, str]:
    """An unchanged re-send read as the finish (lane resend1) and where the setting was read: `AGENTHON_N11_RESEND`
    where it is set, else the one word of `agent/n11/RESEND`, else `refuse`. A word that is no setting
    (`RESEND_SETTINGS`) runs `refuse`, and the source says so."""
    return chosen(environ, here, RESEND_ENV, RESEND_FILE, RESEND_SETTINGS, "refuse", "setting")


RESEND, RESEND_SOURCE = resend_of()     # read once at import


def finishrun_of(environ=None, here: Path = HERE) -> tuple[str, str]:
    """A finish after a write run first (lane finishrun1) and where the setting was read: `AGENTHON_N11_FINISHRUN`
    where it is set, else the one word of `agent/n11/FINISHRUN`, else `off`. A word that is no setting
    (`FINISHRUN_SETTINGS`) runs `off`, and the source says so."""
    return chosen(environ, here, FINISHRUN_ENV, FINISHRUN_FILE, FINISHRUN_SETTINGS, "off", "setting")


FINISHRUN, FINISHRUN_SOURCE = finishrun_of()    # read once at import


def finishfail_of(environ=None, here: Path = HERE) -> tuple[str, str]:
    """A finish on a failing run not taken (lane finishfail1) and where the setting was read: `AGENTHON_N11_FINISHFAIL`
    where it is set, else the one word of `agent/n11/FINISHFAIL`, else `off`. A word that is no setting
    (`FINISHFAIL_SETTINGS`) runs `off`, and the source says so."""
    return chosen(environ, here, FINISHFAIL_ENV, FINISHFAIL_FILE, FINISHFAIL_SETTINGS, "off", "setting")


FINISHFAIL, FINISHFAIL_SOURCE = finishfail_of()     # read once at import


def finishcheck_of(environ=None, here: Path = HERE) -> tuple[str, str]:
    """The finish contract (lane finishcheck1) and where the setting was read: `AGENTHON_N11_FINISHCHECK` where it is
    set, else the one word of `agent/n11/FINISHCHECK`, else `off`. A word that is no setting (`FINISHCHECK_SETTINGS`)
    runs `off`, and the source says so."""
    return chosen(environ, here, FINISHCHECK_ENV, FINISHCHECK_FILE, FINISHCHECK_SETTINGS, "off", "setting")


FINISHCHECK, FINISHCHECK_SOURCE = finishcheck_of()  # read once at import


def firstout_of(environ=None, here: Path = HERE) -> tuple[str, str]:
    """The first output first (lane firstout1) and where the setting was read: `AGENTHON_N11_FIRSTOUT` where it is
    set, else the one word of `agent/n11/FIRSTOUT`, else `off`. A word that is no setting (`FIRSTOUT_SETTINGS`) runs
    `off`, and the source says so."""
    return chosen(environ, here, FIRSTOUT_ENV, FIRSTOUT_FILE, FIRSTOUT_SETTINGS, "off", "setting")


FIRSTOUT, FIRSTOUT_SOURCE = firstout_of()       # read once at import


def clause_of(environ=None, here: Path = HERE) -> tuple[str, str]:
    """The clause on task fidelity (lane clauses1) and where the setting was read: `AGENTHON_N11_CLAUSE` where it is
    set, else the one word of `agent/n11/CLAUSE`, else `off`. A word that is no setting (`CLAUSE_SETTINGS`) runs `off`,
    and the source says so."""
    return chosen(environ, here, CLAUSE_ENV, CLAUSE_FILE, CLAUSE_SETTINGS, "off", "setting")


CLAUSE, CLAUSE_SOURCE = clause_of()     # read once at import


def cardfail_of(environ=None, here: Path = HERE) -> tuple[str, str]:
    """The card after a failing library call (lane cardfail1) and where the setting was read: `AGENTHON_N11_CARDFAIL`
    where it is set, else the one word of `agent/n11/CARDFAIL`, else `off`. A word that is no setting
    (`CARDFAIL_SETTINGS`) runs `off`, and the source says so."""
    return chosen(environ, here, CARDFAIL_ENV, CARDFAIL_FILE, CARDFAIL_SETTINGS, "off", "setting")


CARDFAIL, CARDFAIL_SOURCE = cardfail_of()       # read once at import


def turns_away(exit_: int | None, requests: int, left: float, turned: int) -> bool:
    """Lane finishfail1, `FINISHFAIL=on`: whether a finish whose run exited `exit_` (None: stopped at its limit) is
    turned away: the run not exited 0, a request and `ASK_S` left (`finish`'s `able`), and fewer than
    `FINISHFAIL_MAX` finishes turned away before it in the unit. `finish` also requires the current checks to fail."""
    return exit_ != 0 and requests >= 1 and left >= ASK_S and turned < FINISHFAIL_MAX


def joinask_of(environ=None, here: Path = HERE) -> tuple[str, str]:
    """The ask after a failed continuation (lane joinask1) and where the setting was read: `AGENTHON_N11_JOINASK`
    where it is set, else the one word of `agent/n11/JOINASK`, else `same`. A word that is no setting
    (`JOINASK_SETTINGS`) runs `same`, and the source says so."""
    return chosen(environ, here, JOINASK_ENV, JOINASK_FILE, JOINASK_SETTINGS, "same", "setting")


JOINASK, JOINASK_SOURCE = joinask_of()  # read once at import


def repeat_of(environ=None, here: Path = HERE) -> tuple[str, str]:
    """The repeated failing program state (lane repeat1, agent/n11/repeat.py) and where the setting was read:
    `AGENTHON_N11_REPEAT` where it is set, else the one word of `agent/n11/REPEAT`, else `off`. A word that is no
    setting (`REPEAT_SETTINGS`) runs `off`, and the source says so."""
    return chosen(environ, here, REPEAT_ENV, REPEAT_FILE, REPEAT_SETTINGS, "off", "setting")


REPEAT, REPEAT_SOURCE = repeat_of()     # read once at import


def lint2_of(environ=None, here: Path = HERE) -> tuple[str, str]:
    """The write's two narrow findings (lane lint2, agent/n11/lint.py `check2`) and where the setting was read:
    `AGENTHON_N11_LINT2` where it is set, else the one word of `agent/n11/LINT2`, else `off`. A word that is no setting
    (`LINT2_MODES`) runs `off`, and the source says so."""
    return chosen(environ, here, LINT2_ENV, LINT2_FILE, LINT2_MODES, "off", "setting")


LINT2, LINT2_SOURCE = lint2_of()        # read once at import


def example_of(environ=None, here: Path = HERE) -> tuple[str, str]:
    """The worked example (lane example1, agent/n11/example.py) and where the setting was read: `AGENTHON_N11_EXAMPLE`
    where it is set, else the one word of `agent/n11/EXAMPLE`, else `off`. A word that is no setting
    (`EXAMPLE_SETTINGS`) runs `off`, and the source says so."""
    return chosen(environ, here, EXAMPLE_ENV, EXAMPLE_FILE, EXAMPLE_SETTINGS, "off", "setting")


def snapshot_of(environ=None, here: Path = HERE) -> tuple[str, str]:
    """The failure snapshot (lane snapshot1, agent/n11/snapshot.py) and where the setting was read:
    `AGENTHON_N11_SNAPSHOT` where it is set, else the one word of `agent/n11/SNAPSHOT`, else `off`. A word that is no
    setting (`SNAPSHOT_SETTINGS`) runs `off`, and the source says so."""
    return chosen(environ, here, SNAPSHOT_ENV, SNAPSHOT_FILE, SNAPSHOT_SETTINGS, "off", "setting")


SNAPSHOT, SNAPSHOT_SOURCE = snapshot_of()   # read once at import


def join_of(environ=None, here: Path = HERE) -> tuple[str, str]:
    """The join repaired (lane join2 of 10-16, agent/n11/rejoin.py) and where the setting was read: `AGENTHON_N11_JOIN`
    where it is set, else the one word of `agent/n11/JOIN`, else `today`. A word that is no setting (`JOIN_SETTINGS`)
    runs `today`, and the source says so."""
    return chosen(environ, here, JOIN_ENV, JOIN_FILE, JOIN_SETTINGS, "today", "setting")


JOIN, JOIN_SOURCE = join_of()   # read once at import


def write_thinking_of(environ=None, here: Path = HERE) -> tuple[str, str]:
    """The write replies' thinking (lane thinkwrite1) and where the setting was read: `AGENTHON_N11_WRITE_THINKING`
    where it is set, else the one word of `agent/n11/WRITE_THINKING`, else `off`. A word that is no setting
    (`WRITE_THINKING_SETTINGS`) runs `off`, and the source says so."""
    return chosen(environ, here, WRITE_THINKING_ENV, WRITE_THINKING_FILE, WRITE_THINKING_SETTINGS, "off", "setting")


WRITE_THINKING, WRITE_THINKING_SOURCE = write_thinking_of()     # read once at import


def write_thinking_budget_of(environ=None, here: Path = HERE) -> tuple[int | str, str]:
    """The write replies' thinking budget (lane thinkwrite1) and where it was read:
    `AGENTHON_N11_WRITE_THINKING_BUDGET` where it is set, else the one word of `agent/n11/WRITE_THINKING_BUDGET`, else
    `thinking.BUDGET`: `low` or a whole number within `thinking.BOUNDS`, any other word refused (`thinking.budget_of`)."""
    return thinking.budget_of(*said_of(environ, here, WRITE_THINKING_BUDGET_ENV, WRITE_THINKING_BUDGET_FILE))


WRITE_THINKING_BUDGET, WRITE_THINKING_BUDGET_SOURCE = write_thinking_budget_of()    # read once at import


def derive_of(environ=None, here: Path = HERE) -> tuple[str, str]:
    """The dated derivation (lane derive1, agent/n11/derive.py) and where the setting was read: `AGENTHON_N11_DERIVE`
    where it is set, else the one word of `agent/n11/DERIVE`, else `off`. A word that is no setting
    (`DERIVE_SETTINGS`) runs `off`, and the source says so."""
    return chosen(environ, here, DERIVE_ENV, DERIVE_FILE, DERIVE_SETTINGS, "off", "setting")


DERIVE, DERIVE_SOURCE = derive_of()     # read once at import


def contract_of(environ=None, here: Path = HERE) -> tuple[str, str]:
    """The output contract (lane contract1, agent/n11/contract.py) and where the setting was read:
    `AGENTHON_N11_CONTRACT` where it is set, else the one word of `agent/n11/CONTRACT`, else `off`. A word that is no
    setting (`CONTRACT_SETTINGS`) runs `off`, and the source says so."""
    return chosen(environ, here, CONTRACT_ENV, CONTRACT_FILE, CONTRACT_SETTINGS, "off", "setting")


CONTRACT, CONTRACT_SOURCE = contract_of()   # read once at import


def stated_of(environ=None, here: Path = HERE) -> tuple[str, str]:
    """The conventions stated once (lane conventions1) and where the setting was read: `AGENTHON_N11_STATED` where it
    is set, else the one word of `agent/n11/STATED`, else `off`. A word that is no setting (`STATED_SETTINGS`) runs
    `off`, and the source says so."""
    return chosen(environ, here, STATED_ENV, STATED_FILE, STATED_SETTINGS, "off", "setting")


STATED, STATED_SOURCE = stated_of()     # read once at import


def recon_of(environ=None, here: Path = HERE) -> tuple[str, str]:
    """The reconciliation after a clean run (lane recon1, agent/n11/recon.py) and where the setting was read:
    `AGENTHON_N11_RECON` where it is set, else the one word of `agent/n11/RECON`, else `off`. A word that is no setting
    (`RECON_SETTINGS`) runs `off`, and the source says so."""
    return chosen(environ, here, RECON_ENV, RECON_FILE, RECON_SETTINGS, "off", "setting")


RECON, RECON_SOURCE = recon_of()    # read once at import


def audit_of(environ=None, here: Path = HERE) -> tuple[str, str]:
    """The audit on the first clean run (lane audit1, agent/n11/audit.py) and where the setting was read:
    `AGENTHON_N11_AUDIT` where it is set, else the one word of `agent/n11/AUDIT`, else `off`. A word that is no setting
    (`AUDIT_SETTINGS`) runs `off`, and the source says so."""
    return chosen(environ, here, AUDIT_ENV, AUDIT_FILE, AUDIT_SETTINGS, "off", "setting")


AUDIT, AUDIT_SOURCE = audit_of()    # read once at import


def drafts_of(environ=None, here: Path = HERE) -> tuple[str, str]:
    """The first drafts before the loop (lane drafts1, agent/n11/drafts.py) and where the setting was read:
    `AGENTHON_N11_DRAFTS` where it is set, else the one word of `agent/n11/DRAFTS`, else `off` (`one` and `three` the
    drafts sent). A word that is no setting (`DRAFTS_SETTINGS`) runs `off`, and the source says so."""
    return chosen(environ, here, DRAFTS_ENV, DRAFTS_FILE, DRAFTS_SETTINGS, "off", "setting")


DRAFTS, DRAFTS_SOURCE = drafts_of()     # read once at import


def repair_of(environ=None, here: Path = HERE) -> tuple[str, str]:
    """The repair pair (lane repairpair1, agent/n11/repair.py) and where the setting was read: `AGENTHON_N11_REPAIR`
    where it is set, else the one word of `agent/n11/REPAIR`, else `off` (`pair` the rewrite sent beside each repair).
    A word that is no setting (`REPAIR_SETTINGS`) runs `off`, and the source says so."""
    return chosen(environ, here, REPAIR_ENV, REPAIR_FILE, REPAIR_SETTINGS, "off", "setting")


REPAIR, REPAIR_SOURCE = repair_of()     # read once at import
# lane repairpair1: the loop's state a turn changes and the next turn reads, put back as A's turn found it where the
#   pair keeps B (`repair_settle`): the files' versions and runs, the errors seen, the continuation's and the repeat's
#   ledgers, the streaks, the finish's counts and the last run as shown
TURN_STATE = ("versions", "wrote", "last_ran", "clean_runs", "runs_at", "gave", "outcome", "seen", "last_frame", "entry",
              "gate_said", "held", "held_tail", "joining", "stitching", "staged", "defs_due", "defs_files", "defs_mark",
              "head_ran", "cut_named", "cut_lines", "said_broken", "kept_said", "defs_repeats", "junctions", "recut",
              "_cut_streak", "no_call_streak", "finishes", "sanity_refused", "rules_held", "resent", "last_main",
              "run_failed", "last_clean", "first_outputs", "shown_run", "tail_budget", "repeats", "repeat_hit",
              "audit_met", "audit_at", "recon_standing", "recon_scripts")
TURN_STATE += ("counts", "messages", "recurrences", "digests", "streak_top", "reply_text", "finish_said",
               "replies", "request_seconds", "run_seconds", "short", "short_due", "first_write", "nocall_due",
               "end_kind", "ending", "last_kind", "lastcall", "shim_fired", "redirect_inspects", "redirect_at",
               "redirect_left", "redirect_due", "redirect_saved", "first_saved", "called_named", "called_logged",
               "called_alone", "rules_fired", "kept", "kept_snapshot", "finishcheck_names", "finishcheck_found",
               "cardfail_carded", "cardfail_looked", "repeat_due", "repeat_floor", "snapshot_sizes", "derive_said",
               "derive_script", "witness_kinds", "contract_kinds", "recon_kinds", "repeats_asked", "turn_calls",
               "prompt_attempt", "prompt_attempt_saved", "prompt_first", "prompt_missed", "sketch_kept",
               "sketch_note", "sketch_due", "sketch_did", "sketch_lines", "sketch_thought", "sketch_inside",
               "review_reply", "review_last", "review_asked", "mode", "mode_refused", "system_said", "thought")


def exclude_of(environ=None) -> tuple[frozenset[str], str]:
    """The names the example's retrieval skips (lane example1): `AGENTHON_N11_EXAMPLE_EXCLUDE`'s comma list, unit and
    split names (`example.exclusion`); none where it is unset. The desk's alone: an image carries no file of it."""
    said, source = said_of(environ, HERE, EXAMPLE_EXCLUDE_ENV, None)
    return example.exclusion(said), source if said else "none named"


EXAMPLE, EXAMPLE_SOURCE = example_of()  # read once at import
EXAMPLE_EXCLUDE, EXAMPLE_EXCLUDE_SOURCE = exclude_of()
EXAMPLES = HERE / example.CORPUS        # the corpus the switch reads (scripts/examples_corpus.py builds it)


def state_result_of(environ=None, here: Path = HERE) -> tuple[str, str]:
    """Form `state`'s LAST RUN setting (lane state2; agent/n11/state.py `last_run`) and where it was read:
    `AGENTHON_N11_STATE_RESULT` where it is set, else the one word of `agent/n11/STATE_RESULT` where that file is, else
    `full`. A word that is no setting (`state.RESULTS`) runs `full`, and the source says so."""
    return chosen(environ, here, STATE_RESULT_ENV, STATE_RESULT_FILE, state.RESULTS, "full", "setting")


STATE_RESULT, STATE_RESULT_SOURCE = state_result_of()   # read once at import


def sketch_of(environ=None, here: Path = HERE) -> tuple[str, str]:
    """Form `sketch`'s switch (lane plan4; the owner's in316) and where it was read: `AGENTHON_N11_SKETCH` where it is
    set, else the one word of `agent/n11/SKETCH` where that file is, else `turn`. A word that is no switch (`SKETCHES`)
    runs `turn`, and the source says so."""
    return chosen(environ, here, SKETCH_ENV, SKETCH_FILE, SKETCHES, "turn", "switch")


SKETCH, SKETCH_SOURCE = sketch_of()     # read once at import


def sketch_thinking_of(environ=None, here: Path = HERE) -> tuple[str, str]:
    """The sketch's request's thinking (lane plan4's addendum 4, the owner's in344) and where it was read:
    `AGENTHON_N11_SKETCH_THINKING` where it is set, else the one word of `agent/n11/SKETCH_THINKING`, else `off`. A
    word that is no setting (`SKETCH_THINKINGS`) runs `off`, and the source says so."""
    return chosen(environ, here, SKETCH_THINKING_ENV, SKETCH_THINKING_FILE, SKETCH_THINKINGS, "off", "setting")


SKETCH_THINKING, SKETCH_THINKING_SOURCE = sketch_thinking_of()     # read once at import


def index_mode_of(environ=None, here: Path = HERE) -> tuple[str, str]:
    """The index's setting (lane index2) and where it was read: `AGENTHON_N11_INDEX` where it is set, else the one word
    of `agent/n11/INDEX` where that file is, else `full`. A word that is no setting (`library.INDEX_MODES`) runs
    `full`, and the source says so."""
    return chosen(environ, here, INDEX_ENV, INDEX_SWITCH, library.INDEX_MODES, "full", "index setting")


INDEX_MODE, INDEX_MODE_SOURCE = index_mode_of()     # read once at import


def inputs_of(environ=None, here: Path = HERE) -> tuple[str, str]:
    """The input files' descriptions (lane inputs1) and where the setting was read: `AGENTHON_N11_INPUTS` where it is
    set, else the one word of `agent/n11/INPUTS`, else `standard`. A word that is no setting (`INPUTS_SETTINGS`) runs
    `standard`, and the source says so."""
    return chosen(environ, here, INPUTS_ENV, INPUTS_FILE, INPUTS_SETTINGS, "standard", "setting")


INPUTS, INPUTS_SOURCE = inputs_of()     # read once at import


def review_thinking_of(environ=None) -> tuple[str, str]:
    """Form `review`'s review request's thinking (lane inputs1) and where it was read: `low` where the desk's
    `AGENTHON_N11_REVIEW_THINKING` says so, else `off`; an image never sets it."""
    return chosen(environ, HERE, REVIEW_THINKING_ENV, None, ("off", "low"), "off", "setting")


REVIEW_THINKING, REVIEW_THINKING_SOURCE = review_thinking_of()     # read once at import


def requests_of(environ=None, here: Path = HERE) -> tuple[int, str]:
    """The loop's request cap (lane inputs1) and where it was read: `AGENTHON_N11_REQUESTS` where it is set, else the
    one number of `agent/n11/REQUESTS`, else `REQUESTS`. A cap that is no whole number or lies outside
    `REQUESTS_BOUNDS` is refused: `REQUESTS` runs, and the source says why."""
    said, source = said_of(environ, here, REQUESTS_ENV, REQUESTS_FILE)
    if not said:
        return REQUESTS, "no cap named"
    low, high = REQUESTS_BOUNDS
    if not (said.isascii() and said.isdigit()):     # `²` is a digit to isdigit and no number to int
        return REQUESTS, f"{source} names {said[:40]!r}, no cap: refused"
    if not low <= int(said) <= high:
        return REQUESTS, f"{source} names {int(said)}, {'above' if int(said) > high else 'below'} " \
                         f"{high if int(said) > high else low}: refused"
    return int(said), source


CAP, CAP_SOURCE = requests_of()         # read once at import


def nocall_of(environ=None, here: Path = HERE) -> tuple[str, str]:
    """The loop's answer to a reply with no call (lane cap1) and where it was read: `AGENTHON_N11_NOCALL` where it is
    set, else the one word of `agent/n11/NOCALL`, else `stop`. A word that is no setting (`NOCALLS`) runs `stop`, and
    the source says so."""
    return chosen(environ, here, NOCALL_ENV, NOCALL_FILE, NOCALLS, "stop", "setting")


NOCALL, NOCALL_SOURCE = nocall_of()     # read once at import


def picks_of(environ=None, here: Path = HERE) -> tuple[str, str]:
    """Where the sketch's picks are read from (lane cap1) and where the setting was read: `AGENTHON_N11_SKETCH_PICKS`
    where it is set, else the one word of `agent/n11/SKETCH_PICKS`, else `words`. A word that is no setting
    (`PICKINGS`) runs `words`, and the source says so."""
    return chosen(environ, here, PICKS_ENV, PICKS_FILE, PICKINGS, "words", "setting")


SKETCH_PICKS, SKETCH_PICKS_SOURCE = picks_of()  # read once at import


def pace_of(environ=None, here: Path = HERE) -> tuple[str, str]:
    """The loop at the platform's pace (lane pace1) and where the setting was read: `AGENTHON_N11_PACE` where it is
    set, else the one word of `agent/n11/PACE`, else `off`. A word that is no setting (`PACE_SETTINGS`) runs `off`, and
    the source says so."""
    return chosen(environ, here, PACE_ENV, PACE_FILE, PACE_SETTINGS, "off", "setting")


PACE, PACE_SOURCE = pace_of()           # read once at import


def fallback_of(environ=None, here: Path = HERE) -> tuple[str, str]:
    """The last resort (lane fallback1, agent/n11/fallback.py) and where the setting was read: `AGENTHON_N11_FALLBACK`
    where it is set, else the one word of `agent/n11/FALLBACK`, else `off`. A word that is no setting
    (`fallback.SETTINGS`) runs `off`, and the source says so."""
    return chosen(environ, here, fallback.ENV, fallback.FILE, fallback.SETTINGS, "off", "setting")


FALLBACK, FALLBACK_SOURCE = fallback_of()   # read once at import


def select_of(environ=None, here: Path = HERE) -> tuple[str, str]:
    """The selection call (lane select2, agent/n11/selection.py) and where the setting was read: `AGENTHON_N11_SELECT`
    where it is set, else the one word of `agent/n11/SELECT`, else `off`. A word that is no setting
    (`selection.SETTINGS`) runs `off`, and the source says so; so does a desk-only one (`selection.DESK_ONLY`, lane
    given1's `given`) the file names, an image's, refused (agent/Dockerfile refuses its build too)."""
    setting, source = chosen(environ, here, selection.ENV, selection.FILE, selection.SETTINGS, "off", "setting")
    if setting in selection.DESK_ONLY and source == selection.FILE:
        return "off", f"{source} names {setting!r}: {selection.REFUSED}"
    return setting, source


SELECT, SELECT_SOURCE = select_of()     # read once at import


def select_thinking_of(environ=None, here: Path = HERE) -> tuple[str, str]:
    """The selection call's thinking (lane select2) and where it was read: `AGENTHON_N11_SELECT_THINKING` where it is
    set, else the one word of `agent/n11/SELECT_THINKING`, else `off`. A word that is no setting
    (`selection.THINKINGS`) runs `off`, and the source says so."""
    return chosen(environ, here, selection.THINKING_ENV, selection.THINKING_FILE, selection.THINKINGS, "off",
                  "setting")


SELECT_THINKING, SELECT_THINKING_SOURCE = select_thinking_of()     # read once at import


def convcard_of(environ=None, here: Path = HERE) -> tuple[str, str]:
    """The conventions card (lane convcard1, agent/n11/conventions.py) and where the setting was read:
    `AGENTHON_N11_CONVENTIONS` where it is set, else the one word of `agent/n11/CONVENTIONS`, else `off`. A word that
    is no setting (`CONVCARD_SETTINGS`) runs `off`, and the source says so."""
    return chosen(environ, here, CONVCARD_ENV, CONVCARD_FILE, CONVCARD_SETTINGS, "off", "setting")


CONVCARD, CONVCARD_SOURCE = convcard_of()   # read once at import


def cut_of(environ=None, here: Path = HERE) -> tuple[str, str]:
    """A cut write's handling (lane review2) and where the setting was read: `AGENTHON_N11_CUT` where it is set, else
    the one word of `agent/n11/CUT`, else `split`. A word that is no setting (`CUT_MODES`) runs `split`, and the
    source says so."""
    return chosen(environ, here, CUT_ENV, CUT_FILE, CUT_MODES, "split", "setting")


CUT_MODE, CUT_MODE_SOURCE = cut_of()    # read once at import


def write_of(environ=None, here: Path = HERE) -> tuple[str, str]:
    """A write of a .py file that exists (lane fixdef1) and where the setting was read: `AGENTHON_N11_WRITE` where it
    is set, else the one word of `agent/n11/WRITE`, else `whole`. A word that is no setting (`WRITE_MODES`) runs
    `whole`, and the source says so."""
    return chosen(environ, here, WRITE_ENV, WRITE_FILE, WRITE_MODES, "whole", "setting")


WRITE_MODE, WRITE_MODE_SOURCE = write_of()  # read once at import


def cards_of(environ=None, here: Path = HERE) -> tuple[str, str]:
    """The first message's cards (lane first1) and where the setting was read: `AGENTHON_N11_CARDS` where it is set,
    else the one word of `agent/n11/CARDS`, else `cut`. A word that is no setting (`CARDS_MODES`) runs `cut`, and the
    source says so."""
    return chosen(environ, here, CARDS_ENV, CARDS_FILE, CARDS_MODES, "cut", "setting")


CARDS_MODE, CARDS_MODE_SOURCE = cards_of()      # read once at import


def inspect_of(environ=None, here: Path = HERE) -> tuple[str, str]:
    """An inspect of a described input file (lane first1) and where the setting was read: `AGENTHON_N11_INSPECT` where
    it is set, else the one word of `agent/n11/INSPECT`, else `open`. A word that is no setting (`INSPECT_MODES`) runs
    `open`, and the source says so."""
    return chosen(environ, here, INSPECT_ENV, INSPECT_FILE, INSPECT_MODES, "open", "setting")


INSPECT_MODE, INSPECT_MODE_SOURCE = inspect_of()    # read once at import


def select_bound_of(environ=None, here: Path = HERE) -> tuple[str, str]:
    """The selection call's bound (lane first1, agent/n11/selection.py) and where the setting was read:
    `AGENTHON_N11_SELECT_BOUND` where it is set, else the one word of `agent/n11/SELECT_BOUND`, else `route`. A word
    that is no setting (`selection.BOUNDS`) runs `route`, and the source says so."""
    return chosen(environ, here, selection.BOUND_ENV, selection.BOUND_FILE, selection.BOUNDS, "route", "setting")


SELECT_BOUND, SELECT_BOUND_SOURCE = select_bound_of()   # read once at import


def lastcall_of(environ=None, here: Path = HERE) -> tuple[str, str]:
    """When the last call is due (lane clock1 of 10-13) and where the setting was read: `AGENTHON_N11_LASTCALL` where
    it is set, else the one word of `agent/n11/LASTCALL`, else `late`. A word that is no setting (`LASTCALL_MODES`)
    runs `late`, and the source says so."""
    return chosen(environ, here, LASTCALL_ENV, LASTCALL_FILE, LASTCALL_MODES, "late", "setting")


LASTCALL, LASTCALL_SOURCE = lastcall_of()   # read once at import


def shim_of(environ=None, here: Path = HERE) -> tuple[str, str]:
    """The run's shim (lane clock1 of 10-13, agent/n11/shim.py) and where the setting was read: `AGENTHON_N11_SHIM`
    where it is set, else the one word of `agent/n11/SHIM`, else `off`. A word that is no setting (`SHIM_MODES`) runs
    `off`, and the source says so."""
    return chosen(environ, here, SHIM_ENV, SHIM_FILE, SHIM_MODES, "off", "setting")


SHIM, SHIM_SOURCE = shim_of()       # read once at import


def lint_of(environ=None, here: Path = HERE) -> tuple[str, str]:
    """The write's static check (lane clock1 of 10-13, agent/n11/lint.py) and where the setting was read:
    `AGENTHON_N11_LINT` where it is set, else the one word of `agent/n11/LINT`, else `off`. A word that is no setting
    (`LINT_MODES`) runs `off`, and the source says so."""
    return chosen(environ, here, LINT_ENV, LINT_FILE, LINT_MODES, "off", "setting")


LINT, LINT_SOURCE = lint_of()       # read once at import


def redirect_of(environ=None, here: Path = HERE) -> tuple[str, str]:
    """A wasting loop redirected to its script (lane redirect1) and where the setting was read: `AGENTHON_N11_REDIRECT`
    where it is set, else the one word of `agent/n11/REDIRECT`, else `off`. A word that is no setting
    (`REDIRECT_MODES`) runs `off`, and the source says so."""
    return chosen(environ, here, REDIRECT_ENV, REDIRECT_FILE, REDIRECT_MODES, "off", "setting")


REDIRECT_MODE, REDIRECT_MODE_SOURCE = redirect_of()     # read once at import


def lib_of(environ=None, here: Path = HERE) -> tuple[str, str]:
    """The library lookup (lane libnever1) and where the setting was read: `AGENTHON_N11_LIB` where it is set, else the
    one word of `agent/n11/LIB`, else `open`. A word that is no setting (`LIB_MODES`) runs `open`, and the source says
    so."""
    return chosen(environ, here, LIB_ENV, LIB_FILE, LIB_MODES, "open", "setting")


LIB_MODE, LIB_MODE_SOURCE = lib_of()    # read once at import


def called_of(environ=None, here: Path = HERE) -> tuple[str, str]:
    """The called finding (lane called1) and where the setting was read: `AGENTHON_N11_CALLED` where it is set, else
    the one word of `agent/n11/CALLED`, else `off`. A word that is no setting (`CALLED_MODES`) runs `off`, and the
    source says so."""
    return chosen(environ, here, CALLED_ENV, CALLED_FILE, CALLED_MODES, "off", "setting")


CALLED, CALLED_SOURCE = called_of()     # read once at import


# form `review`'s library pass (lane n11c): one request after a clean first run where the pointer named functions
LIBPASS_SYSTEM = """You are an expert quantitative developer. You revise a Python 3 script that ran cleanly so that \
it calls the method library qf where qf computes what the script computes; the revised script is run once more, \
unattended, with no network, and kept only where it runs as cleanly and writes the same files.

Reply with the whole revised script in a single ```python fenced block."""

LIBPASS_ASK = """# The library's cards

{cards}

# The script

```python
{script}
```

# What to do

The script above ran cleanly and wrote every output file. qf, the method library these cards come from, is installed \
(import qf). Return the same script with each quantity a card's function computes computed by calling that function \
as its card writes it (`qf.<name>`), under the conventions its card states, each argument set to the \
value the script uses; and nothing else changed: the same inputs and paths, the same output files with the same \
names, keys, columns and formats. Leave as it is any part no card computes. Reply with the whole script in one \
```python block.
"""
LIBPASS_FILE = "libpass.py"         # the library pass's script under work/; adopted, it becomes work/main.py
SET_ASIDE = "first-output"          # the first run's output while the library pass runs or the review is open


TOOLS_PART = """You act through five tools. `write` saves a file under work/, whole (a .py file is compiled); a \
write of work/main.py is followed by its run, and a write of another file by that file's run where it has an \
`if __name__ == "__main__":` block and work/main.py does not import it: the result comes back with the write's. `run` runs a .py file under work/ with python3 and reports \
its exit, its output's ends, its traceback, the files in the output directory with their values (a digest), the \
sanity findings, the checks and your budget. `inspect` describes an input data file. `lib` shows the library's \
documentation. `finish` ends the task, and nothing else does. Several calls in one reply run in order."""

FENCED_PART = """You act through the form of your reply. Each file is one fenced block whose opening line names it, \
```python file=work/<name>.py, and is saved whole. After the blocks, work/main.py runs where the reply wrote it (or \
the last file written, where it has an `if __name__ == "__main__":` block and work/main.py does not import it), or the \
file a line `run work/<name>.py` names. These lines, each alone on its line outside any block, are commands: `inspect <path> rows=<n>` describes an \
input data file; `lib name=<function>` or `lib shelf=<module>` shows the library's documentation; `finish` ends the \
task, and nothing else does. What each did comes back in the next message: the run's exit, its output's ends, its \
traceback, the files in the output directory with their values (a digest), the sanity findings, the checks and your \
budget."""

PART_DEFS = {   # lane fixdef1 (`WRITE=defs`): each form's paragraph saying what a write saves (commands.DEFS_SAID)
    #             where it said "whole"
    "tools": TOOLS_PART.replace("`write` saves a file under work/, whole (a .py file is compiled);",
                                f"`write` saves a file under work/ (a .py file is compiled): {commands.DEFS_SAID};", 1),
    "fenced": FENCED_PART.replace("```python file=work/<name>.py, and is saved whole.",
                                  f"```python file=work/<name>.py; {commands.DEFS_SAID}.", 1)}

TOOLS_EXAMPLE = """An example: one reply calls write(path="work/main.py", text=<the whole script: it reads the \
inputs, computes each value (def log_returns(prices): return np.diff(np.log(prices)), and the like) and writes every \
output file>), which is run for you; the next fixes what the run showed, or calls finish."""

FENCED_EXAMPLE = """An example reply: one block opened by ```python file=work/main.py holding the whole script, which \
reads the inputs, computes each value (a function log_returns(prices), and the like) and writes every output file, \
and nothing after it: main.py is run."""

SYSTEM = """You are an expert quantitative developer. You solve a task by writing a Python 3 script and running \
it, unattended, with no network; hidden tests then check the files your scripts wrote into the output directory \
against the task.

{tools}

How to work:
1. Read the task, its paths and its input files, which the first message describes in full, and {settle} \
Start by writing code: `inspect` a file only where its description leaves you short.
2. Write the whole solution as one script, work/main.py, that reads the inputs, computes each quantity (a quantity \
the library computes is a call of its function) and writes every output file the task asks for.
3. A write of work/main.py runs it: read each run. Where you split the script into modules, write them first and \
main.py last; a module's write runs work/main.py only where main.py imports it and its last run was not clean: after \
a clean run, where you change a module, run work/main.py after it in the same reply.
4. Where a run, a check or a sanity finding fails, fix the failing piece: write that one file again, whole.
5. Read each run's digest of the output files against what the task says of each value (its sign, its range, its \
order: ES at least VaR, a probability in [0, 1]); finish when the checks pass and the values hold.

A reply that runs past the output limit is cut, and a write the cut falls in is not saved, only its complete \
definitions as a draft. So write work/main.py whole, in one reply, where one reply holds it; where a reply would be \
cut, split the script into as few files as fit, each a module work/main.py imports, and for no other reason; after a \
cut, write the rest one file a reply.

{api}

Unless the task says otherwise:
{conventions}

Every script must:
- read its inputs from the true paths given under "Paths": the paths in the task's text are those of another machine;
- write every output file the task asks for, with exactly the names, formats, keys and columns it gives, directly \
into the output directory given under "Paths", and nothing anywhere else but work/ and /tmp;
- use only the Python standard library, qf and these installed packages: {libraries}; no network, no pip, no \
subprocess;
- be deterministic: seed any random generator the task does not seed itself;
- write JSON with plain Python numbers (convert numpy values with float() or int());
- never copy the task's text into an output file, and never write a file named reward.json; `<canary>` in the task \
or the data stands for a watermark string: never write it, or an input field that holds it, into any output file;
- end within its run's limit of {run:.0f} seconds, printing a short line a step at most.

{example}

{library}

{index}"""

TOOL_LINES = {     # lane putaway1: the line of each tool a request may lack, as each form's paragraph says it
    "tools": {"inspect": " `inspect` describes an input data file.",
              "lib": " `lib` shows the library's documentation."},
    "fenced": {"inspect": " `inspect <path> rows=<n>` describes an input data file;",
               "lib": " `lib name=<function>` or `lib shelf=<module>` shows the library's documentation;"}}
TOOLS_OPEN = ("You act through five tools.", "You act through these tools.")    # the tools form's opening, and where
#                                                                                 a tool is left unsaid
FINISH_LINE = "`finish` ends the task, and nothing else does."  # the put-away sentence follows it, in either form
STEP_ONE = ("Start by writing code: `inspect` a file only where its description leaves you short.",
            "Start by writing code.")   # How to work's first step, and where `inspect` is withheld or put away
PUT_AWAY = {    # the sentence the tools' paragraph says from the waste state's declaration until a file is saved (lane
    #             redirect1's `REDIRECT_ASK` in its style), by the tools put away
    ("inspect", "lib"): ("`inspect` and `lib` are put away until a file is written: write work/main.py now, whole, "
                         "from what you hold."),
    ("lib",): "`lib` is put away until a file is written: write work/main.py now, whole, from what you hold.",
    ("inspect",): ("`inspect` is put away until a file is written: write work/main.py now, whole, from what you "
                   "hold.")}     # lane libnever1: `lib` withheld all the way, `inspect` alone put away


def told(system: str, mode: str, unsaid: tuple[str, ...] = (), away: tuple[str, ...] = (), defs: bool = False,
         clean: bool = False) -> str:
    """Lane putaway1 (the owner's in739): the loop's system prompt `system` (`N11.system`'s, in `mode`'s form) as a
    request lacking tools says them: each tool of `unsaid` (withheld: `INSPECT=described` or `never`) left out of the
    tools' paragraph (`TOOL_LINES`), the tools form no longer counting them (`TOOLS_OPEN`); those of `away` (put away:
    `REDIRECT=write`'s waste state) said put away in one sentence after `finish`'s line (`PUT_AWAY`); How to work's
    first step without its `inspect` clause where `inspect` is either (`STEP_ONE`). `system` itself where neither
    names a tool: nothing else in it changes. `defs`: the paragraph is lane fixdef1's (`PART_DEFS`); `clean`: as lane
    clean1's word says it (`prompts.clean_tools`), the put-away sentence after its `finish` line."""
    if not unsaid and not away:
        return system
    part = PART_DEFS[mode] if defs else TOOLS_PART if mode == "tools" else FENCED_PART
    part = prompts.clean_tools(part) if clean else part
    finish = prompts.CLEAN_COMPLETION[1] if prompts.CLEAN_COMPLETION[1] in part else FINISH_LINE
    said = part
    for name in (*unsaid, *away):
        said = said.replace(TOOL_LINES[mode][name], "", 1)
    if unsaid and mode == "tools":
        said = said.replace(TOOLS_OPEN[0], TOOLS_OPEN[1], 1)
    if away:
        said = said.replace(finish, f"{finish} {PUT_AWAY[away]}", 1)
    system = system.replace(part, said, 1)
    return system.replace(STEP_ONE[0], STEP_ONE[1], 1) if "inspect" in (*unsaid, *away) else system


LIBRARY_USE = library.USE.format(cards="the cards below")   # lane lib4: when to call it, how to read a card
LOOKUP_CLAUSE = ", or `lib` on a name in the index"         # lane libnever1: its clause naming the lookup, which
LIBRARY_USE_NEVER = LIBRARY_USE.replace(LOOKUP_CLAUSE, "", 1)   # `LIB=never` leaves out
CONVENTIONS = ("Before computing, apply every convention the task states: its day count, annualisation, ddof, "
               "compounding, sign, lag and rounding rules.")
CLAUSE_SAID = {     # lane clauses1 (`CLAUSE`): each word's sentence, a paragraph of its own closing the first message
    "fit": ("Before using a library function, compare its model and calculation steps with the task's stated steps; "
            "where they differ, implement the task's steps."),
    "card": ("For each library call, map its returned quantity to the requested quantity and apply the card's lag, "
             "sign and unit rules in the code."),
    "reuse": ("Where a documented library function matches the task's method and conventions, call it instead of "
              "rewriting that calculation.")}
SETTLE = ("settle every convention the task states (day counts, annualisation, ddof, compounding, signs, rounding) "
          "before computing anything.")     # How to work's first step, its sentence on conventions
SETTLE_STATED = (   # lane conventions1 (`STATED=on`): that sentence, each convention the task pins stated once
    "settle every convention the task states before computing anything. State each convention the task pins once, "
    "at the top of the script, as a named constant or function, and use that name wherever it applies: ddof, the "
    "day count, annualisation and its clock, drawdown sign and start, simple or log returns and wealth, turnover "
    "basis, lag, fee basis, the stage at which values are rounded, the random stream and the path slice. Where the "
    "task is silent on one the script needs, say which you chose, once, in a comment beside it.")
TIMING_LINES = (    # lane n11e: textbook conventions, one line each (docs/research/intermittent.md, fix 8)
    "- a signal computed from bar t's close trades no earlier than bar t+1's open;",
    "- a model's training window for predicting row t ends at row t-1: it never holds row t;",
    "- align predictions with their targets by index (date or key), never by position after a shift;")
MISSING_LINES = (   # today's blanket missing-row lines, which `STATED=on` says operation by operation instead
    "- before a correlation, drop the rows where either side is NaN;",
    "- \"valid observations\" are the rows left after dropping NaN, never values filled in;")
CODE_LINES = (
    "- define each formula once, as a function, and call it wherever it is needed;",
    "- prefer an installed library's function (numpy, scipy, statsmodels, arch, qf) to a hand-written algorithm.")
CONVENTION_LINES = (*TIMING_LINES, *MISSING_LINES, *CODE_LINES)
ELIGIBILITY_LINES = (   # lane conventions1 (`STATED=on`): row eligibility and clocks, operation by operation
    #                     (docs/research/conventions-1017.md item 4; runs/lanes/codex-checks-{a,b,c}.md)
    "- a row enters a calculation when every input that calculation needs is present: decide it calculation by "
    "calculation, so a row one calculation leaves out stays in the others, and fill or drop a missing value only "
    "where the task says, at the step it says;",
    "- where the task says how duplicates are resolved (the one kept, the order or sequence that decides it), resolve "
    "them that way, at that step, before any calculation reads the rows;",
    "- a grid or window the task states in time (a sample every k minutes, a calendar span, month ends) runs by the "
    "clock, never by counting observations, and each date keeps the role the task gives it (formed, estimated, "
    "realised);",
    "- warm-up and the first return as the task says: a value needing a full window starts at the first row that has "
    "one, and where the task's ledger starts at the initial capital the first return is zero; include or exclude "
    "that row from each statistic as the task says;",
    "- a threshold, cutoff or tail is strict or inclusive as the task words it (\"above\", \"beyond\", \"after\" "
    "strict; \"at least\", \"at or above\", \"at or below\", \"at or after\", \"on or before\" inclusive), the same "
    "way wherever it is used;")
STATED_LINES = (*TIMING_LINES, *ELIGIBILITY_LINES, *CODE_LINES)     # the conventions under `STATED=on`
API_DISTRIBUTIONS = ("numpy", "pandas", "scipy", "statsmodels", "arch", "scikit-learn")
API_NOTES = (   # lane n11e: the calls the runtime's versions removed or changed (E039's unrepaired crashes), by major
    ("numpy", (2, 0), "numpy 2: `np.trapezoid` (there is no `np.trapz`), `np.prod` (no `np.product`), `np.nan` (no "
                 "`np.NaN`), `np.isin` (no `np.in1d`); a `np.random.RandomState` draws Student-t by `.standard_t(df)` "
                 "(it has no `.t`)."),
    ("scipy", (1, 14), "scipy: an `integrate.quad` integrand returns a Python float (`float(...)`, never a one-element "
                 "array); `integrate.simpson` and `integrate.cumulative_trapezoid` (no `simps`, `cumtrapz`); "
                 "`linalg.pinv` (no `pinv2`); `optimize.minimize(..., method=\"L-BFGS-B\").hess_inv` is an operator: "
                 "`.todense()` before indexing it."),
    ("pandas", (3, 0), "pandas 3: copy-on-write, so a chained assignment (`df[\"a\"][i] = x`) changes nothing: "
                  "`df.loc[i, \"a\"] = x`; `.iloc[k]` for a position (`s[k]` on a labelled index is a label); "
                  "`.ffill()` and `.bfill()` (no `fillna(method=...)`); `pd.concat` (no `DataFrame.append`); "
                  "frequencies `\"ME\"` and `\"h\"` (not `\"M\"`, `\"H\"`); text columns are dtype `str`."),
    ("pandas", (2, 0), "pandas 2: `.iloc[k]` for a position; `pd.concat` (no `DataFrame.append`)."))
SAME_FRAME = ("same failing line as the previous run, {where}: `{source}`; the fix left the line that fails as it was. "
              "Change that line or the call it makes (a different fix), and check the call against the runtime's "
              "versions (the system prompt's runtime paragraph).")
REVIEW_ASK = {   # lane loop1: the review's ask, generic (no quantity a unit names)
    "tools": ("review: work/main.py ran clean, and the task does not end before it is checked. Check the script "
              "(work/main.py, your first script, saved there) against each convention the task states (day count, "
              "annualisation, ddof, compounding, signs, rounding, windows, alignment, sorting), and each value above "
              "against what the task says of it. Where every one holds, call finish with a reason naming them "
              "(finish(reason=\"<each convention and value checked>\")); where one does not, write that one file "
              "again, whole, and run work/main.py."),
    "fenced": ("review: work/main.py ran clean, and the task does not end before it is checked. Check the script "
               "(work/main.py, your first script, saved there) against each convention the task states (day count, "
               "annualisation, ddof, compounding, signs, rounding, windows, alignment, sorting), and each value above "
               "against what the task says of it. Where every one holds, write the line `finish reason=<each "
               "convention and value checked>`; where one does not, write that one file again, whole, as its block, "
               "and the line `run work/main.py`.")}
REVIEW_LOOP_ASK = {mode: ask.replace("your first script", "your current script").replace(
    "work/main.py ran clean", "the output checks pass")
                   for mode, ask in REVIEW_ASK.items()}
FIX_WHOLE = "write that one file again, whole"
FIX_DEFS = "write that one file whole, or the definitions that change alone"
REVIEW_ASK_DEFS = {mode: ask.replace(FIX_WHOLE, FIX_DEFS, 1) for mode, ask in REVIEW_ASK.items()}
REVIEW_LOOP_ASK_DEFS = {mode: ask.replace(FIX_WHOLE, FIX_DEFS, 1) for mode, ask in REVIEW_LOOP_ASK.items()}
CONTINUE_ASK = ("Your reply was cut at the output limit{where}. The script above is kept whole up to its last line. "
                "Reply with the rest of the script only: {start}, through the code that runs it and writes every "
                "output file. One ```python block of code, no prose before or after it and no line already written "
                "above; a reply past the limit is cut again.")
ENTRY = ("entry: the stitched work/main.py defines `{name}` and nothing at its top level calls it, so it ran and "
         "wrote nothing: add `if __name__ == \"__main__\":` and the call that runs it.")
CUT_NOTE = ("Your first reply was cut at the output limit {where}: its complete top-level definitions are kept in "
            "work/draft.py{defines}. The script does not fit one reply: write the rest in as few files as fit, one "
            "file a reply, each a module work/main.py imports, and work/main.py last, importing what it needs and "
            "writing every output file; then run work/main.py.")
CUT_EMPTY = ("Your first reply was cut at the output limit before any complete definition. The script does not fit "
             "one reply: split it into as few files as fit, one file a reply, each a module work/main.py imports, "
             "and work/main.py last, importing them and writing every output file; then run work/main.py.")
FIRSTOUT_LATER = ("each later output file in a reply of its own: a script work/<name>.py with its own "
                  "`if __name__ == \"__main__\":` block that imports what it needs from the files before it and "
                  "writes that file")    # lane firstout1: the shape after the first output, as each text below says it
FIRSTOUT_SAID = {   # lane firstout1 (`FIRSTOUT=on`): each text of the loop's system prompt asking the whole script
    #                 first, and as the switch says it, in either reply form (`N11.system`)
    "2. Write the whole solution as one script, work/main.py, that reads the inputs, computes each quantity (a "
    "quantity the library computes is a call of its function) and writes every output file the task asks for.":
        ("2. Write the solution one output file at a time, in the order the task gives them. Your first script, "
         "work/main.py, reads the inputs, computes the values of the task's first output file (a quantity the "
         "library computes is a call of its function) and writes that file alone; keep its steps in functions and "
         "its run under `if __name__ == \"__main__\":`, so that later scripts can import them. Then write "
         + FIRSTOUT_LATER + "."),
    "3. A write of work/main.py runs it: read each run. Where you split the script into modules, write them first "
    "and main.py last; a module's write runs work/main.py only where main.py imports it and its last run was not "
    "clean: after a clean run, where you change a module, run work/main.py after it in the same reply.":
        ("3. A write of work/main.py runs it, and a write of a later script runs that script: read each run. Where "
         "you split a script into modules, write them first and the script last; a module's write runs work/main.py "
         "only where main.py imports it and its last run was not clean: after a clean run, where you change a "
         "module, run the script that imports it after it in the same reply."),
    "So write work/main.py whole, in one reply, where one reply holds it; where a reply would be cut, split the "
    "script into as few files as fit, each a module work/main.py imports, and for no other reason; after a cut, "
    "write the rest one file a reply.":
        ("So write one output file's script a reply, each file whole; where a reply would still be cut, split that "
         "script into modules it imports, one file a reply; after a cut, write the rest one file a reply."),
    "- write every output file the task asks for, with exactly the names, formats, keys and columns it gives, "
    "directly into the output directory given under \"Paths\", and nothing anywhere else but work/ and /tmp;":
        ("- write each output file it writes with exactly the name, format, keys and columns the task gives, "
         "directly into the output directory given under \"Paths\", and nothing anywhere else but work/ and /tmp;"),
    TOOLS_EXAMPLE:
        ("An example: the first reply calls write(path=\"work/main.py\", text=<the first script: it reads the inputs, "
         "computes each value of the task's first output file (def log_returns(prices): return "
         "np.diff(np.log(prices)), and the like) and writes that file alone>), which is run for you; the next fixes "
         "what the run showed, or calls write(path=\"work/<name>.py\", text=<a script that imports what it needs "
         "from main and writes the next output file>), which is run for you too, or calls finish."),
    FENCED_EXAMPLE:
        ("An example first reply: one block opened by ```python file=work/main.py holding the first script, which "
         "reads the inputs, computes each value of the task's first output file (a function log_returns(prices), "
         "and the like) and writes that file alone, and nothing after it: main.py is run. A later reply: one block "
         "opened by ```python file=work/<name>.py holding a script that imports what it needs from main and writes "
         "the next output file, and nothing after it: it is run.")}
FIRSTOUT_FILES = ("Write them one at a time, in the order the task gives them: your first script, work/main.py, "
                  "reads the inputs and writes {first} alone; each later one follows in a reply of its own. "
                  "The checks will report missing output files until they are written; continue with the later "
                  "scripts.")  # a
#                   paragraph of its own after today's line, which stays whole
FIRSTOUT_UNNAMED = "the first one it describes"     # `FIRSTOUT_FILES`' first file where the task names none
CUT_NOTE_FIRSTOUT = ("Your first reply was cut at the output limit {where}: its complete top-level definitions are "
                     "kept in work/draft.py{defines}. The script does not fit one reply: write it one output file at "
                     "a time, one file a reply: work/main.py first, importing what it needs from work/draft.py and "
                     "writing the task's first output file alone, then " + FIRSTOUT_LATER + ".")
CUT_EMPTY_FIRSTOUT = ("Your first reply was cut at the output limit before any complete definition. The script does "
                      "not fit one reply: write it one output file at a time, one file a reply: work/main.py first, "
                      "reading the inputs and writing the task's first output file alone, then " + FIRSTOUT_LATER
                      + ".")
NO_SCRIPT = "Your first reply held no script. Write the solution as one script, work/main.py, and run it."
CUT_CALL = ("not run: your reply was cut at the output limit inside this call, so what it held was lost and no file "
            f"was written: it does not fit one reply, so split the script {commands.SPLIT}, the first now.")
CUT_FENCED = ("not written: your reply was cut at the output limit before any file block arrived, so no file was "
              f"written. Split the script {commands.SPLIT}; reply with the first file as a block opened by "
              "```python file=work/<name>.py.")
CUT_DRAFT = ("not written: your reply was cut at the output limit inside this write of {named}, so the file was not "
             f"saved; {{kept}}. It does not fit one reply: split it {commands.SPLIT}, the first now.")
CUT_KEPT = "its complete definitions are kept in {draft}{defines}, the cut falling {where}"
CUT_NONE = "nothing in it was complete enough to keep"
CUT_SHORT = (" {n} replies in a row were cut: write one small file now, a module work/main.py imports holding one or "
             "two functions, and the next after it.")
FIRSTOUT_SPLIT = ("into modules the script for that output imports, and that script last; write one file a reply")
CUT_SHORT_FIRSTOUT = (" The replies were cut: write a complete module for the script that writes this output, "
                      "then the script importing it.")
SHORT_STREAK = 3                # cut replies in a row before one small file is asked
CUT_HELD = ("not written: your reply was cut at the output limit inside this write of {named}, so the file was not "
            "saved whole; what arrived is kept up to its last complete line, which ends:\n\n```python\n{ends}\n```\n\n"
            "Reply with the rest of {named} only: every line after that last one, {through}. One ```python block of "
            "code, no prose before or after it and no line already written above; a reply past the limit is cut "
            "again.")         # lane review2, `CUT=continue`: CONTINUE_ASK's shape in the loop
CUT_THROUGH = {True: "through the code that runs it and writes every output file", False: "through the file's end"}
CUT_THROUGH_FIRSTOUT = "through the code that runs it and writes its output"   # lane firstout1's for work/main.py
STITCHED = "the rest joined under what was kept of {named}, the file whole: "     # the stitched write's observation
JUNCTION_SHOWN = "where what was kept meets the rest you sent, as joined:\n{lines}\n{opened}"
JUNCTION = (JUNCTION_SHOWN + "Reply with the rest of {named} again: every line after what was kept, {through}, in one "
            "```python block of code, joined under what was kept as before, at the depth of what was kept; or write "
            "{named} again whole.")
JUNCTION_MARK = "       | --- the rest you sent ---"             # the line between the two sides
NEVER_CLOSED = "The {what} opened at line {line}, in what was kept, is never closed by the rest you sent. "
JOIN_REFUSED = ("write refused: {named} already holds this join of what was kept and the rest you sent, which does not "
                "compile: {error}; nothing was written and nothing ran.")    # lane join2: a failed join sent again
KEPT_BROKEN = ("The error stands in what was kept, above where the rest you sent joins it, so no rest can mend it: "
               "write {named} again whole, from its first line; what was kept is dropped.")   # and said once a file
JOIN_WHY = {    # lane joinask1 (`JOINASK=change`): what failed, the changed ask's first sentence
    "junction": "The rest you sent does not go on where what was kept ends, so the join does not compile: {error}.",
    "cut_again": "What you sent of {named} was cut at the output limit again, so what is left of it does not fit one "
                 "reply.",
    "refused": "The same text came back as before, and it joins what was kept as before: {error}.",
    "recut": "The same text came back as before, and the output limit cut it where it cut it before."}
JOIN_STATEMENT = ("{why} What was kept of {named} ends{inside} with the statement `{last}`, as here:\n\n```python\n"
                  "{ends}\n```\n\n{cut}Reply with the rest of {named} from {start} on, {through}. One ```python block "
                  "of code, no prose before or after it and no line already written above.")     # the statement shape
JOIN_CUT = "The cut fell inside the statement that begins `{line}`, and that statement was not kept. "
JOIN_START = {True: "that statement's first line", False: "the statement after it"}
JOIN_RETRY = "Begin the rest at {start}; rebuild that statement whole before continuing the file. "
JOIN_MODULE = ("{why} What was kept of {named}, through its last complete top-level statement, is saved whole as the "
               "module {module}, which compiles; it ends with `{last}`. Write {named} anew, {opened}: its first line "
               "`from {stem} import *`, then the code after `{last}`, {through}; no line the module holds, and no "
               "prose before or after it.")                                                  # the module shape
JOIN_OPENED = {"fenced": "as one block opened by ```python file={named}", "tools": 'as one write(path="{named}", ...)'}
JOIN_ONE = {"fenced": "in one block opened by ```python file={named}", "tools": "in one write of {named}"}
JOIN_DEFS = ("not written: your reply wrote {named} again whole and was cut at the output limit, so none of it was "
             "saved; {named} stands as it was when it ran, which gave: {gave}. Write the fix as the definitions that "
             "change alone: each function or class of {named} that changes, whole, {one}; every other line of the "
             "file is kept as it stands, so send nothing else of it.")      # rule 3: a whole rewrite after a failing run
JOIN_HEAD = ("{why} No rest of {named} is asked again: {named} now holds {what}, whole and compiling, and it runs as it "
             "stands.")                                                     # rule 2: the junction had its asks
JOIN_HELD = {True: "its last script that ran clean", False: "what was kept of it, to its last complete "
                                                                           "statement"}
JOIN_KEPT = ("not written: {named} stands as it was; this text is neither definitions of it, each function or class "
             "whole, nor {named} whole, so saving it would have replaced the file. Send the definitions that change "
             "alone, {one}, or {named} whole.")    # a fragment for a file asked for definitions, refused once
JOIN_STOP = "The definitions sent again leave {named} unchanged; its file and output stand, and this attempt ends."
JOIN_STAGED = ("not written: {named} stands as it was; this text is not {named} whole, so saving it would have "
               "replaced the file, and it is saved as {staged} beside it.")    # lane join2 of 10-16 (d): staged
JOIN_RAN = ("Write what {named} lacks, or what changes in it, as definitions alone: each function or class whole, and "
            "the `if __name__ == \"__main__\":` block that runs it where it has none, {one}; what the file holds is "
            "kept, and each definition you send replaces its namesake or is added to it.")
UNCLOSED = re.compile(r"'([(\[{])' was never closed")             # Python's words for a bracket left open
UNTERMINATED = "unterminated triple-quoted string literal"         # and for a string
NO_CALL = {"tools": 'no tool call: write work/main.py now, whole (write(path="work/main.py", text=<the whole '
                    "script>)), or the next module where you split it; or call finish where the output is done; only "
                    "finish ends the task.",
           "fenced": "no tool call: write work/main.py now, whole (a block opened by ```python file=work/main.py), or "
                     "the next module where you split it; or a line `finish` where the output is done; only finish "
                     "ends the task."}
FINISH_REFUSED = {"tools": 'finish refused: {n} sanity finding(s) stand; write a fix or call finish again with '
                           'reason="accept: <why>"\n{lines}',
                  "fenced": "finish refused: {n} sanity finding(s) stand; write a fix or write the line `finish "
                            "accept: <why>`\n{lines}"}
UNCHANGED = ("write refused: {named} unchanged from turn {turn}, which gave: {gave}; nothing was written and nothing "
             "ran. Change the file to fix it, or run it as it is: {run}")
HELD = ("write refused: {named} already holds this text; nothing was written and nothing ran. Change the file to fix "
        "it, or run it as it is: {run}")
CYCLE = ("{named}: a cycle among {k} versions: this text is turn {turn}'s again, which gave: {gave}. Before the next "
         "edit, print the failing line's operands (shapes, dtypes, keys) and read them.")
NO_RUN = "(no run: the file was not written whole and compiling)"     # `after_write`'s line after a write not saved
# lane resend1, `RESEND=finish`: the line heading an unchanged write's observation where it is read as the finish; and
#   the run line of one refused, the file unchanged since the turn it matched and, after a clean run, the outputs missing
RESEND_READ = ("{named} is unchanged from turn {turn}, whose run exited clean with every named output present: read as "
               "the finish.")
RESEND_SINCE = "(no run: {named} is unchanged since turn {turn})"
RESEND_MISSING = ("(no run: {named} is unchanged since turn {turn}, whose run exited clean; these named outputs are now "
                  "missing: {missing}; write the file that writes them, or {run})")
CYCLE_RECURRENCES = 3           # returns to an earlier version of a file before the cycle is named
CYCLE_WINDOW = 6                # the file's last versions a cycle is read among
MALFORMED = ("not run: this call is malformed: {why}. Call it again in the form its tool gives: write(path, text), "
             "run(path), inspect(path, rows), lib(name) or lib(shelf), finish().")
FINISH_FAILED = ("finish: {n} check(s) fail:\n{lines}\nFix them and finish again; a second finish ends the task "
                 "whatever the checks say.")
FINISH_FINDINGS = "\nsanity findings, which stand too:\n{lines}"
FINISHFAIL_SAID = ("finish not taken: the last run, {named}, failed, so the task goes on. Fix it so "
                   "that it runs clean, then finish. That run:")   # lane finishfail1: the run's observation after it
FINISHCHECK_SAID = ("finish not taken: the output does not yet hold what the task names, so the task goes on. Fix the "
                    "script so that its output meets each line below, run it, then finish:\n{lines}")  # lane
#                   finishcheck1: the contract's findings, one line each
FINISHCHECK_FAILED = ("finish: the checks fail:\n{lines}\nFix them and finish again; the output must also meet "
                      "the task's contract.")
CARDFAIL_SAID = "The failing call's function, {call}, is documented below:"   # lane cardfail1: the line over its card
REPEAT_ASK = ("same program, same failure: the files under work/ are those of an earlier run that failed this same way, "
              "and run or sent unchanged they fail the same way again. Make a different repair, and say before its "
              "code what it changes: the failing statement itself, the shape or type of the data where it fails "
              "(print it first), the convention of the function it calls, or another method for that step.")
#   lane repeat1, `REPEAT=change`: closing the last observation of a turn that held the unit's second recurrence of a
#   failing program state or a later one (`repeat_ask`); no count stated
FINISH_AGAIN = "fix them and finish again; a second finish ends the task whatever the checks say."   # FINISH_FAILED's
#                                                                   ask in one line: form `state`'s `lean`, `minimal`
LAST_CALL = {   # lane clock1: the last call's ask, where no named output exists and the budget buys one more reply
    "tools": ("last call: no output file the task names exists yet, and the budget buys this one reply alone. Write "
              "work/main.py, whole, with the write tool, so that it writes every output file the task names from what "
              "the files under work/ already compute: each output file in its own try/except, so that one failure "
              "does not stop the others; where nothing computes a value, leave it out rather than invent one, and "
              "still write the file with what is there. Nothing else in this reply: the harness runs work/main.py "
              "with all the time left, and the task ends on its output."),
    "fenced": ("last call: no output file the task names exists yet, and the budget buys this one reply alone. Write "
               "work/main.py, whole, as one block opened by ```python file=work/main.py, so that it writes every "
               "output file the task names from what the files under work/ already compute: each output file in "
               "its own try/except, so that one failure does not stop the others; where nothing computes a value, "
               "leave it out rather than invent one, and still write the file with what is there. Nothing else in "
               "this reply: the harness runs work/main.py with all the time left, and the task ends on its output.")}
LAST_CALL_BYTES = 3_000         # the last call's evidence (the files under work/ and their runs), at most
FORCED = {      # lane clock1: the ask after `NO_CALL_CAP` replies in a row with no call, while two replies remain
    "tools": ("no tool call again: this reply must call the write tool. Write the next file now, whole "
              "(write(path=\"work/<name>.py\", text=<the whole file>)): work/main.py, or the next module where you "
              "split it; a reply with no file now ends the task."),
    "fenced": ("no file again: write the next file now, whole, as a block opened by ```python file=work/<name>.py: "
               "work/main.py, or the next module where you split it; a reply with no file block now ends the task.")}
WRITE_CHOICE = {"type": "function", "function": {"name": "write"}}     # the forced ask's tool_choice
SKETCH_ASK = {  # lane plan4 (the owner's in316): form `sketch`'s closing ask of the loop's first message, in shapes
    "turn": ("Before any code, describe in a short paragraph how you will solve the task, in what order, and which "
             "library functions from {source} you will call where the task names their methods. Write no code and "
             "no list of files in this reply; where you are unsure, say so."),
    "inline": ("Begin your reply with a short paragraph, before any code: how you will solve the task, in what "
               "order, and which library functions from {source} you will call where the task names their methods; "
               "no list of files; where you are unsure, say so. Then, in the same reply, write the code {how}.")}
SKETCH_SOURCES = ("the cards", "the index")     # where the pointer named functions, and where it named none
SKETCH_HOW = {"tools": "with the write tool", "fenced": "as a block opened by ```python file=work/<name>.py"}
SKETCH_CARDS_HEAD = ("Cards of the library's functions your sketch names; use one where it computes what the task "
                     "asks:")       # addendum 2: the next message's cards of the sketch's picks the pointer missed
SKETCH_LEFT = "the cards of {named} did not fit here: `lib name=<fn>` shows each whole"
SKETCH_UNKNOWN = "the library holds no function `{name}` ({near}): do not call it"
SKETCH_NEAREST = 3              # the index's names said back for a `qf.` name it lacks
SKETCH_NEXT = {  # `turn`: the next message after the sketch, its one line before the budget
    "tools": ("Now write the code with the write tool; where the task or a run's result differs from your sketch, "
              "follow them."),
    "fenced": ("Now write the code as a block opened by ```python file=work/<name>.py; where the task or a run's "
               "result differs from your sketch, follow them.")}
NOCALL_ASK = {  # lane cap1, `ask`: the answer to every reply with no call, before the call asked (`NOCALL_NOW`)
    "tools": "no tool call: your reply made no call, so nothing was written or run; prose is not read.",
    "fenced": "no file: your reply held no file block and no line `finish`, so nothing was written or run; prose is "
              "not read."}
NOCALL_NOW = {"tools": "Call it now: {call}.", "fenced": "Write it now: {call}."}
NOCALL_CUT = ("your reply was cut at the output limit inside its call, so what it held was lost and no file was "
              f"written: it does not fit one reply, so split it {commands.SPLIT}.")     # before the ask, after a cut
NOCALL_CALL = {     # what the ask asks: the file the reply's prose named, after a cut the first part, else the next
    "named": {"tools": 'write(path="{named}", text=<the whole file>), the file your reply named',
              "fenced": "{named}, whole, as one block opened by ```python file={named}"},
    "cut": {"tools": 'write(path="work/<name>.py", text=<the whole file>), the first of those files',
            "fenced": "the first of those files, whole, as one block opened by ```python file=work/<name>.py"},
    "next": {"tools": 'write(path="work/main.py", text=<the whole script>), or the next module where you split it; '
                      "or finish where the output is done",
             "fenced": "work/main.py, whole, as one block opened by ```python file=work/main.py, or the next module "
                       "where you split it; or the line `finish` where the output is done"}}
PROSE_FILE = re.compile(r"(?<![\w/.-])(?:\./)?(?:" + re.escape(commands.WORK)
                        + r"/)?([A-Za-z_]\w*\.py)\b(?![\w/])")     # a file a reply's prose names (`prose_files`)
CODE_SPAN = re.compile(r"(`+)(.+?)\1", re.S)   # lane cap1, `named`: a code span of the sketch, where its bare names count
RESERVE_END = ("a run went past the request's reserve, so no request could follow it: it had the time left, and the "
               "unit ends on its output")
NOT_IMPORTED = "; it does not import {named}, so nothing ran {named}: import it in {main} where it is needed"
SELF_RUN = (" ({named}, by its own `if __name__ == \"__main__\":` block: {main} does not import it, so {main} did not "
            "run)")
NO_WORK = re.compile(r"No module named '" + re.escape(commands.WORK) + r"[.']")   # a package-style import's failure
SHORT_FIRST = {     # lane pace1: the first message's close where the clock at the start buys few requests, in shapes
    "tools": ("The clock is short: make the whole solution your first move. Write work/main.py, whole, with the write "
              "tool, so that it reads the inputs, computes every value and writes every output file the task names; "
              "its write runs it.{cards} Look nothing up first: `lib` and `inspect` only where something you need is "
              "missing."),
    "fenced": ("The clock is short: make the whole solution your first move. Write work/main.py, whole, as one block "
               "opened by ```python file=work/main.py, so that it reads the inputs, computes every value and writes "
               "every output file the task names; it runs after your reply.{cards} Look nothing up first: the lines "
               "`lib` and `inspect` only where something you need is missing.")}
SHORT_CARDS = (" The cards in the system prompt document the library's functions the task names: call each as its "
               "card writes it.")   # where the pointer named functions, whose cards the system prompt carries
SHORT_ASK = {   # lane pace1: the short-clock mode's ask after each turn's observations, by what the last run left
    "write": {"tools": ("the clock is short: write work/main.py now, whole, so that its run writes every output file "
                        "the task names; look nothing up first."),
              "fenced": ("the clock is short: write work/main.py now, whole, as one block opened by ```python "
                         "file=work/main.py, so that its run writes every output file the task names; look nothing up "
                         "first.")},
    "fix": {"tools": ("the clock is short: fix and run in your next reply: write the failing file again, whole, and "
                      "run work/main.py after it where the file is not work/main.py (a write of work/main.py runs "
                      "it); look nothing up first, and call finish as soon as a run is clean with every output file "
                      "the task names present."),
            "fenced": ("the clock is short: fix and run in your next reply: the failing file again, whole, as its "
                       "block, and the line `run work/main.py` where the file is not work/main.py (a block for "
                       "work/main.py runs it); look nothing up first, and write the line `finish` as soon as a run is "
                       "clean with every output file the task names present.")},
    "finish": {"tools": ("the clock is short: the last run was clean, the checks pass and every output file the task "
                         "names is present: call finish now, unless a value breaks a convention the task states (its "
                         "day count, annualisation, ddof, compounding, signs, rounding) or its range; only then write "
                         "the fix, for that value alone."),
               "fenced": ("the clock is short: the last run was clean, the checks pass and every output file the task "
                          "names is present: write the line `finish` now, unless a value breaks a convention the task "
                          "states (its day count, annualisation, ddof, compounding, signs, rounding) or its range; "
                          "only then write the fix, for that value alone.")}}
REDIRECT_WHY = {    # lane redirect1: the demand's first clause, by whether every input file is described (`inspect`
    #                 put away with `lib`) and whether the system prompt carries cards
    (True, True): ("Every input file is described in the first message and the system prompt holds the cards of the "
                   "functions the task needs, so `inspect` and `lib` are put away until a file is written."),
    (True, False): ("Every input file is described in the first message, so `inspect` and `lib` are put away until a "
                    "file is written."),
    (False, True): ("The system prompt holds the cards of the functions the task needs, so `lib` is put away until a "
                    "file is written."),
    (False, False): "`lib` is put away until a file is written."}
REDIRECT_ASK = {    # the demand closing each observation from the declaration until a file is saved, in shapes
    "tools": ("{why} Write work/main.py now, whole, from what you hold: it reads the inputs, computes every value and "
              "writes every output file the task names, and its write runs it. Where it would not fit one reply, "
              f"split it {commands.SPLIT}."),
    "fenced": ("{why} Write work/main.py now, whole, as one block opened by ```python file=work/main.py, from what you "
               "hold: it reads the inputs, computes every value and writes every output file the task names, and it "
               f"runs after your reply. Where it would not fit one reply, split it {commands.SPLIT}.")}
REDIRECT_ASK_FIRSTOUT = {   # lane firstout1 (`FIRSTOUT=on`): the demand asking the task's first output file alone
    "tools": ("{why} Write work/main.py now, whole, from what you hold: it reads the inputs, computes the values of "
              "the task's first output file and writes that file alone, and its write runs it; each later output "
              "file follows in a reply of its own."),
    "fenced": ("{why} Write work/main.py now, whole, as one block opened by ```python file=work/main.py, from what "
               "you hold: it reads the inputs, computes the values of the task's first output file and writes that "
               "file alone, and it runs after your reply; each later output file follows in a reply of its own.")}
REDIRECT_HELD = {   # the one-line answer to an `inspect` or `lib` asked while it is put away, by the same two
    (True, True): ("not run: `inspect` and `lib` are put away until a file is written; the first message describes "
                   "every input file and the system prompt holds the cards. Write work/main.py now, whole."),
    (True, False): ("not run: `inspect` and `lib` are put away until a file is written; the first message describes "
                    "every input file. Write work/main.py now, whole."),
    (False, True): ("not run: `lib` is put away until a file is written; the system prompt holds the cards. Write "
                    "work/main.py now, whole."),
    (False, False): "not run: `lib` is put away until a file is written. Write work/main.py now, whole."}
LIB_UNSAID = (  # lane libnever1: each text of the library's part of the system prompt naming the lookup, and as
    #             `LIB=never` says it: the scored index's head, the selection call's section's head, the index's last
    #             resort's tail (`library_unsaid`); a cut card's last line is `fallback.CUT_SAID`
    (library.SCORED_HEAD, library.SCORED_HEAD.replace(" after `others:` (`lib name=<function>` shows any one's card):",
                                                      " after `others`:", 1)),
    (selection.SHELVES_HEAD, selection.SHELVES_HEAD.replace(
        "; `lib name=<function>` shows a function's card and `lib shelf=<module>` a shelf's, these shelves or the "
        "others:", ", then the other shelves:", 1)),
    ("\n[... the rest by `lib shelf=<module>`]", "\n[...]"))
POINTER_LOOKUP = "; `lib name=<fn>` shows its card whole"    # the pointer's line's last clause, left out under `never`
LIB_ASKED = {   # lane libnever1: the one-line answer to a lookup still asked under `never`, by whether the system
    #             prompt carries cards
    True: ("not run: the library lookup is not offered; the cards of the functions the task needs are above, in the "
           "system prompt."),
    False: "not run: the library lookup is not offered; the system prompt's index above names the library's functions."}
LOOKS_SAID = {  # lane libnever1: `SHORT_FIRST`'s last sentence by the lookups a request offers
    "tools": {("lib", "inspect"): " Look nothing up first: `lib` and `inspect` only where something you need is "
                                  "missing.",
              ("inspect",): " Look nothing up first: `inspect` only where something you need is missing.",
              (): " Look nothing up first."},
    "fenced": {("lib", "inspect"): " Look nothing up first: the lines `lib` and `inspect` only where something you "
                                   "need is missing.",
               ("inspect",): " Look nothing up first: the line `inspect` only where something you need is missing.",
               (): " Look nothing up first."}}
MALFORMED_FORMS = {"inspect": " inspect(path, rows),", "lib": " lib(name) or lib(shelf),"}  # lane libnever1: each
#                                                     withheld tool's form, left out of `MALFORMED`'s list under `never`
SKETCH_LEFT_NEVER = "the cards of {named} did not fit here"     # lane libnever1: `SKETCH_LEFT` with no lookup to name
LIB_NEVER_WHY = {   # lane libnever1: `REDIRECT_WHY` under `never`, by whether every input file is described, whether
    #                 the system prompt carries cards and whether `inspect` is put away (`lib` never is: withheld all
    #                 the way); "" where it says neither, the demand alone
    (True, True, True): ("Every input file is described in the first message and the system prompt holds the cards of "
                         "the functions the task needs, so `inspect` is put away until a file is written."),
    (True, False, True): ("Every input file is described in the first message, so `inspect` is put away until a file "
                          "is written."),
    (True, True, False): ("Every input file is described in the first message and the system prompt holds the cards "
                          "of the functions the task needs."),
    (True, False, False): "Every input file is described in the first message.",
    (False, True, False): "The system prompt holds the cards of the functions the task needs.",
    (False, False, False): ""}
LIB_NEVER_HELD = {  # lane libnever1: `REDIRECT_HELD` under `never`, `inspect` alone put away, by the cards
    True: ("not run: `inspect` is put away until a file is written; the first message describes every input file and "
           "the system prompt holds the cards. Write work/main.py now, whole."),
    False: ("not run: `inspect` is put away until a file is written; the first message describes every input file. "
            "Write work/main.py now, whole.")}


MAIN_FILE = f"{commands.WORK}/main.py"


def solution(work: Path | None) -> list[str]:
    """Lane files1 (`single_file`): the files under `work` (work/) the solution at its end holds: work/main.py first,
    then each module there it imports, directly or through another (as `Workspace.reached` follows them), or
    package-style (`from work.x import`, `commands.packaged`); [] where work/main.py is missing."""
    if work is None or not (Path(work) / "main.py").is_file():
        return []
    work = Path(work)
    here = {path.stem for path in work.glob("*.py")} | {path.name for path in work.iterdir() if path.is_dir()}
    seen, queue = set(), ["main"]
    for file, _, bare in commands.packaged(work):
        if file == MAIN_FILE:
            words = bare.replace(",", " ").split()
            queue += [word.split(".")[0] for word in (words[1:2] if words[0] == "from" else words[1:])]
    while queue:
        name = queue.pop()
        if name in seen or name not in here:
            continue
        seen.add(name)
        queue += commands.imported(work / f"{name}.py") if (work / f"{name}.py").is_file() else []
    return [MAIN_FILE, *sorted(f"{commands.WORK}/{name}" + (".py" if (work / f"{name}.py").is_file() else "")
                               for name in seen - {"main"})]


QF_NAMED = re.compile(r"(?<![\w.])qf(?:\.[A-Za-z_]\w*)+")      # a `qf.` name
IDENTIFIER = re.compile(r"(?<![\w.])[A-Za-z_]\w*")              # a bare name: never another module's (`np.std`)


def sketch_names(text: str, lib: library.Library | None, named: bool = False) -> tuple[list[library.Function],
                                                                                       list[str]]:
    """The index's functions a sketch names (lane plan4's addendum 2, the owner's in330), by an exact lookup
    (`Library.find`), nothing fuzzy: each `qf.<name>` and `qf.<shelf>.<name>`, and each
    bare index name, written as code or as a word, in the order first named, once each; and the `qf.` names the index
    lacks (a shelf's name alone is none), the only names a text holds that claim the library: a bare word the index
    lacks is any word. `named` (lane cap1's `SKETCH_PICKS=named`): a bare index name counts only inside a code span
    (`CODE_SPAN`), so a word of the prose that is also a function's name (`std`, `mean`) is no pick."""
    if lib is None or not text:
        return [], []
    names = {function.name: function for function in lib.functions}
    spans = [match.span(2) for match in CODE_SPAN.finditer(text)] if named else None
    said = sorted([(match.start(), match.group(0)) for match in QF_NAMED.finditer(text)]
                  + [(match.start(), match.group(0)) for match in IDENTIFIER.finditer(text) if match.group(0) in names
                     and (spans is None or any(start <= match.start() < end for start, end in spans))])
    found: dict[str, library.Function] = {}
    unknown: dict[str, None] = {}
    for _, name in said:
        function = lib.find(name) if name.startswith(f"{library.PACKAGE}.") else names[name]
        if function is not None:
            found.setdefault(function.name, function)
        elif lib.shelf(name) is None:
            unknown.setdefault(name, None)
    return list(found.values()), list(unknown)


def prose_files(text: str) -> list[str]:
    """Lane cap1: the `.py` files a reply's prose names, as work/<name>.py, in the order first named, once each; a
    path under another directory, or a standard library module's name (which `write` refuses), is none."""
    found: list[str] = []
    for match in PROSE_FILE.finditer(runner.THINK.sub("", text or "")):
        named = f"{commands.WORK}/{match.group(1)}"
        if named not in found and not commands.shadows(named):
            found.append(named)
    return found


def pointer_line(found: list[tuple[library.Function, str]], lookup: bool = True) -> str:
    """The one exact-name pointer: the library's functions the instruction names by an alias, each to be called for
    its quantity (lane lib4); the lookup that shows a card whole named where it is offered (`lookup`; lane
    libnever1's `LIB=never` withholds it, `POINTER_LOOKUP`)."""
    if not found:
        return ""
    named = ", ".join(f"`{function.call}` for {alias}" for function, alias in found)
    return f"the library has {named}: call each for its quantity rather than writing its formula" \
           + (POINTER_LOOKUP if lookup else "")


def library_unsaid(index: str) -> str:
    """Lane libnever1 (`LIB=never`): the system prompt's index, or the selection call's section in its place, each of
    its texts naming the lookup said as `LIB_UNSAID` says it (the scored index's head, the section's head, the tail of
    an index cut at its bound); the rest as it came."""
    for today, said in LIB_UNSAID:
        index = index.replace(today, said, 1)
    return index


def cards_unsaid(cards: str) -> str:
    """Lane libnever1 (`LIB=never`): cards whose cut ones end on `[cut to fit]`, the lookup they name left out (lane
    fallback1's `fallback.CUT_SAID`)."""
    return fallback.CUT.sub(fallback.CUT_SAID, cards)


def textual(messages: list[dict]) -> list[dict]:
    """A conversation's native tool turns as the fenced form writes them: an assistant turn's calls after its text,
    each a `[call name] arguments` line (a write's file as its block), and the tool results that follow it one user
    turn; any other turn as it is."""
    out: list[dict] = []
    for message in messages:
        if message["role"] == "assistant" and message.get("tool_calls"):
            parts = [message.get("content") or ""]
            for made in message["tool_calls"]:
                function = made.get("function") or {}
                try:
                    arguments = json.loads(function.get("arguments") or "{}")
                except ValueError:
                    arguments = None
                if function.get("name") == "write" and isinstance(arguments, dict) and "text" in arguments:
                    parts.append(f"```python file={arguments.get('path')}\n{str(arguments['text']).rstrip()}\n```")
                else:
                    parts.append(f"[call {function.get('name')}] {function.get('arguments')}")
            out.append({"role": "assistant", "content": "\n".join(part for part in parts if part)})
        elif message["role"] == "tool":
            said = f"[{message.get('tool_call_id', '')}]\n{message.get('content') or ''}"
            if out and out[-1]["role"] == "user" and out[-1].get("from_tools"):
                out[-1]["content"] += "\n\n" + said
            else:
                out.append({"role": "user", "content": said, "from_tools": True})
        else:
            out.append(dict(message))
    return [{key: value for key, value in message.items() if key != "from_tools"} for message in out]


def by_kind(counted: Counter) -> str:
    """Counts by kind as a key's value: `kind:n` pairs, the kinds in order, `-` where none."""
    return ",".join(f"{kind}:{n}" for kind, n in sorted(counted.items()) if n) or "-"


def first_line(text: str) -> str:
    return (text or "").strip().split("\n", 1)[0]


def text_write(call: Call) -> bool:
    """Lane fencedpath1: a `write` of a file that is no `.py` file (a `.txt`, `.json` or `.csv` a block named), which
    never runs: in a fenced reply, the run after its writes and finishrun1's set it aside (E182's script never ran
    behind a requirements block)."""
    return call.name == "write" and not str(call.arguments.get("path", "")).endswith(".py")


def logged(text: str) -> str:
    """A message for the log: any line carrying a GUID-shaped token dropped."""
    return canary.strip(text)[0]


def asked(said: str, short: str, bound: int, tail: str) -> str:
    """A cut call's observation within `bound` bytes before `tail`: `said` and the `SHORT_STREAK` ask `short` on one
    line where they fit; else the ask kept whole after `said` cut, with the tail (a long draft's names and path cut
    the ask away before)."""
    if short and commands.size(said + short + ("\n" + tail if tail else "")) > bound:
        return fit(said, bound, short.strip() + ("\n" + tail if tail else ""))
    return fit(said + short, bound, tail)


OPENS = re.compile(r"^(?:async\s+def|def|class)\s+(\w+)")      # a top-level line opening a definition


def draft(code: str) -> tuple[str, list[str], str]:
    """A cut script's complete top-level definitions (imports, functions, classes, plain assignments), its last
    definition left out where the cut may have fallen in it (no top-level line follows it); the names it defines and
    where the cap cut (the definition the last top-level line after them opens, where one does)."""
    complete = runner.complete_lines(code)
    kept = runner.parsing_prefix(complete)
    tree = runner.parse(kept) if kept else None
    if tree is None:
        return "", [], "before any complete statement"
    lines = kept.split("\n")
    nodes = list(tree.body)
    later = [line for line in complete[len(kept):].split("\n") if line.strip() and not line[0].isspace()
             and not line.startswith("#")]
    opened = OPENS.match(later[-1]) if later else None
    where = f"inside `{opened.group(1)}`" if opened else "after the last complete statement" if later else \
        "at the script's end"
    if not later and nodes and isinstance(nodes[-1], runner.DEFINITIONS):
        last = nodes.pop()
        where = f"inside `{last.name}` (left out of the draft)"
    keep = (ast.Import, ast.ImportFrom, *runner.DEFINITIONS, ast.Assign, ast.AnnAssign)
    chunks, names = [], []
    for node in nodes:
        if not isinstance(node, keep):
            continue
        start = min([node.lineno, *(d.lineno for d in getattr(node, "decorator_list", []))])
        chunks.append("\n".join(lines[start - 1:node.end_lineno]))
        if isinstance(node, runner.DEFINITIONS):
            names.append(node.name)
    text = "\n\n".join(chunks).strip("\n")
    return (text + "\n" if text else ""), names, where


def resume(code: str) -> tuple[str, str]:
    """Lane n11e: a cut script kept up to its last complete top-level statement before the one the cut fell in (its
    longest compiling prefix of whole lines, less its last top-level definition where no top-level line follows it:
    the cut may have fallen in it), and the definition the cut fell in ("" where it fell between statements)."""
    complete = runner.complete_lines(code)
    prefix = runner.parsing_prefix(complete)
    tree = runner.parse(prefix) if prefix else None
    if tree is None or not tree.body:
        return "", ""
    later = [line for line in complete[len(prefix):].split("\n") if line.strip() and not line[0].isspace()
             and not line.startswith("#")]
    opened = OPENS.match(later[-1]) if later else None
    if opened:
        return prefix, opened.group(1)
    last = tree.body[-1]
    if not later and isinstance(last, runner.DEFINITIONS):
        start = min([last.lineno, *(d.lineno for d in getattr(last, "decorator_list", []))])
        kept = "\n".join(prefix.split("\n")[:start - 1]).rstrip("\n")
        return (kept + "\n" if kept.strip() else ""), last.name
    return prefix, ""


def opened_fence(text: str, closed: bool = False) -> str:
    """A write's text with the fence the House opened it with dropped (its first line opening a block) and, where
    `closed`, the one closing it (its last such line); a cut text keeps its last line, the cut one."""
    lines = (text or "").split("\n")
    if lines and lines[0].strip().startswith("```"):
        lines = lines[1:]
    if closed:
        while lines and not lines[-1].strip():
            lines.pop()
        if lines and lines[-1].strip() and set(lines[-1].strip()) == {"`"}:
            lines.pop()
    return "\n".join(lines)


def complete_head(code: str) -> str:
    """Keep newline-terminated physical lines, including those inside unfinished Python constructs."""
    head, mark, _ = code.rpartition("\n")
    return head + mark


def unfinished(head: str) -> bool:
    """Lane join2: `head` does not compile only for its end, a statement left open there (a bracket, a string, a
    block's header or a clause still due), which more lines could finish (`codeop`); False where it compiles or a
    line above its end is broken."""
    if runner.compiles(head):
        return False
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return codeop.compile_command(head, "<head>", "exec") is None
    except (SyntaxError, ValueError, OverflowError, MemoryError, RecursionError):
        return False


def boundary_head(code: str) -> str:
    """Lane join2, `CUT=continue`: a cut write's head held at a statement boundary: its complete lines
    (`complete_head`) to the end of the last statement they finish, never inside a bracket or a string nor on a
    block's header or a decorator with no body, and the comment and blank lines straight after it; its complete lines
    where they compile, where a line above their end is broken (`unfinished`), or where no statement ends in them."""
    head = complete_head(code)
    if not unfinished(head):
        return head
    end, line = 0, []
    try:
        for token in tokenize.generate_tokens(io.StringIO(head).readline):
            if token.type == tokenize.NEWLINE:
                if line and not (line[-1].string == ":" or line[0].string == "@"):
                    end = token.start[0]
                line = []
            elif token.type not in (tokenize.NL, tokenize.COMMENT, tokenize.INDENT, tokenize.DEDENT,
                                    tokenize.ENCODING, tokenize.ENDMARKER):
                line.append(token)
    except (tokenize.TokenError, SyntaxError):
        pass
    lines = head.split("\n")
    while end and end < len(lines) and (not lines[end].strip() or lines[end].lstrip().startswith("#")):
        end += 1
    return "".join(text + "\n" for text in lines[:end]) if 0 < end < len(lines) else head


def quoted(line: str) -> str:
    """A line an ask quotes in a code span: its backquotes made plain quotes."""
    return line.replace("`", "'")


def whole_again(here: Path, text: str) -> bool:
    """Lane joinask1: whether `text` reads as the file `here` written whole again: its first lines are the file's
    (`resent_lines`), or it defines and imports every name the file does (`rewritten`); True where the file is gone."""
    try:
        old = here.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return True
    least = min(RESENT_LINES, len(normalised(old)))
    return bool(least and resent_lines(old, text)[0] >= least) or rewritten(old, text)


def appended(here: Path, text: str) -> str | None:
    """Lane joinask1 (`JOIN_RAN`'s write, over what was kept): `text` put after the file `here` holds where it is no
    rewrite of it (its first lines are not the file's, `resent_lines`, nor does it define every name the file does,
    `rewritten`) and the two compile together; else None."""
    try:
        old = here.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    if not text.strip() or whole_again(here, text):
        return None
    joined = old.rstrip("\n") + "\n\n\n" + text.lstrip("\n")
    return joined if runner.compiles(joined) else None


def rewritten(head: str, code: str) -> bool:
    """Lane review2: a reply asked for a head's rest that wrote the file again whole instead: it compiles by itself and
    defines and imports at its top level every name the head's compiling prefix does, a definition among them (a rest
    never repeats a line already written)."""
    kept, tree = runner.parse(runner.parsing_prefix(head)), runner.parse(code)
    if kept is None or tree is None or not runner.compiles(code):
        return False

    def names(module: ast.Module) -> tuple[set[str], set[str]]:
        return ({node.name for node in module.body if isinstance(node, runner.DEFINITIONS)},
                {alias.asname or alias.name for node in module.body if isinstance(node, (ast.Import, ast.ImportFrom))
                 for alias in node.names})

    (defined_, imported), (redefined, reimported) = names(kept), names(tree)
    return bool(defined_) and defined_ <= redefined and imported <= reimported


def stitched(head: str, code: str, cut: bool) -> tuple[str, str, str]:
    """A continuation's join as `N11.resumed` makes it (lanes review2 and join1): the join, how it was read, and the
    rest as stitched (its complete lines where the cap cut it again, `cut`, else the code newline-ended): the head
    alone where a cut rest holds no complete line, the code alone where it wrote the file again whole (`rewritten`),
    else `runner.stitch` under the head, the head's end the rest repeats taken once."""
    rest = complete_head(code) if cut else code.rstrip() + "\n"
    if cut and not rest.strip():
        return head, "nothing added", rest
    if not cut and rewritten(head, code):
        return code, "written again whole", rest
    joined, how = runner.stitch(head, f"```python\n{rest}```\n", overlap=True, again=cut)
    return joined, how, rest


def normalised(text: str) -> list[tuple[str, int]]:
    """Lane redirect1: `text`'s non-blank lines, whitespace collapsed, each with the offset just past it."""
    lines, at = [], 0
    for line in text.splitlines(keepends=True):
        at += len(line)
        if line.strip():
            lines.append((" ".join(line.split()), at))
    return lines


def resent_lines(head: str, code: str) -> tuple[int, int]:
    """How many of `code`'s first non-blank lines repeat `head`'s first ones, whitespace normalised, and the offset in
    `code` just past the last of them."""
    matched, at = 0, 0
    for (kept, _), (said, end) in zip(normalised(head), normalised(code)):
        if kept != said:
            break
        matched, at = matched + 1, end
    return matched, at


def continuation_read(head: str, code: str | None) -> str:
    """Lane redirect1: what a reply to a continuation ask (`CUT_HELD`) held: `prose` (no code), `resent` (the file
    again whole, `rewritten`, or its first lines repeating the kept head's, `RESENT_LINES` of them or the whole head
    where shorter) or `tail`."""
    if code is None:
        return "prose"
    matched, _ = resent_lines(head, code)
    least = min(RESENT_LINES, len(normalised(head)))
    return "resent" if (least and matched >= least) or rewritten(head, code) else "tail"


def past_head(head: str, code: str) -> str:
    """Lane redirect1 (`REDIRECT=write`): a continuation that re-sent the kept head whole, what follows the repeat
    ("" where nothing does, which the stitch adds nothing from); a re-send that left the head part way, as it came
    (the stitch reads it as a restart)."""
    matched, at = resent_lines(head, code)
    return code[at:] if matched == len(normalised(head)) else code


def ends(head: str) -> str:
    """Lane review2: a kept head's last lines as `CUT_HELD` quotes them, as written: at most `HEAD_LINES` that are
    not blank (the blank ones between them kept) within `HEAD_BYTES`, a last line longer than that by its end."""
    lines = head.rstrip("\n").split("\n")
    while lines and not lines[-1].strip():
        lines.pop()
    picked, shown = [], 0
    for line in reversed(lines):
        if line.strip():
            if shown == HEAD_LINES or (picked and commands.size("\n".join([line, *picked])) > HEAD_BYTES):
                break
            shown += 1
        picked.insert(0, line)
    while picked and not picked[0].strip():
        picked.pop(0)
    text = "\n".join(picked)
    if commands.size(text) > HEAD_BYTES:
        text = text.encode("utf-8")[-HEAD_BYTES:].decode("utf-8", errors="ignore")
    return text


def junction(joined: str, start: int, mark: int | None) -> str:
    """Lane join1: where a join meets, as `JUNCTION` shows it: `joined`'s lines before line `start` (from 1, the rest's
    first) and from it, on each side at most `HEAD_LINES` that are not blank (the blank ones between them kept) within
    `HEAD_BYTES`, numbered as a write's syntax error shows its lines (`commands.around`), line `mark` marked, and
    `JUNCTION_MARK` between the sides."""
    lines = joined.split("\n")

    def side(numbers: range) -> list[str]:
        picked, shown, size = [], 0, 0
        for number in numbers:
            line = lines[number - 1]
            if line.strip():
                if shown == HEAD_LINES:
                    break
                shown += 1
            row = f"{'>' if number == mark else ' '} {number:4d} | {line[:commands.SOURCE_CHARS]}"
            if picked and size + commands.size(row) > HEAD_BYTES:
                break
            picked.append(row)
            size += commands.size(row) + 1
        while picked and not picked[-1].split("|", 1)[1].strip():
            picked.pop()
        return picked

    before = side(range(min(start, len(lines) + 1) - 1, 0, -1))[::-1]
    return "\n".join([*before, JUNCTION_MARK, *side(range(start, len(lines)))])


def never_closed(joined: str, start: int) -> str:
    """Lane join1: `NEVER_CLOSED` where `joined` does not compile for a bracket or a triple-quoted string opened before
    line `start` (what was kept) and never closed; "" otherwise."""
    try:
        compile(joined, "<join>", "exec", dont_inherit=True)
    except SyntaxError as exc:
        found = UNCLOSED.fullmatch(exc.msg or "")
        what = f"`{found.group(1)}`" if found else "string" if (exc.msg or "").startswith(UNTERMINATED) else ""
        if what and exc.lineno and exc.lineno < start:
            return NEVER_CLOSED.format(what=what, line=exc.lineno)
    except (ValueError, MemoryError, RecursionError):
        pass
    return ""


def join_unlocated(text: str) -> str:
    """The changed ask anchors source by its statement, without diagnostic line numbers."""
    text = re.sub(r"(?: at|,? (?:detected|opened) at)? line \d+", "", text)
    text = re.sub(r"^([ >])\s*\d+ \|", r"\1 |", text, flags=re.M)
    return text.replace(" (detected)", "")


def compile_error(text: str) -> tuple[str, int | None]:
    """Lane join2: `text`'s compile error as a write's observation names it (`commands.Workspace.write`) and its line;
    ("", None) where it compiles."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            compile(text, "<join>", "exec", dont_inherit=True)
    except SyntaxError as exc:
        return f"{type(exc).__name__} at line {exc.lineno}: {exc.msg}", exc.lineno
    except (ValueError, MemoryError, RecursionError) as exc:
        return type(exc).__name__, None
    return "", None


def defined(text: str) -> list[str]:
    """Lane clock1: a file's top-level functions, each with its arguments' names, and its classes, in order (the last
    call's evidence); [] where it does not parse."""
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError):
        return []
    found = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            args = node.args
            names = [arg.arg for arg in (*args.posonlyargs, *args.args)]
            names += [f"*{args.vararg.arg}"] if args.vararg else ["*"] if args.kwonlyargs else []
            names += [arg.arg for arg in args.kwonlyargs] + ([f"**{args.kwarg.arg}"] if args.kwarg else [])
            found.append(f"{node.name}({', '.join(names)})")
        elif isinstance(node, ast.ClassDef):
            found.append(f"class {node.name}")
    return found


CODE_START = re.compile(r"^(?:\s|def |class |import |from |if |elif |else:|for |while |try:|except|finally:|with |"
                        r"return |raise |@|#|[A-Za-z_][\w.]*\s*(?:=|\(|\[|\.|,|\+=|-=))")


def rest_asked(text: str) -> tuple[str, str]:
    """Lane repairpair1: a rewrite the output cap cut, as `N11.continued` reads request 1's: its head kept to its last
    complete top-level statement before the definition the cut fell in (`resume`), and the ask for the rest under it
    (`repair.REST_HEAD`, `CONTINUE_ASK`); ("", "") where nothing of it is complete."""
    kept, opened = resume(text)
    if not kept.strip():
        return "", ""
    ask = CONTINUE_ASK.format(where=f" inside `{opened}`" if opened else "",
                              start=f"`{opened}` written again whole, then every line after it" if opened
                              else "every line after the last one above")
    return kept, f"{repair.REST_HEAD}\n\n```python\n{kept}```\n\n{ask}\n"


def continuation_code(reply: str) -> str:
    """Lane n11e: the code a continuation's reply holds: its blocks' lines in order with every fence line dropped (a
    fence the reply restarts inside a block, ```` ```python ```` again, opens nothing new: the code runs on), prose
    outside the blocks dropped; where the reply opens with code and no fence, that code is a block its first bare
    fence closes; the reply whole where it holds no fence and opens with code, else nothing."""
    lines = runner.THINK.sub("", reply or "").split("\n")
    fences = [k for k, line in enumerate(lines) if line.strip().startswith("```")]
    head = [line for line in (lines[:fences[0]] if fences else lines) if line.strip()]
    if not fences:
        return "\n".join(lines).strip("\n") if head and CODE_START.match(head[0]) else ""
    inside, kept = bool(head) and bool(CODE_START.match(head[0])), []
    for line in lines:
        fence = line.strip()
        if fence.startswith("```"):
            inside = True if fence.strip("`").strip() else not inside
            continue
        if inside:
            kept.append(line)
    return "\n".join(kept).strip("\n")


def task_text(task, pathmap, data: str, described: str, named: tuple[str, ...], firstout: bool = False,
              contracted: bool = False, order: tuple[str, ...] | None = None, edit=None) -> str:
    """The loop's first message up to its clock: the task, the paths, the input data and files described, the output
    files named, under lane firstout1's `FIRSTOUT=on` (`firstout`) the first of them asked alone (`FIRSTOUT_FILES`),
    and under lane contract1's `CONTRACT=on` or `refuse` (`contracted`) the contract's sentence
    (`stated_contract.ASK`); each a `# ` section, in that order, or in `order`'s by their heads (lane prompt1's
    `taskfirst`: `prompts.TASKFIRST_ORDER`); `edit`, where given, says the sections (lane clean1's `N11.cleaned`)."""
    files = ("The task names these output files; the checks look for each at its path in the output directory: "
             + ", ".join(named) + "." if named else
             "The task names no output file by name; write each one it describes into the output directory.")
    if firstout:
        files += "\n\n" + FIRSTOUT_FILES.format(first=named[0] if named else FIRSTOUT_UNNAMED)
    if contracted:
        files += "\n\n" + stated_contract.ASK
    sections = {"Task": task.instruction.strip(), "Paths": pathmap.text(), "Input data": data.strip(),
                "Input files, described": described, "Output files": files}
    sections = edit(sections) if edit is not None else sections
    return "".join(f"# {head}\n\n{sections[head]}\n\n" for head in order or sections
                   if sections[head] or head != "Input files, described")     # today's: that one only where any


def review_text(ws: commands.Workspace, stated: props.Stated, ran: commands.Ran | None, mode: str,
                budget: str, ask: str | None = None) -> str:
    """Form `review`'s observation (lane loop1) over the output as it stands, built by code at no request: the run's
    exit, the output's digest, the findings and the checks, "none" where none; every property the instruction states
    with what the output holds of it (`props.audit`: held, broken or unchecked); the output files, keys and columns
    it names, each present or absent; then the ask (`REVIEW_ASK`) and `budget`, kept whole within `REVIEW_BYTES`."""
    failures = ws.checks()
    digest_, findings = ws.sanity()
    try:
        held, named = props.audit(stated, ws.out, ws.named)
    except Exception:  # noqa: BLE001 - the audit never costs the review its observation
        held, named = [], []
    parts = [f"run {commands.WORK}/main.py: exit {ran.exit} after {ran.seconds:.1f} s" if ran is not None else
             f"{commands.WORK}/main.py ran", f"outputs:\n{digest_}" if digest_ else "outputs: none",
             sanity.said(findings) or "findings: none",
             "checks: all pass" if not failures else "checks failing:\n" + "\n".join(f"- {line}" for line in
                                                                                     failures[:12])]
    parts.append(f"the task's stated properties ({len(held)}), as the output holds them:\n" + "\n".join(
        f"- {ws.redact(line)}" for line in held) if held else "the task's stated properties: none read from its text")
    if named:
        parts.append("the output the task names:\n" + "\n".join(f"- {ws.redact(line)}" for line in named))
    return fit("\n".join(parts), REVIEW_BYTES, f"{REVIEW_ASK[mode] if ask is None else ask}\n{budget}")


class N11:
    write_setting = "whole"
    lib_setting = "open"
    firstout_setting = "off"
    stated_setting = "off"
    prompt_setting = "full"
    prompt_carded = None
    audit_setting = "off"

    def __init__(self, task_dir: Path, out: Path, client, steps: guard.Steps, work: Path, deadline: float,
                 mode: str = MODE, form: str = FORMS[0], form_source: str = "") -> None:
        self.task_dir, self.out, self.client, self.log = Path(task_dir), Path(out), client, steps
        self.deadline, self.mode = deadline, mode
        self.mode_setting = mode                            # lane fenced1: the reply form the loop starts in, where
        self.mode_source = MODE_SOURCE if mode == MODE else "given"     # it was read, and whether the route's
        self.mode_refused = False                           # refusal of the tools turned it fenced
        self.form, self.form_source = form, form_source     # one of `FORMS`
        self.prompt_setting, self.prompt_source = PROMPT, PROMPT_SOURCE     # lane prompt1: the loop's prompt's word;
        self.prompt_fell = False                    # today's text where its paragraphs were not read (`control`);
        self.prompt_carded: list[library.Function] | None = None   # the functions the system prompt's cards carry,
        self.prompt_head = None                     # its bytes before them, the first message's layout, the unit's
        self.lean_cards_setting, self.lean_cards_source = LEAN_CARDS, LEAN_CARDS_SOURCE     # lane lean2: `lean`'s
        self.lean_cards_bytes: int | None = None    # cards' form and their bytes in the system prompt
        self.prompt_layout = None                   # first attempted and first saved work/main.py (the reached
        self.prompt_attempt: str | None = None      # checks) and the edits whose anchor was not met, said once
        self.prompt_first: str | None = None
        self.prompt_attempt_saved = False
        self.prompt_missed: list[str] | None = None
        self.prompt_first_bytes: int | None = None  # lane clean1: the first loop request's messages as sent, bytes
        self.work = Path(work)
        self.unit: unit.Unit | None = None
        self.ws: commands.Workspace | None = None
        self.messages: list[dict] = []
        self.counts: Counter = Counter()
        self.band = ""
        self.seen: set = set()
        self.fired: list[tuple[library.Function, str]] = []
        self.gate_said = False
        self.finishes = 0
        self.stopped = ""
        self.summarised = False
        self.index = ""
        self.cards = ""                 # the pointed functions' cards (Library.carded): the loop's and the library pass's
        self.cuts: list[str] = []       # what was cut from them, a line a card
        self.aside: Path | None = None  # the first run's output, set aside while the library pass runs
        self.versions: dict[str, list[tuple[int, str]]] = {}    # each file's writes: (turn, sha256), in order
        self.recurrences: Counter = Counter()                   # each file's returns to an earlier version
        self.gave: dict[int, str] = {}                          # each turn's last run, said in a line
        self.last_ran: dict[str, tuple[int, str, bool]] = {}    # each file's last run: (turn, its sha256 then, whether
        #                                                         it ran clean: exit 0, the checks passing, no finding)
        self.digests: list[int] = []                            # each exit-0 run's digest bytes
        self.sanity_refused = False
        self.streak_top = 0                             # the most replies cut in a row (`cut_streak_max`)
        self.cut_streak = 0
        self.reply_text = ""
        self.finish_said = ""
        self.stated: props.Stated | None = None         # what the instruction states of its outputs (lane n11e)
        self.no_call_streak = 0                         # replies in a row with no call
        self.last_frame = ""                            # the previous run's innermost failing line, "" where none
        self.entry = ""                                 # the stitched script's uncalled entry, said at its run
        self.standing: commands.Ran | None = None       # the run whose output stands at request 1's clean end
        self.review_from: int | None = None             # form `review`: the requests counted when the review opened
        self.review_first: dict = {}                    # the first output's snapshot then (common.contract.snapshot)
        self.review_hashes: dict[str, str] = {}         # and its files' hashes
        self.review_last: tuple | None = None           # the review's last run that wrote to the output: (clean,
        #                                                 the files it wrote, the output's snapshot after it)
        self.replies: list[tuple[str, int, bool]] = []  # each reply: its kind, completion tokens, cut at the cap
        self.result_setting = STATE_RESULT              # form `state`'s LAST RUN setting (lane state2), and its source
        self.result_source = STATE_RESULT_SOURCE
        self.index_mode, self.index_source = INDEX_MODE, INDEX_MODE_SOURCE      # the index's setting (lane index2)
        self.library: library.Library | None = None     # the library, for its `lib` directive and its cards
        self.state_system = ""                          # its system prompt and the input files described, built once
        self.state_inputs = ""
        self.state_shown: dict[str, None] = {}          # the House's files under work/ the state shows, in order
        self.state_last: tuple[int, list[str]] | None = None    # the last runs: their turn, each result as said
        self.state_notes: tuple[int, list[str]] = (0, [])   # the harness's answers to the last reply, and its turn
        self.ledger: list[str] = []                     # one line a turn
        self.state_bytes: list[int] = []                # each request's user turn's bytes, its files section's, its
        self.files_bytes: list[int] = []                # LAST RUN section's, and the ledger's lines it carried
        self.last_run_bytes: list[int] = []
        self.ledger_sent = 0
        self.state_resent: dict[tuple[str, str], list[int]] = {}    # each (file, text) resent identical: the turns
        self.state_failed: dict[tuple, int] = {}        # each failing run's frame and error: the first turn it failed
        self.state_clean: dict[str, str] | None = None  # the last clean run's output files' hashes (CLEAN_OUTPUT)
        self.state_clean_turn = 0
        self.state_opened: list[library.Function] = []  # the functions `lib` opened, the latest first
        self.state_carded: tuple[tuple[str, ...], str] = ((), "")  # the cards' functions and their text, as last built
        self.began = getattr(client, "deadline", None)  # the client's deadline at the start (lane clock1: `moved`)
        self.ending = ""                                # why a run ended the unit on its output (`RESERVE_END`)
        self.outcome: dict[str, tuple[int, str]] = {}   # each file's last run: its turn, said in a line
        self.sketch_mode, self.sketch_source = SKETCH, SKETCH_SOURCE    # form `sketch`'s switch (lane plan4)
        self.sketch_due = False                         # the ask made and the first reply not yet read
        self.sketch_note = ""                           # its picks' cards and unknown names, for the next message
        self.sketch_inside = False                      # `turn`'s sketch kept: the note joins the line's message
        self.sketch_kept: str | None = None             # that message's text: never trimmed (addendum 3)
        self.state_sketch = ""                          # form `state`'s SKETCH section (addendum 3), "" where none
        self.state_picks: list[library.Function] = []   # its picks the CARDS section adds after the pointer's
        self.state_cuts: list[str] = []                 # what the cards' trim cut, as last built
        self.sketch_did: list[str] = []                 # the sketch's part of its turn's ledger line
        self.sketch_lines: list[str] = []               # its picks' lines for that turn's notes
        self.sketch_thinking = SKETCH_THINKING          # its request's thinking (addendum 4), and as said once asked
        self.sketch_thought = "-"
        self.wrote: dict[str, None] = {}                # the files under work/ the House's saves wrote (lane files1)
        self.inputs_setting = INPUTS                    # lane inputs1: the input files' descriptions and their source,
        self.inputs_source = INPUTS_SOURCE
        self.inputs_text: str | None = None             # as first built, and their bytes
        self.review_thinking = REVIEW_THINKING          # form `review`'s review request's thinking, and its source
        self.review_thinking_source = REVIEW_THINKING_SOURCE
        self.review_asked = False                       # the review request sent, and its reply not yet read
        self.review_reply = False
        self.nocall_setting, self.nocall_source = NOCALL, NOCALL_SOURCE         # lane cap1's two switches, and
        self.picks_setting, self.picks_source = SKETCH_PICKS, SKETCH_PICKS_SOURCE   # their sources
        self.nocall_due = False                         # `ask`: an ask made and its reply not yet read
        self.pace_setting, self.pace_source = PACE, PACE_SOURCE     # lane pace1: the loop at the platform's pace
        self.request_seconds: list[float] = []          # each loop request's seconds on the unit's clock, to its reply
        self.run_seconds: list[float] = []              # each run's seconds
        self.clock0: float | None = None                # the time left as the loop's first message was built
        self.short = False                              # the short-clock mode, entered once and never left
        self.short_due = False                          # a turn's observations await the mode's ask
        self.last_clean = False                         # the last run clean with no finding
        self.first_write: tuple[int, bool] | None = None    # the first reply that wrote a file: its completion tokens,
        #                                                     and whether the cap cut it
        self.resort = fallback.Resort(FALLBACK, FALLBACK_SOURCE, form)  # lane fallback1: the last resort and its hold
        self.selection = selection.Selection(SELECT, SELECT_SOURCE, SELECT_THINKING, SELECT_THINKING_SOURCE,
                                             form, SELECT_BOUND, SELECT_BOUND_SOURCE)   # lane select2: the selection
        #                                                 call, and what it did; lane first1's bound on it
        self.end_kind = ""                              # what stopped the loop (`fallback.REASONS`), "" for none
        self.system_said: tuple[tuple[str, ...], tuple[str, ...]] | None = None     # lane putaway1: the tools the
        self.system_first: str | None = None            # last request's system prompt left unsaid and said put away,
        #                                                 and the loop's first request's system prompt as sent
        self.convcard_setting, self.convcard_source = CONVCARD, CONVCARD_SOURCE    # lane convcard1: the card's switch,
        self.convcard: conventions.Card | None = None   # the card as built for the loop's first message, and its size
        self.convcard_size = 0                          # in the process's unit (agent/n11/measure.py)
        self.last_kind = ""                             # and what made the last call due: `requests` or `clock`
        self.cut_setting, self.cut_source = CUT_MODE, CUT_MODE_SOURCE   # lane review2: a cut write's handling,
        self.held: tuple[str, str] | None = None        # under `continue` the named file and its head kept, the rest
        self.stitching: Call | None = None              # asked; the write the stitch made, said as stitched
        self.joining: tuple[str, str, str, bool, str, int | None] | None = None  # lane join1: a join that does not
        #                       compile, its file, its head and its junction's words, said with its write; lane join2:
        #                       whether its head is broken above its end (so the error is the head's), the error, its line
        self.said_broken: set[str] = set()              # lane join2: the files whose broken head was said and dropped
        self.last_main: commands.Ran | None = None      # work/main.py's last run (the review over the loop says it)
        self.cards_setting, self.cards_source = CARDS_MODE, CARDS_MODE_SOURCE   # lane first1: the cards whole,
        self.carded_order: list[library.Function] = []  # the functions the first message cards, in order,
        self.whole: library.Whole | None = None         # and the fit `made_whole` made
        self.inspect_setting, self.inspect_source = INSPECT_MODE, INSPECT_MODE_SOURCE   # and an inspect of a
        self.parts: dict[str, str] | None = None        # described file, the first message's description by file
        self.lastcall_setting, self.lastcall_source = LASTCALL, LASTCALL_SOURCE     # lane clock1 of 10-13: the last
        self.lastcall: dict = {}                        # call's timing, its ask's, its send's and its outcome's
        self.started = time.time()                      # the unit's start on the agent's clock (the keys' seconds)
        self.shim_setting, self.shim_source = SHIM, SHIM_SOURCE     # and the run's shim, and the write's check
        self.lint_setting, self.lint_source = LINT, LINT_SOURCE
        self.shim_fired: Counter = Counter()            # the shim's kinds, by the runs they fired in
        self.redirect_setting, self.redirect_source = REDIRECT_MODE, REDIRECT_MODE_SOURCE   # lane redirect1: the
        self.redirect_inspects = 0                      # wasting loop redirected; the inspect-only replies so far,
        self.redirect_at: int | None = None             # the request the waste state was declared at and the seconds
        self.redirect_left = 0.0                        # left then, the demand due on the next request's last
        self.redirect_due = False                       # observation, the files saved after the declaration
        self.redirect_saved: dict[str, None] = {}
        self.first_saved: float | None = None           # and the unit's first saved file's seconds
        self.lib_setting, self.lib_source = LIB_MODE, LIB_MODE_SOURCE   # lane libnever1: the library lookup
        self.write_setting, self.write_source = WRITE_MODE, WRITE_MODE_SOURCE   # lane fixdef1: a fix by definition
        self.called_setting, self.called_source = CALLED, CALLED_SOURCE     # lane called1: the called finding, each
        self.called_named: dict[str, int] = {}          # function it named by the turn it was first said
        self.called_logged: set[tuple[str, str]] = set()
        self.called_alone = False                       # whether the last finish's findings were its lines alone
        self.rules_setting, self.rules_source = RULES, RULES_SOURCE     # lane rules1: the output rules, the rules
        self.rules_fired: dict[str, None] = {}          # that fired on a run or a finish, the refused finish's rule
        self.rules_held: frozenset | None = None        # findings (what each is known by) until the next run reads
        self.kept: Path | None = None                   # them, and the last clean run's output kept, and its snapshot
        self.kept_snapshot: dict | None = None
        self.resend_setting, self.resend_source = RESEND, RESEND_SOURCE     # lane resend1: the unchanged re-send
        self.runs_at: dict[int, commands.Ran | None] = {}   # each turn's last run, `gave`'s record (None: refused)
        self.resent: tuple[str, str, int, tuple[str, ...]] | None = None    # the last write's unchanged refusal
        #                                                 under `finish`: ("ended", "read" or "refused", the file, the
        #                                                 turn it matched, the named outputs missing after its run)
        self.finishrun_setting, self.finishrun_source = FINISHRUN, FINISHRUN_SOURCE    # lane finishrun1
        self.finishfail_setting, self.finishfail_source = FINISHFAIL, FINISHFAIL_SOURCE    # lane finishfail1: the
        self.finishcheck_setting, self.finishcheck_source = FINISHCHECK, FINISHCHECK_SOURCE    # lane finishcheck1
        self.finishcheck_names: tuple[finishcheck.Named, ...] | None = None    # the contract's names, at a finish
        self.finishcheck_found: set[str] = set()        # the kinds its findings held, over the unit's finishes
        self.firstout_setting, self.firstout_source = FIRSTOUT, FIRSTOUT_SOURCE    # lane firstout1: the words, the
        self.clause_setting, self.clause_source = CLAUSE, CLAUSE_SOURCE    # lane clauses1: the first message's clause
        self.first_outputs: int | None = None           # named output files the first run exiting 0 wrote, and the
        self.named_files: tuple[str, ...] | None = None     # output files the task names, once the workspace is built
        self.shown_run: tuple[int, str, int | None, str] | None = None  # last run the House was shown (its turn,
        self.tail_budget = ""                           # file, exit and observation), and the budget line `tail` ended
        #                                                 the last run's observation with
        self.cardfail_setting, self.cardfail_source = CARDFAIL, CARDFAIL_SOURCE    # lane cardfail1: the card after
        self.cardfail_carded: set[str] = set()          # a failing run, the functions it carded, and those a `lib`
        self.cardfail_looked: set[str] = set()          # answer showed
        self.lint2_setting, self.lint2_source = LINT2, LINT2_SOURCE     # lane lint2: the write's two findings
        self.example_setting, self.example_source = EXAMPLE, EXAMPLE_SOURCE     # lane example1: the worked example
        self.example_shown: example.Shown | None = None     # built once, at the loop's first message (`example_part`)
        self.joinask_setting, self.joinask_source = JOINASK, JOINASK_SOURCE    # lane joinask1: the changed asks,
        self.junctions = joinask.Ledger()               # each file's open junction and the counts the keys carry,
        self.recut: str | None = None                   # the file whose rest the cap cut again (`resumed`, for `hold`),
        self.cut_named: set[str] = set()                # the files a write of was cut, the line each cut fell in
        self.cut_lines: dict[str, str] = {}             # (`joinask.cut_line`), the file whose write in the next reply
        self.defs_due: dict[str, tuple[str, int]] = {}  # is spliced, adding (`defs`, or `rest` over a kept head),
        self.defs_files: set[str] = set()               # the files asked for definitions until cut or written whole,
        self.defs_repeats: Counter = Counter()
        self.kept_said: set[str] = set()                # the files a fragment was refused for since their definitions ask,
        self.clean_runs: dict[str, tuple[int, str]] = {}    # each file's last script that ran clean (its turn, text),
        self.run_failed = False                         # whether the last run failed, the runs at the last
        self.defs_mark = 0                              # definitions ask (rule 3 asks once a failing run), and a head
        self.head_ran = -1                              # run made in a write's turn (`head_run`, `after_write`)
        self.repeat_setting, self.repeat_source = REPEAT, REPEAT_SOURCE    # lane repeat1: the unit's program states,
        self.repeats = repeat.Ledger()                  # the turn's latest state's recurrence, the ask due before the
        self.repeat_hit = 0                             # next request, the one-shot's output kept where it passed the
        self.repeat_due = False                         # checks, and what the one-shot is sent with (`solve`'s task,
        self.repeat_floor: Path | None = None           # path map and digest)
        self.repeat_with: tuple | None = None
        self.snapshot_setting, self.snapshot_source = SNAPSHOT, SNAPSHOT_SOURCE    # lane snapshot1: the state at a
        self.snapshot_sizes: list[int] = []             # failure, each said one's measure (`measure.UNIT`)
        self.derive_setting, self.derive_source = DERIVE, DERIVE_SOURCE    # lane derive1: the dated derivation; the
        self.derive_said: bool | None = None            # loop's first reply read for one, the unit's first script's
        self.derive_script: tuple[int, int] | None = None   # assertions and functions, and the checks whose findings
        self.witness_kinds: Counter = Counter()         # the runs shown carried
        self.contract_setting, self.contract_source = CONTRACT, CONTRACT_SOURCE     # lane contract1: the output
        self.contract_terms: props.Terms | None = None  # contract the instruction states, read once a unit, and the
        self.contract_kinds: Counter = Counter()        # kinds of the findings the runs shown carried
        self.stated_setting, self.stated_source = STATED, STATED_SOURCE     # lane conventions1: the system prompt's
        #                                                 conventions, stated once and operation by operation
        self.recon_setting, self.recon_source = RECON, RECON_SOURCE     # lane recon1: the reconciliation after a
        self.recon_kinds: Counter = Counter()           # clean run; the checks its findings and a refusal held, and
        self.recon_standing: frozenset | None = None    # the findings the last read run (or refusal) said, by ident
        self.recon_scripts: str | None = None           # and the scripts under work/ at the last read run (`recon_same`)
        self.audit_setting, self.audit_source = AUDIT, AUDIT_SOURCE     # lane audit1: the audit on the first clean
        self.audit_met = False                          # run; whether the unit's first clean run in the loop was met
        self.audit_at: int | None = None                # (said or skipped), the turn whose run said it, and whether the
        self.audit_step: bool | None = None             # system prompt said its fifth step (`system`, logged once)
        self.drafts_setting, self.drafts_source = DRAFTS, DRAFTS_SOURCE     # lane drafts1: the first drafts, and the
        self.drafting: drafts.Round | None = None       # round where it ran
        self.repair = repair.Repair(REPAIR, REPAIR_SOURCE, form, getattr(client, "cap", REQUESTS))    # lane
        self.framing = ""           # repairpair1: the unit's pairs, the task as the rewrite frames it (`solve`), and
        self.repair_turned = 0      # the finishes turned away as the last repair turn's read counted them
        self.join_setting, self.join_source = JOIN, JOIN_SOURCE     # lane join2 of 10-16: the join repaired; the
        self.held_tail: tuple[str, str, str, str] | None = None     # held head's file, the head, the received
        #                                                 lines of the statement its cut fell in and the received part
        #                                                 of the line it fell in (`hold`)
        self.staged: str | None = None      # the file the last write was staged beside (`written`, `after_write`)
        self.write_thinking_setting, self.write_thinking_source = WRITE_THINKING, WRITE_THINKING_SOURCE   # lane
        self.write_thinking_budget = WRITE_THINKING_BUDGET              # thinkwrite1: the write replies' thinking,
        self.write_thinking_budget_source = WRITE_THINKING_BUDGET_SOURCE    # its budget and the unit's asks
        self.thought = thinking.Ledger(self.write_thinking_setting if form in WRITE_THINKING_FORMS else "off",
                                       self.write_thinking_budget)
        self.repeats_asked = 0                      # `write_repeats` at the last request: more since, a re-send
        self.turn_calls: tuple[str, ...] = ()       # the last reply's calls by name: a `finish` there was not taken
        self.repair_before: dict | None = None

    @property
    def joinask_armed(self) -> bool:
        """Lane joinask1: `JOINASK=change` in the loop's forms (`JOINASK_FORMS`) under `CUT=continue`."""
        return self.joinask_setting == "change" and self.form in JOINASK_FORMS and self.continuing

    @property
    def repeat_armed(self) -> bool:
        """Lane repeat1: `REPEAT=change` in the loop's forms (`REPEAT_FORMS`)."""
        return self.repeat_setting == "change" and self.form in REPEAT_FORMS

    @property
    def write_defs(self) -> bool:
        """Lane fixdef1: `WRITE=defs` in the loop's forms (`WRITE_FORMS`): a write of a .py file that exists takes a
        text holding only definitions by name (commands.splice), and the write's words say so (`PART_DEFS`,
        commands.TOOLS_DEFS)."""
        return self.write_setting == "defs" and self.form in WRITE_FORMS

    @property
    def cut_streak(self) -> int:
        """Replies cut at the cap in a row; the most of them kept (`streak_top`)."""
        return self._cut_streak

    @cut_streak.setter
    def cut_streak(self, n: int) -> None:
        self._cut_streak, self.streak_top = n, max(self.streak_top, n)

    @property
    def continuing(self) -> bool:
        """Lane review2: a cut write continued (`CUT=continue`), read by the loop's salvage alone, so in the loop's
        forms alone (`CUT_FORMS`; form `state` keeps its own cut path, `state_salvage`)."""
        return self.cut_setting == "continue"

    @property
    def review_loop(self) -> bool:
        """Lane review2: form `review` with the selection call armed in it: every unit's request 1 is the loop's, and
        the review opens at the loop's first clean finish."""
        return self.form == "review" and self.selection.armed

    def moved(self) -> float:
        """The seconds the client's deadline has moved since the unit began: 0 but under the desk's
        `AGENTHON_MODEL_RATE_UNTIMED` (lane clock1; src/agenthon_t1/model.py), where each 429's wait moves it."""
        now = getattr(self.client, "deadline", None)
        return now - self.began if now is not None and self.began is not None else 0.0

    def left(self) -> float:
        """The seconds left to the loop: lane fallback1's reserve held off them while it is held."""
        return self.deadline + self.moved() - RUN_RESERVE_S - self.resort.held_s() - time.time()

    def requests_left(self) -> int:
        """The unit's requests left, and while the review is open its own (`REVIEW_REQUESTS`) where fewer; lane
        fallback1's request held off them while its reserve is held; lane repairpair1's room for B on the client's cap
        and not yet counted (`repair.Repair.unspent`) off them, so the loop's are today's while a pair is out."""
        left = self.client.cap - self.client.usage.requests - self.resort.held_requests() - self.repair.unspent()
        if self.review_from is not None:
            left = min(left, REVIEW_REQUESTS - (self.client.usage.requests - self.review_from))
        return left

    def spent(self) -> str:
        """Why no request is left: the unit's, or the review's."""
        if self.review_from is not None and self.client.cap - self.client.usage.requests >= 1:
            return f"the review's {REVIEW_REQUESTS} requests spent"
        return "no request left"

    def redact(self, text: str) -> str:
        return canary.redact(text, self.unit.canaries if self.unit else ())

    def kill(self) -> None:
        if self.ws is not None:
            self.ws.kill()
        self.resort.kill()
        if self.drafting is not None:   # lane drafts1: a draft's script running
            self.drafting.kill()
        self.repair.kill()              # lane repairpair1: B's script running, and nothing moved after it

    # --------------------------------------------------------------------------------------------- the clock

    def run_cap(self) -> float:
        """A loop run's seconds: at most `RUN_S`, and the time left less `ASK_S` where a request could follow it."""
        left = self.left()
        if self.requests_left() >= 1 and left - ASK_S >= MIN_RUN_S:
            return min(RUN_S, left - ASK_S)
        return min(RUN_S, left)

    def run_limit(self) -> tuple[float, float]:
        """Lane clock1: a loop run's limit and its reserve's line, `run_cap` both, but where that cap keeps a request's
        reserve below `RUN_S`: there the run's limit is the whole of the time left and the line the cap. A run that
        ends before the line leaves the request its reserve; one that runs past it leaves under `ASK_S`, so no request
        could follow it either way, and it runs on (`past_reserve`)."""
        cap, left = self.run_cap(), self.left()
        if self.requests_left() >= 1 and MIN_RUN_S <= left - ASK_S < RUN_S:
            return left, cap
        return cap, cap

    def past_reserve(self, ran: commands.Ran | None, limit: float, line: float) -> bool:
        """Whether a run given the rest of the clock (`run_limit`) ran past its reserve's line: then no request can
        follow it, and the unit ends on its output (`RESERVE_END`, `run_uncapped`)."""
        if ran is None or limit <= line or ran.seconds <= line:
            return False
        self.counts["run_uncapped"] = 1
        self.ending = RESERVE_END
        self.log("run", f"past the request's reserve at {line:.0f} s: it had the time left, {limit:.0f} s; the unit "
                        "ends on its output")
        trace.event("uncapped", line_s=round(line, 1), limit_s=round(limit, 1), seconds=round(ran.seconds, 1))
        return True

    # ---------------------------------------------------------------------------------------- the last call

    def unwritten(self) -> bool:
        """No output file the task names exists (no output file at all where it names none)."""
        if self.ws is None:
            return False
        if self.ws.named:
            return all(not (self.out / name).is_file() for name in self.ws.named)
        return not any(path.is_file() for path in self.out.rglob("*")) if self.out.is_dir() else True

    def last_call_due(self) -> bool:
        """Lane clock1: the last call, once a unit, where no output file the task names exists and the budget buys one
        more reply and not two (one request left, or the clock short of two: `two_replies`, `LAST_CALL_S` but under
        lane pace1's measured pace; under lane clock1's `LASTCALL=early` of 10-13, also where the next request's send
        would leave under a whole reply and its run: `by_send`); the loop's own stops come first."""
        if self.counts["last_call_asked"] or (self.requests_left() > 1 and self.two_replies() and not self.by_send()):
            return False
        return self.unwritten()

    def can_force(self) -> bool:
        """Lane clock1: the forced ask after replies with no call, once a unit, while the budget buys two more replies."""
        return not self.counts["prose_forced"] and self.requests_left() >= 2 and self.two_replies()

    def evidence(self) -> str:
        """The last call's evidence: each .py file under work/ as it stands (work/main.py last), the functions and
        classes it defines, and its last run's line where it ran (changed since, where it was), within
        `LAST_CALL_BYTES`."""
        lines = []
        for path in sorted(self.ws.dir.glob("*.py"), key=lambda each: (each.name == "main.py", each.name)):
            named = f"{commands.WORK}/{path.name}"
            text = path.read_text(encoding="utf-8", errors="replace")
            ran = self.outcome.get(named)
            last = self.last_ran.get(named)
            said = f"; its last run, turn {ran[0]}: {ran[1]}" if ran else "; not run"
            if ran and last is not None and last[1] != sha256(text):
                said += " (changed since)"
            lines.append(f"- {named}: " + (", ".join(defined(text)) or "no function") + said)
        return self.redact(fit("the files under work/ as they stand:\n" + "\n".join(lines) if lines else
                               "no file under work/ yet", LAST_CALL_BYTES))

    def call_last(self) -> None:
        """The last call's ask (`LAST_CALL` in the reply form, the evidence and the budget) after the conversation's
        last turn: into its last user turn where it ends on one, else a user turn of its own."""
        self.counts["last_call_asked"] = 1
        self.last_kind = "requests" if self.requests_left() <= 1 else "clock"     # lane fallback1's reason
        self.counts["last_call_by_pace"] = int(self.requests_left() > 1 and self.left() >= LAST_CALL_S
                                               and not self.two_replies())  # the pace's: LAST_CALL_S and the cap
        #                                                                     would not ask it yet
        timed = self.last_asked()
        ask = f"{LAST_CALL[self.mode]}\n{self.evidence()}\n{self.budget()}"
        if self.messages and self.messages[-1]["role"] == "user":
            self.messages[-1] = {**self.messages[-1], "content": f"{self.messages[-1]['content']}\n\n{ask}"}
        else:
            self.messages.append({"role": "user", "content": ask})
        self.log("last call", f"asked: {self.requests_left()} request(s) and {max(self.left(), 0):.0f} s left; no "
                              "output file the task names" + timed)

    def last_turn(self, found: list[Call], turn: int) -> str:
        """The last call's reply: each whole write saved as `write` saves one (a cut one's definitions kept as its
        draft), nothing else in it run; then work/main.py run with all the time left (`run_last`)."""
        for call in found:
            if call.cut and call.name in ("write", "cut", "unparsed"):
                self.salvage(call, "", final=True)
            elif call.name == "write" and not call.problem and not call.cut:
                said = self.written(call.arguments["path"], call.arguments["text"], "")
                self.log("write", first_line(said)[:200])
                self.lastcall["saved"] = self.lastcall.get("saved", 0) + said.startswith("wrote ")
        trace.event("turn", turn=turn, calls=[call.describe() for call in found], ended=True)
        return self.stop(self.run_last(), self.last_kind)

    def run_last(self) -> str:
        """The last call's run: work/main.py as it stands, run with all the time left (no reserve: no request follows),
        and the unit's end on its output (`last_call_output`: a file the task names written by it; `last_call_clean`:
        it exited 0 with the checks passing). The stop's reason."""
        mainpath = f"{commands.WORK}/main.py"
        if not (self.ws.dir / "main.py").is_file():
            self.log("last call", f"no {mainpath} to run")
            return f"the last call: no {mainpath} to run; the unit ends with the output as it stands"
        seconds = self.left()
        if seconds < MIN_RUN_S:
            self.log("last call", f"not run: {max(seconds, 0):.0f} s left")
            return f"the last call: {max(seconds, 0):.0f} s left, under a run's {MIN_RUN_S:.0f} s"
        self.counts["implicit_runs"] += 1
        _, ran = self.ws.run(mainpath, seconds, self.seen, lambda done: "")
        self.ran(ran, seconds)
        self.counts["last_call_output"] = int(ran is not None and bool(ran.wrote) and not self.unwritten())
        self.counts["last_call_clean"] = int(ran is not None and ran.clean)
        self.log("last call", f"{mainpath} ran with the time left: " + ("a named file written" if
                                                                         self.counts["last_call_output"] else
                                                                         "no named file written")
                 + ("; clean" if self.counts["last_call_clean"] else ""))
        trace.event("last_call", output=self.counts["last_call_output"], clean=self.counts["last_call_clean"])
        return f"the last call: {mainpath} ran with the time left; the unit ends on its output"

    # ----------------------------------------------------------------- lane clock1 of 10-13: the last call by its send

    def client_end(self) -> float:
        """The last moment a reply can arrive: the client's deadline (moved with it, under the desk's untimed waits)
        less its reserve (`model.RESERVE_S`), never past the loop's own end (lane fallback1's reserve held off it)."""
        own = self.deadline + self.moved() - RUN_RESERVE_S - self.resort.held_s()
        deadline = getattr(self.client, "deadline", None)
        return min(own, deadline - model.RESERVE_S) if deadline is not None else own

    def send_room(self) -> tuple[float, float, float]:
        """A request asked now: its send (where the desk paces its sends, after their interval:
        `common.house.Client.next_send`), the client's clock left at it, and at the next request's send, a cycle later:
        the measured pace (else the platform's, `PACE_PRIOR_S`) and a run's expected seconds, never under the desk's
        interval."""
        upcoming = getattr(self.client, "next_send", None)
        send = upcoming() if callable(upcoming) else time.time()
        pace = self.pace()
        cycle = max((pace if pace is not None else PACE_PRIOR_S) + self.run_expected(), model.MIN_INTERVAL_S)
        room = self.client_end() - send
        return send, room, room - cycle

    def by_send(self) -> bool:
        """`LASTCALL=early`: the last call due by its send, not its ask: where the next request's send would leave the
        client under a whole reply and its run (`LASTCALL_REPLY_S`, `LASTCALL_RUN_S`), this request is the last whose
        reply has that room, so it is the last call. Under `late`, never: the last call as today."""
        return self.lastcall_setting == "early" and self.send_room()[2] < LASTCALL_REPLY_S + LASTCALL_RUN_S

    def clocked(self, at: float) -> float:
        """A moment as seconds on the unit's clock: since the agent's start, less what the desk's untimed waits moved."""
        return at - self.started - self.moved()

    def last_asked(self) -> str:
        """The last call asked (either form's loop): whether the send rule asked it where today's would not
        (`lastcall_by_send`), its seconds on the unit's clock and the client's last send before it; the log's words
        under `early`, else ""."""
        self.counts["lastcall_by_send"] = int(self.requests_left() > 1 and self.two_replies())
        send, room, after = self.send_room()
        self.lastcall = {"asked": self.clocked(time.time()), "before": getattr(self.client, "last_sent", None)}
        if self.lastcall_setting != "early":
            return ""
        return (f"; by its send ({self.lastcall_source}): sent at about {self.clocked(send):.0f} s with "
                f"{room:.0f} s of the client's clock left, the next request's {after:.0f} s")

    def last_sent(self, reply) -> None:
        """The last call's request done: its send (the client's last, where it sent one), the client's clock left at
        it, and its outcome (`lost`: no reply, abandoned at the clock or refused unsent; `cut` at the output cap;
        `whole`), traced."""
        sent = getattr(self.client, "last_sent", None)
        if sent is not None and sent != self.lastcall.get("before"):
            self.lastcall.update(sent=self.clocked(sent), left=self.client_end() - sent)
        self.lastcall["outcome"] = "lost" if reply is None else "cut" if reply.finish_reason == "length" else "whole"
        trace.event("lastcall", mode=self.lastcall_setting, by_send=self.counts["lastcall_by_send"],
                    **{key: round(self.lastcall[key], 1) for key in ("asked", "sent", "left") if key in self.lastcall},
                    outcome=self.lastcall["outcome"])

    def lastcall_keys(self) -> dict:
        """Lane clock1's keys of 10-13 (`LASTCALL_KEYS`): the setting, whether the send rule asked the last call, its
        ask's and send's seconds on the unit's clock, the client's clock left at its send (`-` where none was asked or
        sent), and its outcome with the files its reply saved (`whole:1`)."""
        timed = {key: f"{self.lastcall[name]:.1f}" if name in self.lastcall else "-"
                 for key, name in (("lastcall_asked_at_s", "asked"), ("lastcall_sent_at_s", "sent"),
                                   ("lastcall_left_s", "left"))}
        outcome = self.lastcall.get("outcome")
        return {"lastcall_mode": self.lastcall_setting, "lastcall_by_send": self.counts["lastcall_by_send"], **timed,
                "lastcall_outcome": f"{outcome}:{self.lastcall.get('saved', 0)}" if outcome else "-"}

    # ----------------------------------------------------------------------------- lane pace1: the platform's pace

    @property
    def paced(self) -> bool:
        """`PACE=adaptive` in the loop's forms `loop` and `sketch`; forms `review` and `state` keep today's clock."""
        return self.pace_setting == "adaptive" and self.form in ("loop", "sketch")

    def pace(self) -> float | None:
        """The measured pace: the median of the unit's loop requests' seconds on its clock so far, each from its send
        to its reply (the pacing's waits in it, but a wait the desk's untimed switch took off the clock); None before
        any reply."""
        return statistics.median(self.request_seconds) if self.request_seconds else None

    def run_expected(self) -> float:
        """A run's expected seconds: the median of the unit's runs' seconds so far, `RUN_PRIOR_S` before any."""
        return statistics.median(self.run_seconds) if self.run_seconds else RUN_PRIOR_S

    def bought(self, pace: float | None = None, left: float | None = None) -> float:
        """The requests the clock still buys: the time left (`left`, else now's) over a request at `pace` (else the
        measured one) and a run of its expected seconds."""
        pace = self.pace() if pace is None else pace
        left = self.left() if left is None else left
        return max(left, 0.0) / max((pace or 0.0) + self.run_expected(), 0.001)    # a zero cycle buys without bound

    def two_replies(self) -> bool:
        """Whether the clock buys two more replies, this one and one after it (the last call's and the forced ask's
        test): under `adaptive` with a measured pace, where after one reply at that pace and a run of its expected
        seconds a request's `ASK_S` would still be left, and never fewer seconds than lane clock1's
        `LAST_CALL_S`, which is that rule at a pace of `ASK_S` (2 x `ASK_S`): at a faster pace the window before the
        loop's own stop is one turn wide, which a long run steps over, so the pace makes the last call earlier and
        never later; else `LAST_CALL_S`."""
        pace = self.pace() if self.paced else None
        if pace is None:
            return self.left() >= LAST_CALL_S
        return self.left() - pace - self.run_expected() >= ASK_S and self.left() >= LAST_CALL_S

    def short_start(self) -> str:
        """The first message's pace text, before an armed clause: the time left now kept as the unit's clock (`clock0`);
        under `adaptive`, where that clock at the prior pace (`PACE_PRIOR_S`: no reply yet) buys fewer than
        `SHORT_CLOCK_REQUESTS`, `SHORT_FIRST` (the whole script as the first move, the cards already shown, nothing
        looked up first; form `sketch`'s turn skipped, `sketch_ask`), else "". The mode itself waits on a measured
        pace (`paced_turn`)."""
        self.clock0 = self.left()
        if not self.paced:
            return ""
        bought = self.bought(PACE_PRIOR_S, self.clock0)
        short = bought < SHORT_CLOCK_REQUESTS
        self.counts["short_first"] = int(short)
        self.log("pace", f"{self.pace_setting} ({self.pace_source}): {self.clock0:.0f} s buy {bought:.1f} request(s) at "
                         f"the prior {PACE_PRIOR_S:.0f} s a request and {RUN_PRIOR_S:.0f} s a run, "
                         + (f"under {SHORT_CLOCK_REQUESTS}: the first message asks the whole script first" if short
                            else f"not under {SHORT_CLOCK_REQUESTS}: the first message as the loop's"))
        if not short:
            return ""
        said = SHORT_FIRST[self.mode].format(cards=SHORT_CARDS if self.cards else "")
        if self.lib_never:      # lane libnever1: the lookups the loop offers, `inspect` but under `INSPECT=never`
            said = said.replace(LOOKS_SAID[self.mode][("lib", "inspect")],
                                LOOKS_SAID[self.mode][() if self.never_armed else ("inspect",)], 1)
        return "\n" + said + "\n"

    def paced_turn(self) -> None:
        """After each reply (lane pace1): the pace and the requests the clock buys on the trace (`pace`); under
        `adaptive`, the short-clock mode entered where they fall under `SHORT_CLOCK_REQUESTS`, at this reply, and
        never left (`short_clock`, `short_clock_at`)."""
        pace, run, left = self.pace(), self.run_expected(), self.left()
        bought = self.bought(pace, left)
        if self.paced and not self.short and bought < SHORT_CLOCK_REQUESTS:
            self.short = True
            self.counts["short_clock"], self.counts["short_clock_at"] = 1, self.counts["turns"]
            self.log("pace", f"the short-clock mode from reply {self.counts['turns']}: {bought:.1f} request(s) bought "
                             f"at {pace:.1f} s a request and {run:.1f} s a run, under {SHORT_CLOCK_REQUESTS}")
        trace.event("pace", reply=self.counts["turns"], request_s=round(self.request_seconds[-1], 3),
                    pace_s=round(pace, 3), run_s=round(run, 3), left_s=round(left, 1), bought=round(bought, 2),
                    short=int(self.short))

    def short_ask(self) -> None:
        """The short-clock mode's ask (`SHORT_ASK`) after the last turn's observations, by what the last run left: finish
        where it was clean (exit 0, the checks passing, every named output file among them, and no finding:
        `short_finish_asked`), the script where work/main.py is not written, else the fix and its run in one reply;
        withheld after the turn that said lane audit1's audit, whose message asks its answers instead."""
        if self.audit_at is not None and self.audit_at == self.counts["turns"]:     # lane audit1: the message asks
            self.log("pace", "the short clock's ask withheld: the audit asks its answers")  # the audit's answers
            return
        kind = self.state_kind()
        self.counts["short_finish_asked"] += kind == "finish"
        kept = self.messages[-1]["content"] == self.sketch_kept     # the sketch's kept message: whole, the ask in it
        self.messages[-1] = {**self.messages[-1], "content": f"{self.messages[-1]['content']}\n"
                                                             f"{SHORT_ASK[kind][self.mode]}"}
        if kept:
            self.sketch_kept = self.messages[-1]["content"]
        self.log("pace", f"the short clock's ask: {kind}")

    def budget(self) -> str:
        missing = self.ws.missing() if self.ws else []
        return (f"budget: {self.requests_left()} request(s) and {max(self.left(), 0):.0f} s left; "
                + (f"missing: {', '.join(missing[:8])}" + (f" and {len(missing) - 8} more" if len(missing) > 8 else "")
                   if missing else "every named output file is present"))

    # ----------------------------------------------------------------------------------------------- the gate

    def used(self) -> tuple[bool, set[str]]:
        """Whether any file under work/ imports qf, and the qf functions they call."""
        imported, called = False, set()
        for path in sorted(self.ws.dir.rglob("*.py")) if self.ws else []:
            try:
                found, names = qf_use(path.read_text(encoding="utf-8", errors="replace"))
            except OSError:
                continue
            imported, called = imported or found, called | names
        return imported, called

    def rewritten(self) -> int:
        """`lib_rewritten` (lane lib4): once a pointed function's card reached the House (`pointer_shown`), the
        pointed functions a file under work/ at the end computes by hand (`library.rewritten`)."""
        if not (self.fired and self.counts["pointer_shown"] and self.ws and self.library is not None):
            return 0
        texts = []
        for path in sorted(self.ws.dir.rglob("*.py")):
            try:
                texts.append(path.read_text(encoding="utf-8", errors="replace"))
            except OSError:
                continue
        known = frozenset(function.name for function in self.library.functions
                          if function.shelf not in library.UNPOINTED)   # a write_json call computes nothing
        return len(library.rewritten([function for function, _ in self.fired], texts, known))

    def gate(self, ran: commands.Ran | None) -> str:
        """The gate's line after a run (above), "" where it says nothing; adoption counted."""
        if ran is None or not self.fired:
            return ""
        imported, called = self.used()
        pointed = {function.name for function, _ in self.fired}
        if self.gate_said and not self.counts["gate_adopted"] and imported and called & pointed:
            self.counts["gate_adopted"] = 1
            self.log("gate", "adopted: " + ", ".join(sorted(called & pointed)))
        if self.gate_said or ran.exit != 0 or (ran.clean and not ran.findings and not GATE_HOLDS_CLEAN):
            return ""
        if imported and called & pointed:
            return ""
        function, alias = next(((f, a) for f, a in self.fired if f.name not in called), self.fired[0])
        self.gate_said, self.counts["gate_fired"] = True, 1
        self.log("gate", f"fired: {function.call}")
        trace.event("gate", function=function.name)
        said = function.conventions
        return (f"the task names {alias}; `{function.typed}` computes it" + (f" with {said[:300]}" if said else "")
                + "; call it")

    def tail(self, ran: commands.Ran | None, last: bool) -> str:
        """What a run's observation ends with, kept whole: the stitched script's uncalled entry, the same failing
        line, the gate's line and, where the run is the reply's last call, the budget's. The called finding the
        observation says beside the findings is counted here (lane called1)."""
        if ran is not None:
            self.called_found(ran.called)
        entry, self.entry = self.entry, ""
        parts = (entry, self.same_frame(ran), self.recon_same(ran), self.packaged(ran), self.gate(ran))
        self.tail_budget = self.budget() if last else ""     # lane finishfail1: kept for `saw` to drop
        return "\n".join(part for part in (*parts, self.tail_budget) if part)

    def packaged(self, ran: commands.Ran | None) -> str:
        """Lane clock1: a run that failed on importing work/ as a package, said with the import's form and the bare
        name modules under work/ import each other by (commands.packaged); "" otherwise."""
        if ran is None or not ran.error or not NO_WORK.search(ran.error):
            return ""
        return commands.packaged_said(commands.packaged(self.ws.dir), ran.frames[-1][0] if ran.frames else ran.path)

    def same_frame(self, ran: commands.Ran | None) -> str:
        """Lane n11e: a run that fails on the same source line, in the same file, as the run before it failed on is
        answered by quoting that line and asking for a different fix (`same_frame`); "" otherwise."""
        if ran is None:
            return ""
        here = f"{ran.frames[-1][0]}\n{ran.line}" if ran.error and ran.frames and ran.line else ""
        before, self.last_frame = self.last_frame, here
        if not here or here != before:
            return ""
        self.counts["same_frame"] += 1
        self.log("run", "the same failing line as the previous run")
        return SAME_FRAME.format(where=f"{ran.frames[-1][0]}:{ran.frames[-1][1]}", source=ran.line)

    # ------------------------------------------------------------------------------------------- the commands

    def execute(self, call: Call, last: bool = False, final: bool = False) -> tuple[str, bool]:
        """One call's observation, the budget line after it where it is the reply's `last` call, and whether it ends
        the unit; `final`, the reply's last write, after which the harness runs (`implicit`)."""
        tail = self.budget() if last else ""
        if call.cut:
            return self.salvage(call, tail), bool(self.ending)
        if call.name == "lib" and self.lib_never:
            self.counts["lib_asked"] += 1           # lane libnever1: a lookup still asked, in one line
            self.log("lib", f"{call.describe()}: not offered, answered in one line")
            return fit(LIB_ASKED[bool(self.cards)], commands.MALFORMED_BYTES, tail), False
        if call.problem:
            return fit(self.malformed(call.problem), commands.MALFORMED_BYTES, tail), False
        args, ws = call.arguments, self.ws
        if call.name in ("inspect", "lib") and self.redirecting and call.name in self.put_away():
            self.counts["redirect_looks"] += 1      # lane redirect1: a look while it is put away, in one line
            self.log("redirect", f"{call.describe()}: put away, answered in one line")
            return fit(self.redirect_held(), commands.MALFORMED_BYTES, tail), False
        if call.name == "write":
            stitched = call is self.stitching       # lane fixdef1: a join is the file whole, never spliced
            said = self.joined(args, tail) if stitched and self.joining else \
                self.written(args["path"], args["text"], tail, whole=stitched, final=final)
            self.log("write", first_line(said)[:200])
            return said, (self.resent is not None and self.resent[0] == "ended") or bool(self.ending)
        if call.name == "run":
            limit, line = self.run_limit()
            if limit < MIN_RUN_S:
                return fit(f"not run: {max(self.left(), 0):.0f} s left, under a run's {MIN_RUN_S:.0f} s",
                           commands.BOUNDS["run"], tail), False
            said, ran = ws.run(args["path"], limit, self.seen, lambda done: self.tail(done, last))
            self.ran(ran, limit)
            self.saw(said, ran)
            said = self.audited(self.carded_after(said, ran), ran)
            return said, self.past_reserve(ran, limit, line)
        if call.name == "inspect":
            said = self.inspected(args["path"], args.get("rows"), tail)
            self.counts["inspect_calls"] += 1
            self.log("inspect", first_line(said)[:120])
            return said, False
        if call.name == "lib":
            self.counts["lib_calls"] += 1
            said = ws.lib(args.get("name"), args.get("shelf"), tail)
            self.opened(args.get("name"))
            self.log("lib", first_line(said)[:120])
            return said, False
        return self.finish(last, str(args.get("reason") or ""))

    def malformed(self, why: str) -> str:
        """A malformed call's answer (`MALFORMED`, `why` its problem); under lane libnever1's `LIB=never` the tools it
        lists those the loop offers: `lib` left out, and `inspect` too under `INSPECT=never` (`MALFORMED_FORMS`)."""
        said = MALFORMED.format(why=why)
        if not self.lib_never:
            return said
        gone = ("inspect", "lib") if self.never_armed else ("lib",)
        said = said.replace(f"the tools are {', '.join(commands.NAMES)}",
                            "the tools are " + ", ".join(name for name in commands.NAMES if name not in gone), 1)
        for name in gone:
            said = said.replace(MALFORMED_FORMS[name], "", 1)
        return said
    def written(self, path: str, text: str, tail: str, whole: bool = False, final: bool = False) -> str:
        """A `write` hashed (lane n11d): a text the file already holds refused, naming the turn that wrote it and what
        its run gave; a return to an earlier version counted, and on the file's `CYCLE_RECURRENCES`-th, where its last
        `CYCLE_WINDOW` versions are at most three, the cycle named. Lane fixdef1: under `WRITE=defs` the file the write
        would leave is hashed, a definitions text's splice (commands.splice); `whole` (a join's) is never spliced. Lane
        resend1, `RESEND=finish`: the reply's last write (`final`) refused as unchanged from a turn whose run exited
        clean with no check failing, every named output present, is read as the finish (`resend_finish`); any other
        unchanged refusal kept for `after_write`'s line (`resent`)."""
        self.resent, self.staged = None, None
        here, named = self.ws.local(path)
        first_prompt_attempt = named == MAIN_FILE and self.prompt_attempt is None
        if first_prompt_attempt:     # lane prompt1: the first attempt, saved or not
            self.prompt_attempt = commands.unfenced(text)
        if here is None or commands.shadows(named):
            return self.ws.write(path, text, tail)
        due = self.defs_due.pop(named, None)    # lane joinask1: the write that answers a definitions ask
        mode = due[0] if due is not None and due[1] == self.counts["turns"] - 1 and not whole else None
        if mode is None and not whole and named in self.defs_files:
            mode = "defs"                       # and a file asked for definitions takes them by name after it
        defs = (self.write_defs or mode is not None) and not whole
        made = commands.splice(here, commands.unfenced(text), mode is not None) if defs and named.endswith(".py") \
            else None
        if mode == "rest" and made is not None and made.kind != "spliced":
            added = appended(here, commands.unfenced(text))
            if added is not None:               # statements sent for a head that lacks them: put after it
                text, defs, made = added, False, None
                self.log("write", f"{named}: what was sent put after what the file held")
        if mode is not None and made is not None and made.kind != "spliced" and named not in self.kept_said \
                and not whole_again(here, commands.unfenced(text)):
            self.kept_said.add(named)           # a fragment that would replace the file: refused, once per definitions ask
            self.log("write", f"{named}: a fragment neither definitions of it nor it whole: refused, the file kept")
            return fit(JOIN_KEPT.format(named=named, one=JOIN_ONE[self.mode].format(named=named)),
                       commands.BOUNDS["write"], tail)
        if self.join_armed and not whole and mode is None and named.endswith(".py") and here.is_file() and \
                (made is None or made.kind != "spliced"):    # lane join2 of 10-16 (d): a fragment staged beside
            why = rejoin.staged(commands.unfenced(text), here.read_text(encoding="utf-8", errors="replace"))
            if why:
                return self.stage(named, commands.unfenced(text), why, tail)
        sha = sha256(made.text if made is not None and made.kind == "spliced" else commands.unfenced(text))
        history = self.versions.setdefault(named, [])
        if here.is_file() and sha256(here.read_text(encoding="utf-8", errors="replace")) == sha:
            self.counts["write_repeats"] += 1
            turn = history[-1][0] if history and history[-1][1] == sha else None
            if turn is not None and self.resend_armed:
                ran = self.resend_ran(turn)
                missing = tuple(self.ws.missing()) if ran is not None else ()
                if ran is not None and not ran.failures and not missing and final:
                    return self.resend_finish(named, turn)
                self.resent = ("refused", named, turn, missing)
            self.repeat_met(self.repeats.resent(repeat.program(self.ws.dir)), named, "re-sent unchanged")  # lane
            #   repeat1: the program re-sent as it stands, after resend1's read (form `state` writes its own way)
            said = UNCHANGED.format(named=named, turn=turn, gave=self.gave.get(turn, "no run"),
                                    run=self.run_form(named)) if turn is not None else \
                HELD.format(named=named, run=self.run_form(named))
            if self.joinask_armed and mode is not None:
                self.defs_repeats[named] += 1
                if self.defs_repeats[named] >= 2:
                    self.ending = "unchanged definitions"
                    said = JOIN_STOP.format(named=named)
            return fit(said, commands.BOUNDS["write"], tail)
        earlier = next((turn for turn, seen in history if seen == sha), None)
        note = ""
        if earlier is not None:
            self.recurrences[named] += 1
            versions = {seen for _, seen in history[-(CYCLE_WINDOW - 1):]} | {sha}
            if self.recurrences[named] == CYCLE_RECURRENCES and len(versions) <= 3:
                note = CYCLE.format(named=named, k=len(versions), turn=earlier, gave=self.gave.get(earlier, "no run"))
                self.log("write", f"{named}: a cycle among {len(versions)} versions")
        said = self.ws.write(path, text, "\n".join(part for part in (note, tail) if part), defs=defs,
                             adding=mode is not None)
        if said.startswith("wrote "):
            self.defs_repeats.pop(named, None)
            history.append((self.counts["turns"], sha))
            self.saved(named)
            if first_prompt_attempt:
                self.prompt_attempt_saved = self.prompt_first == self.prompt_attempt
            if self.joinask_armed and "does not compile" not in first_line(said):
                self.junctions.close(named)        # lane joinask1: the file whole and compiling closes its junction
            if made is None or made.kind != "spliced":
                self.defs_files.discard(named)  # written whole: no longer held for definitions
        return said

    def stage(self, named: str, text: str, why: str, tail: str) -> str:
        """Lane join2 of 10-16, `JOIN=repair` (d): a write over `named`, a saved file that compiles, shorter than it and
        unable to run in its place (`rejoin.staged`'s `why`), saved beside it as `named` + `rejoin.STAGED` and the file
        kept, answered in `JOIN_STAGED`'s words naming both; `after_write` then runs work/main.py where it changed
        since its last run (`staged`)."""
        beside = f"{named}{rejoin.STAGED}"
        self.ws.write(beside, text)
        self.staged = named
        self.counts["join_staged"] += 1
        self.log("write", f"{named}: {len(text.encode()):,} B that cannot run in its place ({why}) staged as {beside}; "
                          "the file kept")
        trace.event("join_staged", path=named, staged=beside, why=why, bytes=len(text.encode()))
        return fit(JOIN_STAGED.format(named=named, staged=beside), commands.BOUNDS["write"], tail)

    def resend_ran(self, turn: int) -> commands.Ran | None:
        """Lane resend1: turn `turn`'s run, the record its `gave` line was built from, where it exited clean: exit 0
        with no traceback (so not stopped at its limit, nor refused); else None."""
        ran = self.runs_at.get(turn)
        return ran if ran is not None and ran.exit == 0 and not ran.error else None

    def resend_finish(self, named: str, turn: int) -> str:
        """Lane resend1: an unchanged write read as the finish (`written`): `finish` run as a written one is, its
        observation under `RESEND_READ`, the budget line after it; `resent` says whether it ended the unit."""
        self.counts["resend_finished"] += 1
        said, ended = self.finish(True, resent=True)
        self.resent = ("ended" if ended else "read", named, turn, ())
        return f"{RESEND_READ.format(named=named, turn=turn)}\n{said}"

    def run_form(self, named: str) -> str:
        """How the House asks a run of `named` in its reply form."""
        return f'run(path="{named}")' if self.mode == "tools" else f"a line `run {named}`"

    def salvage(self, call: Call, tail: str, final: bool = False) -> str:
        """A call the output cap cut: a write's complete definitions kept as `work/<name>.draft` and named, else the
        call said lost; the `SHORT_STREAK`-th cut reply in a row asks for one small file. Under `CUT=continue` (lane
        review2) a .py write's head is kept and the rest asked instead (`hold`; `final`, the last call's reply, keeps
        it and asks nothing), until the `SHORT_STREAK`-th cut in a row, where the split returns."""
        arguments = self.recovered(call)
        text, path = arguments.get("text"), arguments.get("path") or f"{commands.WORK}/main.py"
        short = (CUT_SHORT_FIRSTOUT if self.firstout_armed else CUT_SHORT.format(n=self.cut_streak)) \
            if self.cut_streak >= SHORT_STREAK else ""
        here, named = self.ws.local(path)
        if call.name not in ("write", "cut", "unparsed") or not text or here is None:
            empty = CUT_FENCED if call.name == "cut" and self.mode_setting == "fenced" and self.form in MODE_FORMS \
                and self.mode_source in (MODE_ENV, MODE_FILE) else CUT_CALL
            return asked(self.split_words(empty), short, commands.MALFORMED_BYTES, tail)
        if self.continuing and named.endswith(".py"):
            if not short or self.joinask_armed:     # lane joinask1: the junction's asks bound the cuts instead
                held = self.hold(named, text, tail, final)
                if held is not None:
                    return held
            else:
                self.drop_part(named)
                self.log("continue", f"{named}: {self.cut_streak} replies cut in a row: the split asked")
        if named.endswith(".py"):
            kept, names, where = draft(text)
        else:
            kept, names, where = "".join(text.splitlines(keepends=True)[:-1]), [], "at its last line"
        draft_path = f"{commands.WORK}/{Path(named).stem}.draft"
        if kept:
            self.ws.write(draft_path, kept)
            self.counts["drafts"] += 1
            self.log("draft", f"{named}: {len(kept.encode()):,} B kept as {draft_path}, {len(names)} definition(s); "
                              f"the cut {where}")
        said = CUT_DRAFT.format(named=named, kept=CUT_KEPT.format(
            draft=draft_path, defines=f" (it defines {', '.join(names[:12])})" if names else "", where=where)
            if kept else CUT_NONE)
        return asked(self.split_words(said), short, commands.BOUNDS["write"], tail)

    def hold(self, named: str, text: str, tail: str, final: bool = False) -> str | None:
        """Lane review2, `CUT=continue`: a .py write the cap cut kept to its last complete line as the head of
        `named`, saved as `named` + `PART` (never run), and the rest asked in `CONTINUE_ASK`'s shape (`CUT_HELD`: the
        head's last lines quoted as written, no count; one block of code, no line already written), the next reply
        read as that rest (`resumed`). No draft is written. None where no line of it is complete (the split's words
        then); "" under `final`, which asks nothing. Lane join2: kept to the end of its last whole statement
        (`boundary_head`), so the quote ends one and the rest starts the next, the line the cut fell in among them."""
        body = opened_fence(text)
        head = boundary_head(body)
        if not head.strip():
            return None
        if self.join_armed:         # lane join2 of 10-16: the received lines of the statement the cut fell in
            complete = complete_head(body)
            self.held_tail = (named, head, rejoin.tail_of(complete, head), body[len(complete):])
        recut, self.recut = self.recut == named, None
        if self.joinask_armed and not final:     # lane joinask1: a cut that is a failure gets the changed ask
            asked = self.join_cut(named, text, head, recut, tail)
            if asked is not None:
                return asked
        self.ws.write(f"{named}{PART}", head)
        self.log("continue", f"{named}: {len(head.encode()):,} B kept as {named}{PART} to its last complete line"
                             + ("" if final else "; the rest asked"))
        trace.event("cut_head", path=named, head_bytes=len(head.encode()), compiles=runner.compiles(head),
                    asked=not final)
        if final:
            return ""
        self.held = (named, head)
        self.counts["cut_continued"] += 1
        said = CUT_HELD.format(named=named, ends=self.redact(ends(head)), through=self.through(named))
        return fit(said, HELD_BYTES, tail)

    def drop_part(self, named: str) -> None:
        """The head's file of `named` removed, where there is one: the stitched file, or the split's draft, stands in
        its place."""
        here, _ = self.ws.local(f"{named}{PART}")
        if here is not None:
            here.unlink(missing_ok=True)

    # ------------------------------------------------------------- lane joinask1: the ask after a failed continuation

    def join_cut(self, named: str, text: str, head: str, recut: bool, tail: str) -> str | None:
        """`JOINASK=change`: a cut write of `named` read before its head is held (`hold`). A whole rewrite of a file cut
        before, cut after a failing run with no junction open, holds nothing and asks the fix as definitions (rule 3,
        `defs_ask`); a rest cut again (`recut`), a head held before in the unit (the same text again) or a cut of a
        file whose junction is open is a failure on its junction (`join_failed`); None where the cut is today's."""
        seen = self.junctions.seen(named, head)
        before = named in self.cut_named
        self.defs_files.discard(named)          # a cut of the file: its definitions mode ends there
        self.cut_named.add(named)
        self.cut_lines[named] = joinask.cut_line(opened_fence(text), head)
        here = self.ws.local(named)[0]
        if not recut and before and not self.junctions.open(named) and self.run_failed \
                and self.counts["runs"] > self.defs_mark and here is not None and here.is_file():
            return self.defs_ask(named, tail)
        if not (recut or seen or self.junctions.open(named)):
            return None
        kind = "refused" if seen else "cut_again"
        return self.join_failed(named, head, kind, JOIN_WHY["recut" if seen else "cut_again"].format(named=named),
                                tail)

    def join_failed(self, named: str, head: str, kind: str, why: str, tail: str, shown: str = "") -> str:
        """A failure of `kind` on `named`'s junction counted (`joinask.Ledger.failed`) and its ask, never the one
        before it: the statement shape (`join_statement`), the module shape (`join_module`), or, the junction's asks
        spent, what stands run (`head_run`); `shown`, the junction as joined, before the statement's ask."""
        module = joinask.module_part(head)
        previous = self.junctions.last.get(named)
        shape = self.junctions.failed(named, kind, joinask.module_fits(module, head))
        self.log("continue", f"{named}: {kind.replace('_', ' ')}, " + (f"the ask changed: the {shape} shape" if shape
                                                                       else "its asks spent: what stands runs"))
        trace.event("continue", path=named, kind=kind, ask=shape or "head run",
                    failures=self.junctions.failures[named])
        if shape is None:
            return self.head_run(named, head, why, tail)
        if shape == "module":
            return self.join_module(named, module, why, tail)
        return self.join_statement(named, head, why, tail, shown, retry=previous == "statement")

    def join_statement(self, named: str, head: str, why: str, tail: str, shown: str = "", retry: bool = False) -> str:
        """The statement shape (`JOIN_STATEMENT`): the head held again, its last statement named where it ends (and
        the definition it ends inside), the statement the cut fell in named by its first line where the head left it
        out, and the rest asked from there."""
        self.ws.write(f"{named}{PART}", head)
        self.held = (named, head)
        self.counts["cut_continued"] += 1
        line, inside = self.cut_lines.get(named, ""), joinask.enclosing(head)
        said = JOIN_STATEMENT.format(why=why, named=named, inside=f" inside `{inside}`" if inside else "",
                                     last=quoted(joinask.last_statement(head)), ends=self.redact(ends(head)),
                                     cut=(JOIN_CUT.format(line=quoted(line)) if line else "") +
                                         (JOIN_RETRY.format(start=JOIN_START[bool(line)]) if retry else ""),
                                     start=JOIN_START[bool(line)],
                                     through=self.through(named))
        return fit(self.redact(shown) + said, HELD_BYTES + HEAD_BYTES, tail)

    def join_module(self, named: str, module: str, why: str, tail: str) -> str:
        """The module shape (`JOIN_MODULE`): what was kept through its last complete top-level statement saved whole as
        a module of its own (`joinask.module_name`), the head dropped, and `named` asked anew importing it."""
        target = joinask.module_name(named, lambda path: (self.ws.local(path)[0] or Path("/")).exists())
        self.ws.write(target, module)
        self.drop_part(named)
        self.held = None
        self.log("continue", f"{named}: {len(module.encode()):,} B of what was kept saved as {target}; {named} asked "
                             "anew importing it")
        said = JOIN_MODULE.format(why=why, named=named, module=target, last=quoted(joinask.last_top(module)),
                                  opened=JOIN_OPENED[self.mode].format(named=named), stem=Path(target).stem,
                                  through=self.through(named))
        return fit(said, HELD_BYTES, tail)

    def head_run(self, named: str, head: str, why: str, tail: str) -> str:
        """Rule 2, the junction's asks spent: `named` written as its last script that ran clean where one did, else
        as what was kept to its last complete statement (`runner.parsing_prefix`), and run as it stands (main.py, or
        `named` where it is main.py's or runs alone), the next reply's write of it taken as definitions added or
        replacing their namesakes (`defs_due`: `defs`, or `rest` over what was kept, which takes statements after it
        too), asked in `JOIN_RAN`'s words after the run; the junction closed."""
        clean = self.clean_runs.get(named)
        text = clean[1] if clean is not None else runner.parsing_prefix(head)
        self.drop_part(named)
        self.held = None
        self.junctions.close(named)
        self.junctions.head_runs += 1
        self.junctions.defs_asked += 1
        self.defs_due[named] = ("defs" if clean is not None else "rest", self.counts["turns"])
        self.defs_files.add(named)
        self.kept_said.discard(named)
        wrote = self.ws.write(named, text)
        self.versions.setdefault(named, []).append((self.counts["turns"], sha256(text)))
        self.saved(named)
        self.log("continue", f"{named}: written as " + (f"turn {clean[0]}'s clean script" if clean is not None else
                                                         "what was kept") + f", {len(text.encode()):,} B, and run")
        target = named if named == MAIN_FILE or commands.has_main_block(text) else MAIN_FILE
        limit, line = self.run_limit()
        if limit < MIN_RUN_S or not (self.ws.dir / Path(target).name).is_file():
            run = f"(no run of {target}: " + (f"{max(self.left(), 0):.0f} s left)" if limit < MIN_RUN_S else
                                               "it does not exist yet)")
        else:
            self.counts["implicit_runs"] += 1
            shown, ran = self.ws.run(target, limit, self.seen, lambda done: self.tail(done, False))
            self.ran(ran, limit)
            self.saw(shown, ran)
            shown = self.audited(self.carded_after(shown, ran), ran)
            self.past_reserve(ran, limit, line)
            run = f"then, run for you: {shown}"
        self.head_ran = self.counts["turns"]
        said = JOIN_HEAD.format(why=why, named=named, what=JOIN_HELD[clean is not None].format(
            turn=clean[0] if clean is not None else 0))
        return "\n".join(part for part in (said, first_line(wrote), run, JOIN_RAN.format(
            named=named, one=JOIN_ONE[self.mode].format(named=named)), tail) if part)

    def defs_ask(self, named: str, tail: str) -> str:
        """Rule 3: a whole rewrite of a file cut before, cut after a failing run: nothing of it held, the file as it ran
        kept, and the fix asked as the definitions that change (`JOIN_DEFS`), the next reply's write of it spliced,
        a definition it lacks added (`defs_due`)."""
        last = self.last_ran.get(MAIN_FILE) or self.last_ran.get(named)
        gave = self.gave.get(last[0], "no run") if last is not None else "no run"
        self.defs_due[named] = ("defs", self.counts["turns"])
        self.defs_files.add(named)
        self.kept_said.discard(named)
        self.junctions.defs_asked += 1
        self.defs_mark = self.counts["runs"]
        self.log("continue", f"{named}: written again whole after a failing run and cut: nothing held, the fix asked "
                             "as definitions")
        trace.event("continue", path=named, kind="rewrite", ask="defs", failures=0)
        return fit(JOIN_DEFS.format(named=named, gave=self.redact(gave), one=JOIN_ONE[self.mode].format(named=named)),
                   commands.BOUNDS["write"], tail)

    def resumed(self, reply, found: list[Call], native: bool, turn: int) -> tuple[list[Call], bool]:
        """Lane review2, `CUT=continue`: the reply after a head was kept (`held`) read as the rest of its file. Its
        code (`rest_of`) is stitched under the head (`runner.stitch`; a reply that wrote the file again whole,
        `rewritten`, taken as it is) and given back as one write of the file, whole, in the
        reply's place before its other calls (a run, a finish), so it is saved, compiled and run as a write is; a
        stitched work/main.py whose top level calls nothing said at its run (`ENTRY`). Lane join1: the rest opening
        with the head's last lines, whitespace aside, has them once, as it has them (`runner.stitch`'s `overlap`,
        counted `cut_overlap`); a join that does not compile, a restart aside, is said with its junction and the head
        held again (`joined`). A reply the cap cut again gives back a cut write of the head and its complete lines,
        which `salvage` holds again. A reply holding no code is read as it came, the head dropped. (the calls, whether
        they are native)"""
        named, head = self.held
        self.held, self.joining = None, None
        code, taken, untagged = self.rest_of(reply, found, named)
        sent = continuation_read(head, code)       # lane redirect1: a tail, a re-send of the head, or prose
        self.counts[f"cut_{sent}"] += 1
        if sent == "resent" and self.redirect_armed and (self.at_cap(reply) or not rewritten(head, code)):
            rest = past_head(head, code)
            self.log("continue", f"{named}: the reply re-sent the kept head" + (
                ", whole: the repeat dropped before the stitch" if rest != code else ", leaving it part way"))
            code = rest
        if code is None:
            self.drop_part(named)
            self.log("continue", f"{named}: the reply held no code; the head dropped")
            trace.event("cut_stitch", path=named, read="no code")
            return found, native
        read = (named, MAIN_FILE) if untagged else (named,)     # the blocks read as the rest, as the parser named them
        others = [call for call in found if call is not taken and call.name not in ("cut", "unparsed")
                  and not (taken is None and call.name == "write" and self.ws.local(str(call.arguments.get(
                      "path", "")))[1] in read)]
        cut = self.at_cap(reply)                  # the cap's, as the old continuation read it: a block the reply
        #                                           left open below it is the rest, whole where the join compiles
        joined, how, rest = stitched(head, code, cut)
        repaired = self.rejoined(named, head, code, rest, cut, joined, how) if self.join_armed else None
        if repaired is not None:
            joined, how = repaired.joined, repaired.how
        text, whole = (joined + code[len(rest):], False) if cut else (joined, runner.compiles(joined))
        overlap = runner.repeated(head, rest) if how.startswith(runner.OVERLAP) else 0
        self.counts["cut_overlap"] += bool(overlap)
        self.counts["cut_join_reread"] += how.endswith(runner.REREAD)
        write = Call("write", {"path": named, "text": text}, taken.id if taken is not None else f"call_{turn}_0",
                     taken.raw if taken is not None else "{}", None, cut)
        self.log("continue", f"{named}: {len(head.encode()):,} B kept, the rest {len(code.encode()):,} B"
                             + (", cut again" if cut else "") + f"; the join {len(joined.encode()):,} B ({how})"
                             + ("" if cut else ", compiles" if whole else ", does not compile"))
        trace.event("cut_stitch", path=named, head_bytes=len(head.encode()), rest_bytes=len(code.encode()),
                    joined_bytes=len(joined.encode()), read=how, cut=cut, compiles=whole, overlap=overlap,
                    share=round(runner.share(joined), 3), **({"repair": repaired.kind} if repaired else {}))
        if cut and self.joinask_armed:            # lane joinask1: `hold` reads it as a rest cut again
            self.recut = named
        if not cut:
            self.drop_part(named)
            self.stitching = write
            self.counts["cut_stitched"] += whole
            if not whole and how != "restarted":        # lane join1: said with its junction, the head held again
                start = runner.unrepeated(head, overlap).count("\n") + 1 if overlap else \
                    head.rstrip("\n").count("\n") + 2         # the rest's first line in the join
                error, line = compile_error(joined)
                shown = {"lines": junction(joined, start, line), "opened": never_closed(joined, start)}
                self.joining = (named, head, JUNCTION.format(**shown, named=named,
                                                             through=self.through(named)),
                                not runner.compiles(head) and not unfinished(head),     # its error the head's own,
                                error, line, JUNCTION_SHOWN.format(**shown))            # above it
            if whole and named == MAIN_FILE and not runner.has_driver(joined):
                name = runner.entry(joined) or "main"
                self.entry = ENTRY.format(name=name)
                self.log("entry", f"the stitched script calls none of what it defines (`{name}`)")
        return [write, *others], native and taken is not None and bool(taken.id) and taken.raw != "{}"

    def rejoined(self, named: str, head: str, code: str, rest: str, cut: bool, joined: str,
                 how: str) -> rejoin.Repair | None:
        """Lane join2 of 10-16, `JOIN=repair`: today's join of `named` (`joined`, read `how`) repaired where
        agent/n11/rejoin.py's `repair` reads it otherwise, with the received lines of the statement the head's cut fell
        in and of the line it fell in (`held_tail`): a rest that starts again in the head's place (`join_restarts`);
        where today's join fails and the cut fell inside a statement, the reading with that statement's received lines
        kept, tried (`join_validated`), made where the rest goes on from the cut and taken where it compiles
        (`join_prefix_kept`). None where today's stands."""
        held = self.held_tail
        tail, partial = held[2:] if held is not None and held[:2] == (named, head) else ("", "")
        repaired, tried = rejoin.repair(head, tail, partial, code, rest, cut, joined, how)
        self.counts["join_validated"] += tried
        if repaired is None:
            if tried:
                self.log("continue", f"{named}: no join with the cut statement's received lines kept (the rest begins "
                                     "elsewhere, has ambiguous overlap, or that join does not compile either); "
                                     "today's stands")
            return None
        self.counts["join_restarts" if repaired.kind == "restart" else "join_prefix_kept"] += 1
        self.log("continue", f"{named}: the join repaired ({repaired.how}), {len(repaired.joined.encode()):,} B in "
                             f"place of today's {len(joined.encode()):,} B ({how})")
        return repaired

    def joined(self, args: dict, tail: str) -> str:
        """Lane join1: the write of a join that does not compile (`joining`), said as a write is with its junction
        after the error's lines (`JUNCTION`), the head held again as `named` + `PART` so the next reply is read as
        its rest (`resumed`). Lane join2: the join refused as the file already saved is said so with its error, never
        a run offered, and its junction, the head held again (`JOIN_REFUSED`, counted `cut_join_refused`); where the
        error stands in a head broken above its end, the file asked whole and the head dropped (`KEPT_BROKEN`), once a
        file. A write refused otherwise, as it came."""
        named, head, said_after, broken, error, line, shown = self.joining
        self.joining = None
        repeats = self.counts["write_repeats"]
        said = self.written(args["path"], args["text"], "", whole=True)
        refused = self.counts["write_repeats"] > repeats
        if not refused and not (said.startswith("wrote ") and "does not compile" in first_line(said)):
            return f"{said}\n{tail}" if tail else said
        self.counts["cut_join_failed"] += 1
        self.counts["cut_join_refused"] += refused
        self.resent = None              # lane resend1: a refused join's run line is today's
        if refused:
            said = fit(JOIN_REFUSED.format(named=named, error=error) + (
                f"\n{commands.around(commands.unfenced(args['text']), line)}" if line else ""), commands.BOUNDS["write"])
        if self.joinask_armed:      # lane joinask1: the changed ask; a head broken above its end kept to what compiles
            error, shown, said = (join_unlocated(text) for text in (error, shown, said))
            kind = "refused" if refused else "junction"
            kept = runner.parsing_prefix(head) if broken else head
            asked = self.join_failed(named, kept, kind, JOIN_WHY[kind].format(named=named, error=error or "it does "
                                                                              "not compile"), "", shown)
            return f"{said}\n{asked}" + (f"\n{tail}" if tail else "")
        if broken and named not in self.said_broken:
            self.said_broken.add(named)
            self.log("continue", f"{named}: the join does not compile, its error in what was kept; the file asked "
                                 "whole, the head dropped")
            said = f"{said}\n{KEPT_BROKEN.format(named=named)}"
        else:
            self.ws.write(f"{named}{PART}", head)
            self.held = (named, head)
            self.log("continue", f"{named}: the join does not compile" + (", refused as the file saved" if refused
                                                                          else "")
                     + "; its junction said, the head held again")
            said = f"{said}\n{fit(self.redact(said_after), HELD_BYTES)}"
        return f"{said}\n{tail}" if tail else said

    def rest_of(self, reply, found: list[Call], named: str) -> tuple[str | None, Call | None, bool]:
        """The code a reply gives as `named`'s rest, the call that carried it, and whether a block naming no file was
        read: the text of its first write (or cut call) of `named`, or naming no path; else its ```python blocks' code
        that name no other file, in order; else, where it holds no block, the code it opens with
        (`continuation_code`). (None, None, False) where it holds none."""
        for call in found:
            if call.name not in ("write", "cut", "unparsed") or (call.problem and not call.cut):
                continue
            arguments = self.recovered(call)
            path, text = arguments.get("path"), arguments.get("text")
            if text and (not path or self.ws.local(path)[1] == named):
                return opened_fence(text, closed=not call.cut), call, False
        items = commands.fenced(runner.THINK.sub("", reply.text or ""))
        blocks = [item for item in items if isinstance(item, commands.Fenced) and not item.void
                  and (item.path is None and (item.tag in commands.PYTHON or not item.tag)
                       or item.path is not None and self.ws.local(item.path)[1] == named)]
        if blocks:
            return ("\n".join("\n".join(block.lines) for block in blocks), None,
                    any(block.path is None for block in blocks))
        if not any(isinstance(item, commands.Fenced) for item in items):
            code = continuation_code(reply.text or "")
            if code.strip():
                return code, None, False
        return None, None, False

    def recovered(self, call: Call) -> dict:
        """A cut call's arguments as far as they got: its own where they hold a text, else its raw JSON's
        (`commands.cut_arguments`), else the reply's text's (`cut_from_text`)."""
        return call.arguments if isinstance(call.arguments, dict) and call.arguments.get("text") else \
            (commands.cut_arguments(call.wrote) if call.wrote else {}) or cut_from_text(self.reply_text)

    def implicit(self, found: list[Call], fenced: bool = False) -> int | None:
        """The index of the reply's last call after which the harness may run a file (`after_write`): a `write` of a
        `.py` file under work/ that no `run` or `finish` follows; None where no run follows the reply's calls. In a
        fenced reply a text file's write after it, which never runs, is set aside (`text_write`)."""
        last = next((k for k in range(len(found) - 1, -1, -1) if found[k].name in ("write", "run", "finish")
                     and not found[k].problem and not found[k].cut and not (fenced and text_write(found[k]))), None)
        if last is None or found[last].name != "write" or not str(found[last].arguments.get("path", "")).endswith(".py"):
            return None
        return last

    def finish_writes(self, found: list[Call], fenced: bool = False) -> dict[int, int]:
        """Lane finishrun1, `FINISHRUN=on`: each `finish` of the reply that follows a `write` of a `.py` file with no
        `run` or earlier `finish` between (in a fenced reply a text file's write between set aside, as `implicit`
        sets it), by its index, mapped to that write's: the file the harness runs before the finish (`after_write`'s
        `finishing`). Empty where the switch is not armed."""
        if not self.finishrun_armed:
            return {}
        pairs, last = {}, None
        for number, call in enumerate(found):
            if call.problem or call.cut or call.name not in ("write", "run", "finish") or (fenced and text_write(call)):
                continue
            if call.name == "finish" and last is not None and found[last].name == "write" \
                    and str(found[last].arguments.get("path", "")).endswith(".py"):
                pairs[number] = last
            last = number
        return pairs

    def after_write(self, said: str, path: str, finishing: bool = False) -> str:
        """A reply's last write's observation with the run the harness makes after it (lane n11d), the budget line at
        the end: no run where the write was refused or does not compile, nor while work/main.py is missing (the
        imports still missing named); work/main.py where it changed since its last run (else said unchanged, not run);
        a file written other than main.py with an `if __name__ == "__main__":` block that main.py does not import,
        itself, said so (lane clock1: one main.py imports is a module, and main.py runs). After a module's write (lane
        n11e's addendum: a live unit wrote five modules one a turn and never ran again), nothing runs while
        work/main.py imports a module that is nowhere (named; work/'s own modules imported package-style said as such,
        lane clock1); and work/main.py runs where it imports the module written (directly or through a module under
        work/) and its last run was not clean, the observation naming the module (`module_runs` counts every run of
        main.py a module's write makes), and says so where main.py does not import it (lane clock1). The run's limit
        is `run_limit`'s. Lane finishrun1 (`finishing`: a write a `finish` follows, `finish_writes`): the run's
        observation without the budget line, which the reply's last call carries, counted `finish_ran`; where no run
        is made, the write's observation as it stood, nothing added and nothing counted."""
        main, mainpath = self.ws.dir / "main.py", f"{commands.WORK}/main.py"
        here, named = self.ws.local(path)
        staged, self.staged = self.staged is not None and self.staged == named, None
        if staged:      # lane join2 of 10-16 (d): the file kept, work/main.py run where it changed since its last run
            here, named = main, mainpath
        if self.head_ran == self.counts["turns"]:      # lane joinask1: `head_run` ran it in this write's turn
            self.head_ran = -1
            return said if finishing else f"{said}\n{self.budget()}"

        def unrun(line: str) -> str:
            return said if finishing else f"{said}\n{line}\n{self.budget()}"
        if self.resent is not None and self.resent[0] == "read":     # lane resend1: the finish's, its budget line
            return said                                              # after it
        if (said.startswith(("write refused", "not written")) or "does not compile" in first_line(said)) and not staged:
            return unrun(self.unrun_line())
        if not main.is_file():
            missing, note = self.ws.unwritten(self.ws.missing_modules())
            return unrun(f"(no run: {mainpath} does not exist yet" + (
                f"; the files import these modules not yet under {commands.WORK}/: "
                + ", ".join(f"{name}.py" for name in missing[:12]) + "; write them, then work/main.py, which runs"
                if missing else "; write it next, and it runs") + (f"; {note}" if note else "") + ")")
        script = here.read_text(encoding="utf-8", errors="replace") if here is not None and here.is_file() else ""
        top = named != mainpath and named.count("/") == 1          # a module main.py can import by its name
        local, missing = self.ws.reached() if named != mainpath else (set(), [])
        imports = top and Path(named).stem in local
        alone = named != mainpath and commands.has_main_block(script) and not imports
        target = named if alone else mainpath
        module = target == mainpath and named != mainpath
        why = SELF_RUN.format(named=named, main=mainpath) if alone and top else ""
        if module:
            missing, note = self.ws.unwritten(missing, mainpath)
            if missing or note:
                return unrun("(no run: " + "; ".join(part for part in (
                    f"{mainpath} imports these modules not yet under {commands.WORK}/: "
                    + ", ".join(f"{name}.py" for name in missing[:12]) + f"; write them, and {mainpath} runs"
                    if missing else "", note) if part) + ")")
            last = self.last_ran.get(mainpath)
            if imports and last is not None and not last[2]:
                why = f" ({mainpath} imports {named}, and its last run, turn {last[0]}, was not clean)"
        if target == mainpath and not why:
            last = self.last_ran.get(mainpath)
            if last is not None and last[1] == sha256(main.read_text(encoding="utf-8", errors="replace")):
                self.counts["unchanged_main"] += not finishing
                hint = NOT_IMPORTED.format(named=named, main=mainpath) if top and not imports else \
                    f"; {self.run_form(mainpath)} runs it with {named}" if named != mainpath else ""
                return unrun(f"({mainpath} unchanged since turn {last[0]}: not run{hint})")
            if module:
                why = f" ({mainpath} changed since its last run; " + (
                    f"it does not import {named})" if top and not imports else f"{named} written)")
        limit, line = self.run_limit()
        if limit < MIN_RUN_S:
            return unrun(f"(no run of {target}: {max(self.left(), 0):.0f} s left)")
        self.counts["implicit_runs"] += 1
        self.counts["module_runs"] += module
        if finishing:
            self.counts["finish_ran"] += 1
            self.log("finishrun", f"{target} run before the finish that follows {named}")
        shown, ran = self.ws.run(target, limit, self.seen, lambda done: self.tail(done, not finishing))
        self.ran(ran, limit)
        self.saw(shown, ran)
        shown = self.audited(self.carded_after(shown, ran), ran)
        self.past_reserve(ran, limit, line)
        return f"{said}\nthen, run for you{why}: {shown}"

    def unrun_line(self) -> str:
        """The run line after a write not saved whole and compiling (`after_write`); lane resend1, under `finish`,
        an unchanged one's said true: the file unchanged since the turn it matched (`RESEND_SINCE`), and after a
        clean run the named outputs still missing (`RESEND_MISSING`)."""
        if self.resent is None or self.resent[0] != "refused":
            return NO_RUN
        _, named, turn, missing = self.resent
        if missing:
            return RESEND_MISSING.format(named=named, turn=turn, missing=", ".join(missing[:12]),
                                         run=self.run_form(named))
        return RESEND_SINCE.format(named=named, turn=turn)

    def ran(self, ran: commands.Ran | None, cap: float) -> None:
        """A run counted and logged: its turn's line (`gave`), its file's sha at the run (`last_ran`), its digest's
        bytes and whether it met a sanity finding."""
        turn = self.counts["turns"]
        if ran is None:
            self.log("run", "refused")
            self.gave[turn] = "the run was refused"
            self.runs_at[turn] = None
            return
        self.counts["runs"] += 1
        self.run_seconds.append(ran.seconds)    # lane pace1: a run's expected seconds
        if ran.shimmed:                         # lane clock1 of 10-13: the shim's kinds that fired in it
            self.shim_fired.update(ran.shimmed)
            self.counts["shim_runs"] += 1
            self.log("shim", ", ".join(ran.shimmed))
        if ran.path == MAIN_FILE:
            self.last_main = ran
        seen_before = ran.identity is not None and ran.identity in self.seen
        if ran.identity is not None:
            self.seen.add(ran.identity)
        here = self.ws.local(ran.path)[0]
        if here is not None and here.is_file():
            self.last_ran[ran.path] = (turn, sha256(here.read_text(encoding="utf-8", errors="replace")),
                                       ran.clean and not ran.findings)
            if ran.exit == 0 and not ran.error:     # lane joinask1: the script rule 2 runs again where it can
                self.clean_runs[ran.path] = (turn, here.read_text(encoding="utf-8", errors="replace"))
        if self.first_outputs is None and ran.exit == 0 and not ran.error:     # lane firstout1, under either setting
            self.first_outputs = len(set(ran.wrote) & set(self.named_files or ()))
        self.run_failed = ran.exit != 0 or bool(ran.error)
        if ran.exit == 0:
            self.digests.append(len(ran.digest.encode("utf-8")))
            self.ruled_after(ran)
        self.found(ran.findings, ran.rules)
        self.called_ran(ran)                    # lane called1: its finding, said beside the findings
        if ran.unread:                          # lane tools1: the unread-parameter finding said after this run
            self.counts["param_unread"] = 1
        for key, line in (("span_finding", ran.span), ("scale_finding", ran.scale)):     # lane convcard1's two
            if line:
                self.counts[key] = 1
                self.log("finding", f"{key.split('_')[0]}: {line[:200]}")
        self.last_clean = ran.clean and not ran.findings and not (ran.reconciled and (ran.recon or ran.witness))
        #                                                 lane pace1: the short clock's ask reads it; lane recon1: a
        #                                                 reconciliation's or derive1's finding standing is not clean
        if self.review_from is not None and ran.wrote:  # a run writing nothing leaves it as it was
            self.review_last = (ran.clean and not ran.findings, frozenset(ran.wrote), contract.snapshot(self.out))
        self.gave[turn] = ("stopped at its limit" if ran.exit is None else ran.error.split("\n", 1)[0][:160]
                           if ran.error else f"exit {ran.exit}, {len(ran.failures)} check(s) failing" if ran.failures
                           else f"exit 0, {len(ran.findings)} sanity finding(s)" if ran.findings
                           else f"exit {ran.exit}, the checks pass")
        self.runs_at[turn] = ran
        self.outcome[ran.path] = (turn, self.gave[turn])
        if self.form in REPEAT_FORMS:           # lane repeat1: the program as run and its failure, either setting
            self.repeat_met(self.repeats.ran(repeat.program(self.ws.dir), repeat.signature(ran.exit, ran.error,
                                                                                          ran.failures)), ran.path, "run")
        self.log("run", ("stopped at its limit" if ran.exit is None else f"exit {ran.exit}")
                 + f" after {ran.seconds:.1f} s of {cap:.0f}; {len(ran.failures)} check(s) failing"
                 + (f"; {len(ran.findings)} sanity finding(s)" if ran.findings else "")
                 + (f"; {ran.error.split(':', 1)[0][:60]}" if ran.error else "")
                 + ("; the same error as an earlier run" if seen_before else ""))
        if self.resort.holding and fallback.graded(self.out, self.ws.named):     # lane fallback1: for good
            self.resort.holding = False
            self.log("fallback", "the reserve released: the output holds a file to grade")

    def finish(self, last: bool, reason: str = "", resent: bool = False) -> tuple[str, bool]:
        """`finish`: today's rule (`counted`), but under lane finishfail1's `FINISHFAIL=on` a finish whose run did not
        exit 0 (`shown_run`: the last run the House was shown, `FINISHRUN`'s before it included) is turned away
        (`turns_away`): answered with that run's observation under `FINISHFAIL_SAID` and not counted among the
        finishes, only where the current output fails the checks, while a request and `ASK_S` remain, at most
        `FINISHFAIL_MAX` times a unit; the next finish on a failing run is counted as today's (`finishfail_ended` where
        it ends the unit), and with no request or time left one ends the unit as today. Passing checks keep today's rule, even after a diagnostic crash. The finish an
        unchanged re-send is read as (`resent`, lane resend1's `RESEND=finish`, read before this rule) is never turned away."""
        shown = self.shown_run if self.finishfail_armed and not resent else None
        if shown is None or not self.ws.checks():
            return self.counted(last, reason)
        turn, named, exit_, said = shown
        if turns_away(exit_, self.requests_left(), self.left(), self.counts["finishfail_turned"]):
            self.counts["finishfail_turned"] += 1
            self.log("finish", f"not taken: the last run, {named} at turn {turn}, failed; turned away "
                               f"{self.counts['finishfail_turned']} of {FINISHFAIL_MAX}")
            trace.event("finishfail", turn=turn, path=named, exit=exit_, turned=self.counts["finishfail_turned"],
                        left_s=round(self.left(), 1))
            return fit(f"{FINISHFAIL_SAID.format(named=named)}\n{said}", commands.BOUNDS["run"],
                       self.budget() if last else ""), False
        bound = exit_ != 0 and self.requests_left() >= 1 and self.left() >= ASK_S
        said, ended = self.counted(last, reason)
        if bound and ended:
            self.counts["finishfail_ended"] = 1
            self.log("finish", f"on a failing run, counted after {FINISHFAIL_MAX} turned away: the task ends")
        return said, ended

    def finishcheck_turn(self, tail: str) -> str | None:
        """Lane finishcheck1, `FINISHCHECK=on`: a finish today's rules take (`counted`'s, `FINISHFAIL`'s before it)
        checked against the instruction's contract with no request (`finishcheck_findings`); where a finding of
        `finishcheck.ARMED` stands, a request and `ASK_S` remain and none was turned before in the unit
        (`FINISHCHECK_MAX`), the findings answer it under `FINISHCHECK_SAID` and it is not counted; None where the
        finish stands. `tail` is the budget line where the finish was the reply's last call."""
        if not self.finishcheck_armed or self.counts["finishcheck_turned"] >= FINISHCHECK_MAX:
            return None
        if self.requests_left() < 1 or self.left() < ASK_S:
            return None
        findings = self.finishcheck_findings()
        if not findings:
            return None
        self.counts["finishcheck_turned"] += 1
        kinds = finishcheck.flags(findings)
        self.log("finish", f"not taken: the contract's findings stand ({kinds}); turned away "
                           f"{self.counts['finishcheck_turned']} of {FINISHCHECK_MAX}")
        trace.event("finishcheck", flags=kinds, turned=self.counts["finishcheck_turned"], left_s=round(self.left(), 1))
        lines = "\n".join(f"- {self.redact(finding.line)}" for finding in findings[:finishcheck.SHOWN])
        return fit(FINISHCHECK_SAID.format(lines=lines), commands.BOUNDS["finish"], tail)

    def finishcheck_findings(self, out: Path | None = None, counted: bool = True) -> list[finishcheck.Finding]:
        """Lane finishcheck1: the contract's findings over the output as it stands (agent/n11/finishcheck.py): the
        instruction read once, at the first finish checked, for the names it ties to an output (`finishcheck.TIE`),
        the input files' names and the names they hold left out; the kinds found kept for `finishcheck_flags`. A
        reader's fault is logged and never raised into the run. Lane repairpair1's keep rule reads another output
        (`out`) and keeps no kind (`counted`)."""
        try:
            if self.finishcheck_names is None:
                task = self.ws.task
                data = Path(task.data_dir) if task is not None else None
                files = [path.name for path in data.rglob("*") if path.is_file()] if data and data.is_dir() else []
                held = finishcheck.input_names(data) if data is not None else frozenset()
                self.finishcheck_names = finishcheck.stated(task.instruction if task is not None else "",
                                                            self.ws.named, files, finishcheck.TIE, held)
            findings = finishcheck.check(self.out if out is None else out, self.ws.named, self.finishcheck_names)
        except Exception as exc:    # noqa: BLE001 - the contract is advice; the finish stands where it fails
            self.log("finishcheck", f"unavailable ({type(exc).__name__}): the finish stands")
            return []
        if counted:
            self.finishcheck_found.update(finding.kind for finding in findings)
        return findings

    def saw(self, said: str, ran: commands.Ran | None) -> None:
        """Lane finishfail1: a run the House is shown (`execute`'s, `after_write`'s, a head run's, request 1's) kept
        with its observation, the budget line `tail` ended it with dropped, for a finish on it (`finish`); a run refused
        keeps the one before, and the runs the House is not shown (the library pass's, its output put back where it
        fails; the last call's, which ends the unit) are not kept."""
        if ran is not None:
            budget = f"\n{self.tail_budget}" if self.tail_budget else ""
            self.shown_run = (self.counts["turns"], ran.path, ran.exit, said.removesuffix(budget) if budget else said)
            self.snapshot_seen(ran)
            self.derive_seen(ran)
            self.contract_seen(ran)
            self.recon_seen(ran)

    def derive_read(self, reply, found: list[Call]) -> None:
        """Lane derive1, under either setting in the loop's forms: the loop's first reply read once for a derivation
        (`derive.opened`: its text before its first code, or where it has none the leading comments of the `.py` file
        it writes), and the unit's first `.py` write read once for its assertions and top-level functions; logged
        where the switch is armed."""
        if self.form not in DERIVE_FORMS:
            return
        script = next((str(call.arguments.get("text") or "") for call in found if call.name == "write"
                       and isinstance(call.arguments, dict) and str(call.arguments.get("path", "")).endswith(".py")),
                      None)
        if self.derive_said is None:
            self.derive_said = derive.opened(reply.text or "", script or "")
            if self.derive_armed:
                self.log("derive", "the first reply " + ("opened" if self.derive_said else "did not open")
                         + " with a derivation")
        if self.derive_script is None and script is not None:
            self.derive_script = (derive.checks(script), derive.functions(script))
            if self.derive_armed:
                self.log("derive", f"the first script: {self.derive_script[0]} assertion(s), "
                                   f"{self.derive_script[1]} top-level function(s)")

    def derive_seen(self, ran: commands.Ran) -> None:
        """Lane derive1: a run the House is shown (`saw`'s) whose observation carried the harness's own checks'
        findings, counted, logged and traced."""
        if ran.witness:
            self.counts["witness_runs"] += 1
            self.counts["witness_findings"] += len(ran.witness)
            self.witness_kinds.update(each.check for each in ran.witness)
            self.log("witness", f"{ran.path}: " + "; ".join(each.line for each in ran.witness)[:400])
            trace.event("witness", path=ran.path, checks=[each.check for each in ran.witness])

    def contract_read(self, instruction: str, named: tuple[str, ...]) -> None:
        """Lane contract1, under any setting in the loop's forms: the contract the instruction states, read once a unit
        (props.terms, beside today's `stated`), its terms counted and, where armed, logged; a reader's fault leaves
        none, logged, never raised."""
        if self.form not in CONTRACT_FORMS:
            return
        try:
            self.contract_terms = props.terms(instruction, named, self.stated)
        except Exception as exc:  # noqa: BLE001 - the contract is advice; the unit runs without it
            self.contract_terms = props.Terms()
            self.log("contract", f"unread ({type(exc).__name__}): no term")
        self.counts["contract_terms"] = len(self.contract_terms.terms)
        if self.contract_armed:
            kinds = Counter(term.kind for term in self.contract_terms.terms)
            self.log("contract", f"{len(self.contract_terms.terms)} term(s) stated: "
                                 + (", ".join(f"{kind} {kinds[kind]}" for kind in props.TERM_KINDS if kinds[kind])
                                    or "none")
                                 + (f"; waived: {', '.join(sorted(self.contract_terms.waived))}"
                                    if self.contract_terms.waived else ""))

    def contract_seen(self, ran: commands.Ran) -> None:
        """Lane contract1: a run the House is shown (`saw`'s) whose observation carried the contract's findings,
        counted, logged and traced."""
        if ran.contract:
            self.counts["contract_runs"] += 1
            self.counts["contract_said"] += len(ran.contract)
            self.contract_kinds.update(each.kind for each in ran.contract)
            self.log("contract", f"{ran.path}: " + "; ".join(each.line for each in ran.contract)[:400])
            trace.event("contract", path=ran.path, kinds=[each.kind for each in ran.contract])

    def contract_turn(self, tail: str) -> str | None:
        """Lane contract1, `CONTRACT=refuse`: a finish the checks pass, today's refusals not taken, checked against the
        instruction's contract with no request (the kinds of `stated_contract.KEPT` over the output as it stands);
        where a finding stands, a request and `ASK_S` remain and none was turned before in the unit (`CONTRACT_MAX`),
        the findings answer it under `CONTRACT_SAID` and it is not counted; None where the finish stands. `tail` is
        the budget line where the finish was the reply's last call."""
        if self.contract_setting != "refuse" or not self.contract_armed or self.contract_terms is None \
                or self.counts["contract_refused"] >= CONTRACT_MAX:
            return None
        if self.requests_left() < 1 or self.left() < ASK_S:
            return None
        try:
            found = stated_contract.run(self.contract_terms, self.out)
        except Exception as exc:    # noqa: BLE001 - the contract is advice; the finish stands where it fails
            self.log("contract", f"unavailable ({type(exc).__name__}): the finish stands")
            return None
        if not found:
            return None
        self.counts["contract_refused"] += 1
        self.contract_kinds.update(each.kind for each in found)
        kinds = ",".join(dict.fromkeys(each.kind for each in found))
        self.log("finish", f"not taken: the contract's findings stand ({kinds}); turned away "
                           f"{self.counts['contract_refused']} of {CONTRACT_MAX}")
        trace.event("contract_refused", kinds=kinds.split(","), left_s=round(self.left(), 1))
        lines = "\n".join(f"- {self.redact(each.line)}" for each in found[:stated_contract.SAID])
        return fit(CONTRACT_SAID.format(lines=lines), commands.BOUNDS["finish"], tail)

    def recon_seen(self, ran: commands.Ran) -> None:
        """Lane recon1: a run the House is shown (`saw`'s) that the reconciliation read (`Ran.reconciled`): its
        findings counted, logged and traced (derive1's beside them, `derive_seen`'s), and those the last read run (or a
        refusal) said and this one does not, counted acted (`recon_acted`)."""
        if not ran.reconciled:
            return
        now = frozenset(recon.ident_of(each) for each in (*ran.witness, *ran.recon))
        if self.recon_standing:
            gone = self.recon_standing - now
            self.counts["recon_acted"] += len(gone)
            if gone:
                self.log("recon", f"{ran.path}: {len(gone)} finding(s) said before are gone")
        self.recon_standing = now
        trace.event("recon", path=ran.path, checks=[each.check for each in ran.recon], held=sorted(now))
        if ran.recon:
            self.counts["recon_runs"] += 1
            self.counts["recon_said"] += len(ran.recon)
            self.recon_kinds.update(each.check for each in ran.recon)
            self.log("recon", f"{ran.path}: " + "; ".join(each.line for each in ran.recon)[:400])

    def recon_same(self, ran: commands.Ran | None) -> str:
        """Lane recon1: a read run whose findings include one the last read run said, the scripts under work/ changed
        since, is answered in the same-frame form (`recon.SAME`, the quantities it names): the change did not reach the
        computation; "" otherwise. The scripts' digest is kept at each read run."""
        if ran is None or not ran.reconciled:
            return ""
        scripts = hashlib.sha256(b"".join(path.relative_to(self.ws.dir).as_posix().encode() + b"\0" + path.read_bytes()
                                          for path in sorted(self.ws.dir.rglob("*.py")) if path.is_file())).hexdigest()
        changed, self.recon_scripts = self.recon_scripts is not None and scripts != self.recon_scripts, scripts
        standing = self.recon_standing or frozenset()
        same = [each for each in (*ran.witness, *ran.recon) if recon.ident_of(each) in standing]
        if not changed or not same:
            return ""
        self.counts["recon_same"] += 1
        names = list(dict.fromkeys(recon.quantity(each.line) for each in same if recon.quantity(each.line)))
        self.log("recon", f"{ran.path}: {len(same)} finding(s) stand after the scripts changed")
        trace.event("recon_same", path=ran.path, held=sorted(recon.ident_of(each) for each in same))
        return recon.SAME.format(names=", ".join(names[:recon.NAMED]) or "the named quantity")

    def recon_turn(self, tail: str) -> str | None:
        """Lane recon1, `RECON=refuse`: a finish the checks pass, today's refusals not taken, reconciled with no request
        (the checks of `recon.KEPT` over the output as it stands); where a finding stands, a request and `ASK_S` remain
        and none was turned before in the unit (`RECON_MAX`), the findings answer it under `RECON_SAID` and it is not
        counted; None where the finish stands. `tail` is the budget line where the finish was the reply's last call."""
        if not self.recon_refusing or self.counts["recon_refused"] >= RECON_MAX:
            return None
        if self.requests_left() < 1 or self.left() < ASK_S:
            return None
        witnessed, reconciled = self.ws.reconciliation()
        found = (*witnessed, *reconciled)
        if not found:
            return None
        self.counts["recon_refused"] += 1
        self.recon_kinds.update(each.check for each in reconciled)
        self.recon_standing = frozenset(recon.ident_of(each) for each in found)
        kinds = ",".join(dict.fromkeys(each.check for each in found))
        self.log("finish", f"not taken: the reconciliation's findings stand ({kinds}); turned away "
                           f"{self.counts['recon_refused']} of {RECON_MAX}")
        trace.event("recon_refused", checks=kinds.split(","), left_s=round(self.left(), 1))
        lines = "\n".join(f"- {self.redact(each.line)}" for each in (*witnessed[:witness.SAID], *reconciled))
        return fit(RECON_SAID.format(lines=lines), commands.BOUNDS["finish"], tail)

    def guards_turn(self, tail: str) -> str | None:
        """The finish guards after a finish the checks pass and today's refusals do not take: lane finishcheck1's alone
        (`finishcheck_turn`, today's) where no other refuses; under `CONTRACT=refuse` or `RECON=refuse`, or where lane
        audit1's audit holds the finish (`audit_holding`), every armed guard asked with its findings over the output as
        it stands, and those that turn the finish away joined as one refusal, each one's allowance spent (the contract
        review's fix: a finish two guards would refuse is refused once, both quoted); None where the finish stands."""
        refusing = self.contract_setting == "refuse" and self.contract_armed
        if not (refusing or self.recon_refusing or self.audit_holding):
            return self.finishcheck_turn(tail)
        parts = [part for part in (self.finishcheck_turn(""), self.contract_turn("") if refusing else None,
                                   self.recon_turn(""), self.audit_turn("")) if part]
        return fit("\n\n".join(parts), commands.BOUNDS["finish"], tail) if parts else None

    def audited(self, said: str, ran: commands.Ran | None) -> str:
        """Lane audit1, `AUDIT=on`: a run shown in the loop (`execute`'s, `after_write`'s, `head_run`'s) that is the
        unit's first clean one (exit 0, no error, the checks passing: every named output present) answered with the
        audit after its observation (`audit.section` over the task's text, in the reply's form), the budget line `said`
        ended with kept last, where `audit.FLOOR_S` of the loop's clock (`left`) and a request remain; else the audit is
        skipped, logged, traced and counted, never said. Once a unit, said or skipped."""
        if not self.audit_armed or self.audit_met or ran is None or not ran.clean:
            return said
        self.audit_met = True
        turn, left, requests = self.counts["turns"], self.left(), self.requests_left()
        if left < audit.FLOOR_S or requests < 1:
            self.counts["audit_skipped"] = 1
            self.log("audit", f"skipped at turn {turn}: " + (f"{max(left, 0):.0f} s left, under {audit.FLOOR_S:.0f} s"
                                                              if left < audit.FLOOR_S else "no request left"))
            trace.event("audit_skipped", turn=turn, left_s=round(left, 1), requests=requests)
            return said
        text, names = audit.section(self.unit.instruction if self.unit else "", self.mode)
        self.audit_at = turn
        self.counts["audit_shown"], self.counts["audit_items"] = 1, len(audit.FIXED) + len(names)
        self.log("audit", f"said after {ran.path}'s clean run at turn {turn}: outputs, method"
                          + "".join(f", {name}" for name in names) + f"; {left:.0f} s left")
        trace.event("audit", turn=turn, path=ran.path, items=["outputs", "method", *names], left_s=round(left, 1))
        budget = f"\n{self.tail_budget}" if self.tail_budget and said.endswith(f"\n{self.tail_budget}") else ""
        return f"{said.removesuffix(budget)}\n{text}{budget}"

    def audit_turn(self, tail: str) -> str | None:
        """Lane audit1: a finish in the reply whose run said the audit (`audit_holding`), with a request and `ASK_S`
        left, held once a unit (`AUDIT_MAX`) with `audit.HELD` and not counted; None where the finish stands. `tail` is
        the budget line where the finish was the reply's last call."""
        if not self.audit_holding or self.requests_left() < 1 or self.left() < ASK_S:
            return None
        self.counts["audit_held"] += 1
        self.log("finish", f"not taken: held for the audit's answers; turned away {self.counts['audit_held']} of "
                           f"{AUDIT_MAX}")
        trace.event("audit_held", turn=self.counts["turns"], left_s=round(self.left(), 1))
        return fit(audit.HELD, commands.BOUNDS["finish"], tail)

    def audit_read(self, reply, found: list[Call]) -> None:
        """Lane audit1: the reply after the audit's (`audit_at`) read once for its answers (`audit.answers` against the
        task's text) and for a write among its calls; counted, logged and traced."""
        read = audit.answers(reply.text or "", self.unit.instruction if self.unit else "")
        rewrote = any(call.name == "write" and not call.problem and not call.cut for call in found)
        self.counts["audit_answered"], self.counts["audit_quoted"] = read.answered, read.quoted
        self.counts["audit_fix"], self.counts["audit_rewrote"] = read.fix, int(rewrote)
        self.log("audit", f"the reply after it: {read.answered} answer(s) in the form, {read.quoted} quoting the task, "
                          f"{read.fix} FIX; " + ("a write followed" if rewrote else "no write"))
        trace.event("audit_answers", turn=self.counts["turns"], answered=read.answered, quoted=read.quoted,
                    fix=read.fix, rewrote=int(rewrote))

    def snapshot_seen(self, ran: commands.Ran) -> None:
        """Lane snapshot1: a run the House is shown (`saw`'s) whose answer carried the state at its failure, or where
        one was due and none was said (the hook wrote nothing, or it did not fit its budget), counted, logged and
        traced."""
        shown = ran.snapshot
        if shown.text:
            self.counts["snapshot_shown"] += 1
            self.snapshot_sizes.append(shown.size)
            self.log("snapshot", f"{ran.path}: the state at the failure said, {shown.size} {measure.UNIT.name}")
            trace.event("snapshot", path=ran.path, size=shown.size, unit=measure.UNIT.name)
        elif shown.dropped:
            self.counts["snapshot_dropped"] += 1
            self.log("snapshot", f"{ran.path}: dropped ({shown.dropped})")
            trace.event("snapshot", path=ran.path, dropped=shown.dropped)

    def carded_after(self, said: str, ran: commands.Ran | None) -> str:
        """Lane cardfail1, `CARDFAIL=on`: a run the House is shown (`saw`'s four) that failed in a library function
        (`Ran.failed_in`, read only where armed: `Workspace.cardfail`) said with that function's card after its words, the line `CARDFAIL_SAID`
        over it and the budget line `tail` ended it with kept last; never a function whose card the House has
        (`cardfail_carded`, an earlier failure's; `cardfail_looked`, a `lib` answer's; the first message's `cards`),
        under `again` an earlier failure's alone. `saw` keeps the
        run without it, so a finish turned away on it does not card it again."""
        function = ran.failed_in if ran is not None else None
        if function is None or function.name in self.cardfail_carded:
            return said
        if self.cardfail_setting != "again" and (function.name in self.cardfail_looked
                                                 or function.typed in self.cards.split("\n")):
            return said
        self.cardfail_carded.add(function.name)
        self.counts["cardfail_shown"] += 1
        line = CARDFAIL_SAID.format(call=function.call)
        card = library.answer(function, commands.BOUNDS["lib"] - commands.size(line) - 1)
        self.log("cardfail", f"{function.call}: its card after the failed run of {ran.path}")
        trace.event("cardfail", path=ran.path, function=function.name, shown=self.counts["cardfail_shown"])
        budget = f"\n{self.tail_budget}" if self.tail_budget and said.endswith(f"\n{self.tail_budget}") else ""
        return f"{said.removesuffix(budget)}\n{line}\n{card}{budget}"

    def counted(self, last: bool, reason: str = "") -> tuple[str, bool]:
        """Today's `finish`: the checks and the findings over the output as it stands. The checks failing, the first
        finish with a request left shows them (and any finding) and the loop goes on, a second ends the unit. The checks
        passing with a finding standing, one finish is refused (a review turn: fix it, or finish again with `reason`,
        "accept: <why>"), once a unit; the next ends it, its findings accepted; with no finding the first ends it (the
        verify turn's second route left with lane prune2). A refusal is made only with a request and `ASK_S` left."""
        self.finishes += 1
        failures = self.ws.checks()
        _, findings = self.ws.sanity()
        tail = self.budget() if last else ""
        ruled = self.ws.ruled if self.rules_armed else ()
        self.found(findings, ruled)
        unread = commands.unread_finding(self.ws.unread()) if self.rules_armed else ""   # lane rules1: a finding here
        findings = [*findings, self.redact(unread)] if unread else findings
        findings = self.called_finish(findings, failures)   # lane called1: its lines among them, under `CALLED=on`
        able = self.requests_left() >= 1 and self.left() >= ASK_S
        listed = "\n".join(f"- {line}" for line in findings[:sanity.FINDINGS])
        if not failures:
            if findings and not self.sanity_refused and able:
                self.sanity_refused, self.counts["finish_refused"] = True, 1
                self.counts["called_refused"] |= int(self.called_alone)     # lane called1: on its lines alone
                self.refused_on(findings, ruled, unread)
                self.log("finish", f"refused: {len(findings)} sanity finding(s) stand")
                return fit(FINISH_REFUSED[self.mode].format(n=len(findings), lines=listed), commands.BOUNDS["finish"],
                           tail), False
            turned = self.guards_turn(tail)
            if turned:
                self.finishes -= 1
            return (turned, False) if turned else self.ended(findings, reason)
        final = self.finishes > 1 or not able
        turned = self.finishcheck_turn(tail) if final else None
        if turned:
            self.finishes -= 1
            return turned, False
        self.finish_said = "finish: the checks fail"
        self.log("finish", f"{len(failures)} check(s) failing" + ("; the task ends" if final else ""))
        if final:
            return fit(f"finish: {len(failures)} check(s) fail; the task ends with the output as it stands.",
                       commands.BOUNDS["finish"]), True
        if findings:                            # shown with the failures: the one review turn they get
            self.sanity_refused, self.counts["finish_refused"] = True, 1
        failed = FINISHCHECK_FAILED if self.finishcheck_armed else FINISH_FAILED
        return fit(failed.format(n=len(failures), lines="\n".join(f"- {line}" for line in failures[:10]))
                   + (FINISH_FINDINGS.format(lines=listed) if findings else ""), commands.BOUNDS["finish"],
                   tail), False

    def ended(self, findings: list[str], reason: str) -> tuple[str, bool]:
        """A finish with the checks passing that ends the unit: clean, or its findings accepted; a clean one under form
        `review` over the loop opens the review instead, the first time (`review_at_finish`, lane review2)."""
        if not findings:
            self.counts["finish_clean"] = 1
            self.log("finish", "the checks pass")
            said = self.review_at_finish()
            if said:
                return said, False
            self.finish_said = "finish: the checks pass"
            return fit("finish: the checks pass; the task ends.", commands.BOUNDS["finish"]), True
        self.counts["finish_clean"] = 1
        self.finish_said = f"finish: the checks pass; {len(findings)} sanity finding(s) accepted"
        self.log("finish", f"the checks pass; {len(findings)} sanity finding(s) stand, accepted"
                           + (f" with a reason of {len(reason):,} characters" if reason.strip() else ""))
        return fit(f"finish: the checks pass; {len(findings)} sanity finding(s) accepted; the task ends.",
                   commands.BOUNDS["finish"]), True

    def found(self, findings, ruled=()) -> None:
        """A run's or a finish's findings counted by kind: any (`sanity_units`), the instruction's own properties
        (`instruction_findings`, lane rules1's properties left out) and its contract (`contract_findings`); and the
        rules that fired among them (`rules_fired`, lane rules1)."""
        rules = {each.line for each in ruled}
        if findings:
            self.counts["sanity_units"] = 1
        if any(line.startswith("instruction:") and line not in rules for line in findings):
            self.counts["instruction_findings"] = 1
        if any(line.startswith("contract:") for line in findings):
            self.counts["contract_findings"] = 1
        for each in ruled:
            if each.rule not in self.rules_fired:
                self.rules_fired[each.rule] = None
                self.log("rules", f"{each.rule} fired: {each.line[:200]}")

    # --------------------------------------------------------------------------------------- lane rules1's rules

    def refused_on(self, findings: list[str], ruled, unread: str) -> None:
        """A finish refused under the rules (lane rules1): `rules_refused` where a rule's finding stood and no other
        finding (the instruction's, the contract's, the sanity checks'), its rules' findings held for `rules_fixed`;
        `unread_refused` where the unread parameters' finding stood alone."""
        if not self.rules_armed:
            return
        rules = {each.line for each in ruled}
        others = [line for line in findings if line not in rules and line != self.redact(unread)]
        if ruled and not others:
            self.counts["rules_refused"] = 1
            self.rules_held = frozenset(each.ident for each in ruled)
            self.log("rules", "the finish refused on the rules' findings alone: "
                              + ", ".join(dict.fromkeys(each.rule for each in ruled)))
        elif unread and not ruled and not others:
            self.counts["unread_refused"] = 1
            self.log("rules", "the finish refused on the unread parameters alone")

    def ruled_after(self, ran: commands.Ran) -> None:
        """After a run's exit 0 under the rules (lane rules1): the first after a rule-refused finish clears its rules'
        findings or not (`rules_fixed`); a clean run with no finding has its output kept (`keep_clean`)."""
        if not self.rules_armed:
            return
        if self.rules_held is not None:
            held, self.rules_held = self.rules_held, None
            self.counts["rules_fixed"] = int(not held & {each.ident for each in ran.rules})
            self.log("rules", "the run after the refusal " + ("cleared" if self.counts["rules_fixed"] else
                                                               "left") + " the rules' findings")
        if ran.clean and not ran.findings:
            self.keep_clean()

    def keep_clean(self) -> None:
        """The output as it stands copied beside work/ (`KEPT`), where it changed since the copy kept and holds at
        most `KEPT_BYTES`; where it holds more, no copy is kept (an older output is never put back over a later
        clean one). Never raises: a copy that fails keeps none."""
        try:
            now = contract.snapshot(self.out)
            if self.kept is not None and now == self.kept_snapshot:
                return
            kept, self.kept, self.kept_snapshot = self.work / KEPT, None, None
            shutil.rmtree(kept, ignore_errors=True)
            if sum(path.stat().st_size for path in self.out.rglob("*") if path.is_file()) > KEPT_BYTES:
                self.log("rules", "the clean output is past the copy's bound: none kept")
                return
            shutil.copytree(self.out, kept, symlinks=True)
            self.kept, self.kept_snapshot = kept, now
        except Exception as exc:  # noqa: BLE001 - the copy never costs the run
            shutil.rmtree(self.work / KEPT, ignore_errors=True)
            self.log("rules", f"the clean output not kept ({type(exc).__name__})")

    def put_back(self, ended: str = "") -> None:
        """Lane rules1, the caller holding guard.WRITING: where the unit ended with no request left (the cap, the clock,
        the last call; `ended` the watchdog's close) and no review guards the output, and the output as it stands
        breaks a `PUTBACK_RULES` property that the kept clean output holds (every one it breaks), that output is put
        back whole (`putback`). Never raises, so the close that follows always runs."""
        if self.kept is None or self.review_from is not None or \
                (self.end_kind not in ("requests", "clock") and not ended):     # a copy is kept under the rules alone
            return
        try:
            self.ws.sanity()
            broken = [each for each in self.ws.ruled if each.rule in PUTBACK_RULES and each.prop is not None]
            if not broken:
                return
            if not all(props.holds(each.prop, self.kept) for each in broken):
                self.log("rules", "not put back: the kept clean output does not hold "
                         + ", ".join(dict.fromkeys(each.rule for each in broken)))
                return
            clear(self.out)
            for entry in sorted(self.kept.iterdir()):
                (shutil.copytree if entry.is_dir() and not entry.is_symlink() else shutil.copy2)(
                    str(entry), str(self.out / entry.name))
            self.counts["putback"] = 1
            self.log("rules", "the last clean output put back: the output at the end broke "
                     + ", ".join(dict.fromkeys(each.rule for each in broken))
                     + f", which it held ({ended or self.end_kind})")
            trace.event("putback", rules=sorted({each.rule for each in broken}))
        except Exception as exc:  # noqa: BLE001 - the close follows whatever failed here
            self.log("rules", f"not put back ({type(exc).__name__})")

    def rules_keys(self) -> dict:
        """Lane rules1's keys (`RULES_KEYS`): the setting (`-` outside `RULES_FORMS`), each rule that fired on a run or
        a finish, any, the finish refused on the rules alone or on the unread parameters alone, the output put back,
        and the run after a rule-refused finish clearing its findings."""
        return {"rules_mode": self.rules_setting if self.form in RULES_FORMS else "-",
                **{f"rule_{name}": int(name in self.rules_fired) for name in (*commands.RULES,
                                                                             *commands.WIDE_RULES)},
                "rules_findings": int(bool(self.rules_fired)),
                **{key: self.counts[key] for key in ("rules_refused", "unread_refused", "putback", "rules_fixed")}}

    def resend_keys(self) -> dict:
        """Lane resend1's keys (`RESEND_KEYS`): the setting (`-` outside `RESEND_FORMS`), and the unchanged writes
        read as the finish (`write_repeats` counts them among every unchanged write)."""
        return {"resend_mode": self.resend_setting if self.form in RESEND_FORMS else "-",
                "resend_finished": self.counts["resend_finished"]}

    def joinask_keys(self) -> dict:
        """Lane joinask1's keys (`JOINASK_KEYS`): the setting (`-` outside `JOINASK_FORMS`), the changed asks by kind
        (`kind:n`, `-` where none), the module asks, the heads run and the definitions asked."""
        return {"joinask_mode": self.joinask_setting if self.form in JOINASK_FORMS else "-",
                "joinask_changed": self.junctions.changed_said(), "joinask_module_asked": self.junctions.modules,
                "joinask_head_run": self.junctions.head_runs, "joinask_defs_asked": self.junctions.defs_asked}

    def finishrun_keys(self) -> dict:
        """Lane finishrun1's keys (`FINISHRUN_KEYS`): the setting (`-` outside `FINISHRUN_FORMS`), and the finishes
        whose write's file ran first (`finish_ran`; `implicit_runs` counts those runs too)."""
        return {"finishrun_mode": self.finishrun_setting if self.form in FINISHRUN_FORMS else "-",
                "finish_ran": self.counts["finish_ran"]}

    def firstout_keys(self) -> dict:
        """Lane firstout1's keys (`FIRSTOUT_KEYS`): the setting (`-` outside `FIRSTOUT_FORMS`); the output files the
        task names that the unit's first run exiting 0 with no traceback wrote (`-` where none did), under either
        setting; and the output files the task names (`-` before the workspace)."""
        return {"firstout_mode": self.firstout_setting if self.form in FIRSTOUT_FORMS else "-",
                "first_outputs": "-" if self.first_outputs is None else self.first_outputs,
                "outputs_named": "-" if self.named_files is None else len(self.named_files)}

    def clause_keys(self) -> dict:
        """Lane clauses1's key (`CLAUSE_KEYS`): the setting, `-` outside `CLAUSE_FORMS`."""
        return {"clause_mode": self.clause_setting if self.form in CLAUSE_FORMS else "-"}

    def recon_keys(self) -> dict:
        """Lane recon1's keys (`RECON_KEYS`): the setting; the runs shown whose observation carried the reconciliation's
        findings, those findings, the checks they and a refusal held (`-` where none), the finishes turned away, and the
        findings said at one read run and gone at the next. Every key `-` outside `RECON_FORMS`."""
        if self.form not in RECON_FORMS:
            return {key: "-" for key in RECON_KEYS}
        return {"recon_mode": self.recon_setting, "recon_runs": self.counts["recon_runs"],
                "recon_said": self.counts["recon_said"],
                "recon_kinds": ",".join(check for check in recon.CHECKS if self.recon_kinds[check]) or "-",
                "recon_refused": self.counts["recon_refused"], "recon_acted": self.counts["recon_acted"]}

    def audit_keys(self) -> dict:
        """Lane audit1's keys (`AUDIT_KEYS`): the setting; whether the audit was said, its items; the reply after it
        read: its lines in the answer form, those quoting the task, those answered FIX, and whether a write followed in
        it; whether a finish was held, and whether the first clean run came too late for it. Every key `-` outside
        `AUDIT_FORMS`."""
        if self.form not in AUDIT_FORMS:
            return {key: "-" for key in AUDIT_KEYS}
        return {"audit_mode": self.audit_setting, **{key: self.counts[key] for key in AUDIT_KEYS[1:]}}

    def drafts_first(self, task, pathmap, data: str) -> str:
        """Lane drafts1 (`DRAFTS=one` or `three` in `DRAFTS_FORMS`; agent/n11/drafts.py): the round before the loop's
        first request, where the loop keeps `drafts.FLOOR_REQUESTS` beyond the round's allowance; the block the first
        message closes on where the clean drafts disagree, else "" (always under `one`). A failure in the round's own
        code is logged, what it left in the output removed, and the loop starts as today. The keys are logged on the
        `drafts` step line."""
        if not self.drafts_armed:
            return ""
        allowed = drafts.allowance(self.drafts_setting)
        if self.requests_left() < allowed + drafts.FLOOR_REQUESTS:
            self.log("drafts not sent", f"{self.requests_left()} request(s) left, under the round's {allowed} and the "
                                        f"loop's {drafts.FLOOR_REQUESTS}")
            text = ""
        else:
            self.drafting = drafts.Round(self, task, pathmap, data, self.drafts_setting)
            try:
                text = self.drafting.go()
            except Exception as exc:  # noqa: BLE001 - the round's failure costs its drafts, never the unit
                self.drafting.given_up(f"{type(exc).__name__}; the loop as today")
                text = ""
        self.log("drafts", " ".join(f"{key}={value}" for key, value in self.drafts_keys().items()))
        return text

    def drafts_close(self) -> None:
        """Lane drafts1, at the unit's close, the caller holding guard.WRITING: the round's held floor written where the
        output holds nothing to grade (`drafts.Round.close`); nothing where no round ran."""
        if self.drafting is not None:
            self.drafting.close()

    def drafts_keys(self) -> dict:
        """Lane drafts1's keys (`DRAFTS_KEYS`): the setting; the drafts sent, those clean, whether a clean draft's files
        stood as the floor, whether two or more matched, whether the table was said and its fields, the round's seconds
        on the unit's clock (`-` where no round ran). Every key `-` outside `DRAFTS_FORMS`."""
        if self.form not in DRAFTS_FORMS:
            return {key: "-" for key in DRAFTS_KEYS}
        found = self.drafting.keys() if self.drafting is not None else \
            {**dict.fromkeys(DRAFTS_KEYS[1:-1], 0), "drafts_s": "-"}
        return {"drafts_mode": self.drafts_setting, **found}

    # ------------------------------------------------------------------------------ lane repairpair1: the pair

    def repair_due(self, last: bool) -> bool:
        """Lane repairpair1 (`REPAIR=pair` in `REPAIR_FORMS`, agent/n11/repair.py): whether this request is a repair
        turn, B sent beside it: the last turn's run failed its own observation (an exit other than 0, a traceback, the
        checks failing, a named output file missing among them), or its finish was not taken on a failing output (the
        checks failing, or lane finishcheck1's or finishfail1's turn away), a run having been shown; never the last
        call, a held head's rest, the review's request or the sketch's, nor after the watchdog's kill."""
        if not self.repair.armed or self.repair.killed:
            return False
        turned = self.counts["finishcheck_turned"] + self.counts["finishfail_turned"]
        turned_now, self.repair_turned = turned > self.repair_turned, turned
        if last or self.held is not None or self.review_from is not None or self.sketch_due or self.shown_run is None:
            return False
        turn = self.counts["turns"]
        ran = self.runs_at.get(turn) if turn else None
        if ran is not None and (ran.exit != 0 or bool(ran.error) or bool(ran.failures)):
            return True
        return turn > 0 and "finish" in self.turn_calls and (turned_now or bool(self.ws.checks()))

    def repair_scripts(self) -> list[tuple[str, str]]:
        """Lane repairpair1: the failed run's scripts as B's request shows them: the file the last run shown ran, then
        the modules under work/ it imports, each by its name; [] where that file is gone."""
        here, named = self.ws.local(self.shown_run[1])
        if here is None or not here.is_file():
            return []
        found = [(named, here.read_text(encoding="utf-8", errors="replace"))]
        local, _ = self.ws.reached(Path(named).stem)
        for name in sorted(local):
            path, said = self.ws.dir / f"{name}.py", f"{commands.WORK}/{name}.py"
            if path.is_file() and said != named:
                found.append((said, path.read_text(encoding="utf-8", errors="replace")))
        return found

    def repair_state(self) -> dict:
        """The state before repair asks decorate the preceding observation."""
        return {name: copy.deepcopy(getattr(self, name)) for name in TURN_STATE}

    def repair_start(self, system: str, offered: list[dict] | None, think: dict) -> None:
        """Lane repairpair1: B sent beside A at a repair turn (`repair_due`) where the time left holds the run cap and
        a request's median (`pace`, else `PACE_PRIOR_S`) and B's room a request, else counted skipped: its user turn
        (`repair.rewrite_turn`: the task as the loop's first message frames it, the failed run's scripts, its
        observation as shown, the ask, the budget line), A's system prompt and tools with the write tool required, A's
        thinking, its bound the time left less the run cap; what the loop holds before A's turn kept to put back where B
        is kept (`TURN_STATE`, the conversation's length, the files under work/); then its client and thread
        (`repair.Repair.send`)."""
        median, cap = self.pace() or PACE_PRIOR_S, self.run_cap()
        if self.left() < cap + median:
            self.repair.counts["repair_skipped_clock"] += 1
            self.log("repair", f"not sent: {max(self.left(), 0):.0f} s left, under a run's {cap:.0f} s and a "
                               f"request's {median:.0f} s")
            return
        allowed = min(repair.PER_PAIR, self.repair.room())
        if allowed < 1:
            self.repair.counts["repair_skipped_requests"] += 1
            self.log("repair", "not sent: the rewrite's requests are spent")
            return
        scripts = self.repair_scripts()
        if not scripts:
            self.log("repair", f"not sent: {self.shown_run[1]} is gone")
            return
        files = repair.files_of(self.ws.dir)
        if files is None:
            self.log("repair", "not sent: the workspace is too large to restore")
            return
        turn = self.counts["turns"] + 1
        user = self.redact(repair.rewrite_turn(self.framing, scripts, self.shown_run[3], self.mode, self.budget()))
        keywords = {**think, "bound": max(model.MIN_CALL_S, self.left() - cap)}
        if offered is not None:
            keywords.update(tools=offered, tool_choice=WRITE_CHOICE)
        self.repair.number += 1
        root = (self.work / repair.WORK / str(self.repair.number)).absolute()
        shutil.rmtree(root, ignore_errors=True)
        last = next((self.runs_at[k] for k in sorted(self.runs_at, reverse=True) if self.runs_at[k] is not None), None)
        held = self.repair_before or self.repair_state()
        pair = repair.Pair(turn, len(held["messages"]), held,
                           files, root, repair.beside(self.ws, root), None,
                           [{"role": "system", "content": system}, {"role": "user", "content": user}], keywords,
                           allowed, set(self.seen), self.mode, (str(self.work), str(self.ws.dir)), MIN_RUN_S, last,
                           rest=rest_asked, code_of=continuation_code,
                           workspace={name: copy.deepcopy(getattr(self.ws, name)) for name in repair.WORKSPACE_STATE},
                           fallback=copy.deepcopy({name: value for name, value in vars(self.resort).items()
                                                   if name != "agent"}))
        repair.seeded(pair.ws.dir, pair.files)
        self.repair.send(pair, self.client, self.run_cap)
        self.log("repair", f"turn {turn}: a rewrite sent beside the repair ({', '.join(name for name, _ in scripts)} "
                           f"and its run shown), its requests {allowed} at most, bound {keywords['bound']:.0f} s")
        trace.event("repair", phase="sent", turn=turn, allowed=allowed, bound_s=round(keywords["bound"], 1),
                    left_s=round(self.left(), 1), scripts=[name for name, _ in scripts])

    def repair_settle(self, ending: bool = False) -> bool:
        """Lane repairpair1: the pair out settled before the loop's next request, or at its stop (`ending`)
        (`repair_settled`); a failure in its own code logged, B's room returned and its directory removed, and the
        loop goes on as it stands: the pair costs B, never the unit. True where B was kept."""
        pair = self.repair.pending
        try:
            return self.repair_settled(pair, ending)
        except Exception as exc:  # noqa: BLE001 - the pair's failure costs B, never the unit
            pair.abandoned = True
            pair.ws.kill()
            if self.repair.pending is pair:
                self.repair.close(pair, self.client)
            shutil.rmtree(pair.root, ignore_errors=True)
            self.log("repair", f"turn {pair.turn}: not settled ({type(exc).__name__}); the loop goes on as it stands")
            return False

    def repair_settled(self, pair: repair.Pair, ending: bool) -> bool:
        """The pair settled: B waited for a request's median after A's turn at most (`pace`, else `PACE_PRIOR_S`), within
        the time left (B abandoned where it is not done by then: its run killed, no further request or run, B lost),
        the waits the desk took off B's clock carried onto the loop's beyond A's own (common.house `Client.carry`),
        B's room returned to what it spent, the keep rule read (`repair.keep` over `repair_candidate`: A's turn's last
        run, else the run it repaired, over the output; B's over its own), B kept where it takes it and no watchdog
        has closed the unit (`repair_kept`); B's directory removed either way. True where B was kept."""
        began, patience = time.time(), self.pace() or PACE_PRIOR_S
        pair.thread.join(max(0.0, min(self.left(), patience)))
        alive = pair.thread.is_alive()
        if alive:
            pair.abandoned = True
            pair.ws.kill()
            pair.lost = pair.lost or f"not done a request's median ({patience:.0f} s) after A's turn"
        if self.client.deadline is not None and (moved := self.client.carry([pair.client], self.client.deadline)):
            self.log("clock", f"{moved:.0f} s of the rewrite's untimed waits carried onto the unit's clock")
        waited = time.time() - began
        self.repair.wait_s += waited
        self.repair.close(pair, self.client)
        main = self.ws.dir / "main.py"
        a = self.repair_candidate(self.runs_at.get(pair.turn) or pair.last, self.out,
                                  main.read_text(encoding="utf-8", errors="replace") if main.is_file() else "")
        b = None if alive or pair.lost or pair.ran is None else \
            self.repair_candidate(pair.ran, pair.ws.out, pair.call.arguments["text"])
        kept, why = repair.keep(a, b)
        counts = self.repair.counts
        counts["repair_a_clean"] += a.clean
        counts["repair_b_clean"] += b is not None and b.clean
        counts["repair_b_lost"] += b is None
        counts["repair_b_joined"] += b is not None and pair.joined
        if kept == "rewrite" and not self.repair_kept(pair, ending):
            kept, why = "patch", "closed"
        counts["repair_kept_rewrite" if kept == "rewrite" else "repair_kept_patch"] += 1
        shutil.rmtree(pair.root, ignore_errors=True)
        self.log("repair", f"turn {pair.turn}: the {'rewrite' if kept == 'rewrite' else 'repair'} kept ({why}); A "
                           f"{'clean' if a.clean else 'not clean'}, B "
                           + (f"lost ({pair.lost})" if b is None else "clean" if b.clean else "not clean")
                           + f"; {waited:.1f} s waited, {pair.admitted} request(s) of B's")
        trace.event("repair", phase="settled", turn=pair.turn, kept=kept, by=why, a=a.said(),
                    b=b.said() if b is not None else None, lost=pair.lost, joined=pair.joined,
                    wait_s=round(waited, 1), b_requests=pair.admitted,
                    call=pair.call.describe() if pair.call is not None else None)
        return kept == "rewrite"

    def repair_candidate(self, ran: commands.Ran | None, out: Path, text: str) -> repair.Candidate:
        """Lane repairpair1: a side of the pair as the keep rule reads it: its run, its output as it stands, and its
        findings: the run's checks failing and sanity findings, lane finishcheck1's contract over its output where
        armed (`finishcheck_findings`, uncounted), lint's rows over its script (`repair.lint_rows`)."""
        found = 0 if ran is None else len(ran.failures) + len(ran.findings)
        if self.finishcheck_armed:
            found += len(self.finishcheck_findings(out, counted=False))
        return repair.candidate(ran, out, self.ws.named, found + repair.lint_rows(self.ws, text))

    def repair_kept(self, pair: repair.Pair, ending: bool) -> bool:
        """Lane repairpair1, B kept: under guard.WRITING and never after the watchdog's close, the loop's state put
        back as A's turn found it (`TURN_STATE`, the files under work/) and B's files moved to the output's root, what
        it held first removed (`repair.moved`); B's script then saved as work/main.py by the House's write, and B's run
        the turn's, counted and shown as a loop run is (`tail`, `ran`, `saw`, the card and the audit after it). Mid-loop
        (not `ending`) the conversation, cut back to its length before A's turn, carries B's reply as one write of
        work/main.py answered by that write and that run, and the turn's closing reads are made as after any turn.
        False where the watchdog closed first."""
        with guard.WRITING:
            if self.repair.killed:
                return False
            for name, value in pair.held.items():
                setattr(self, name, value)
            for name, value in pair.workspace.items():
                setattr(self.ws, name, value)
            vars(self.resort).update(pair.fallback)
            if pair.files is not None:
                repair.put_back(self.ws.dir, pair.files)
            laid = repair.moved(pair.ws.out, self.out)
        self.counts["turns"] = pair.turn
        self.counts["calls"] += 1
        self.request_seconds.append(pair.request_s)
        self.replied("loop", pair.reply)
        if "thinking" in pair.keywords:
            self.write_thought(self.request_kind(), pair.reply)
        self.paced_turn()
        text = pair.call.arguments["text"]
        said = self.ws.write(MAIN_FILE, text)
        if said.startswith("wrote "):
            self.versions.setdefault(MAIN_FILE, []).append((pair.turn, sha256(commands.unfenced(text))))
            self.saved(MAIN_FILE)
            self.defs_files.discard(MAIN_FILE)
        ran = pair.ran
        self.ws.runs += 1
        tail = self.tail(ran, True)
        shown = f"{pair.said}\n{tail}" if tail else pair.said
        self.ran(ran, pair.cap)
        self.saw(shown, ran)
        shown = self.audited(self.carded_after(shown, ran), ran)
        self.log("repair", f"the rewrite's {laid} output entr{'y' if laid == 1 else 'ies'} moved to the output, its "
                           f"script {MAIN_FILE}")
        self.messages = self.messages[:pair.mark]
        self.record(pair.text, [pair.call], [f"{said}\nthen, run for you: {shown}"], pair.native)
        self.reply_text, self.turn_calls = pair.text, ("write",)
        self.short_due = self.short
        self.redirected([pair.call])
        self.redirect_due = self.redirecting
        self.repeat_turn(False)
        return True

    def repair_keys(self) -> dict:
        """Lane repairpair1's keys (`REPAIR_KEYS`): the setting; the pairs sent, B kept and A kept, the repair turns
        skipped for the clock and for B's room, the pairs whose A and whose B were clean, B lost, the seconds waited on
        B after A's turn; every key `-` outside `REPAIR_FORMS`."""
        return self.repair.keys()

    def output_keys(self) -> dict:
        """Lane outguard1's key (`OUTPUT_KEYS`): 1 where the loop's last run ran with `OUTPUT_DIR` naming the output
        (common.runner.child_env), 0 where no run ran or the environment named another directory."""
        return {"output_env": getattr(self.ws, "output_env", 0)}

    def stated_keys(self) -> dict:
        """Lane conventions1's key (`STATED_KEYS`): the setting, `-` outside `STATED_FORMS`."""
        return {"stated_mode": self.stated_setting if self.form in STATED_FORMS else "-"}

    def cardfail_keys(self) -> dict:
        """Lane cardfail1's keys (`CARDFAIL_KEYS`): the setting (`-` outside `CARDFAIL_FORMS`), and the cards after a
        failing run the House was shown (`carded_after`)."""
        return {"cardfail_mode": self.cardfail_setting if self.form in CARDFAIL_FORMS else "-",
                "cardfail_shown": self.counts["cardfail_shown"]}

    # ------------------------------------------------------------------- lane repeat1: the repeated failing state

    def repeat_met(self, n: int, named: str, how: str) -> None:
        """Lane repeat1: a run's or a re-send's state as `repeat.Ledger` read it, `n` the unit's recurrence it is (0:
        none): the turn's latest kept for `repeat_turn`, logged and traced, under either setting. A later new or clean
        state cancels an earlier state's ask and hand-off without clearing the unit's history."""
        self.repeat_hit = n
        if not n:
            return
        self.log("repeat", f"{named} {how}: the program failing as an earlier run of it failed, the unit's recurrence "
                           f"{n}")
        trace.event("repeat", path=named, how=how, recurrence=n)

    def repeat_turn(self, cut: bool) -> None:
        """Lane repeat1, a turn's observations recorded: under `change`, where the turn ends on the unit's
        `repeat.ASK_AT`-th recurrence or a later one, the ask closes its last observation before the next request
        (`repeat_ask`; never after a cut, whose ask stands alone, nor with the last call's); at the
        `repeat.FALLBACK_AT`-th or later, once a unit, lane fallback1's one-shot first where it is armed and not yet
        sent (`repeat_handoff`). Nothing here ends the unit."""
        hit, self.repeat_hit = self.repeat_hit, 0
        if not self.repeat_armed or hit < repeat.ASK_AT:
            return
        if hit >= repeat.FALLBACK_AT and self.resort.armed and not self.resort.used and \
                not self.counts["repeat_fallback"]:
            self.repeat_handoff()
        self.repeat_due = not cut

    def repeat_ask(self) -> None:
        """Lane repeat1: `REPEAT_ASK` closing the last observation before the next request, as lane redirect1's demand
        does (`redirect_ask`), counted `repeat_asked`."""
        kept = self.messages[-1]["content"] == self.sketch_kept
        self.messages[-1] = {**self.messages[-1], "content": f"{self.messages[-1]['content']}\n{REPEAT_ASK}"}
        if kept:
            self.sketch_kept = self.messages[-1]["content"]
        self.counts["repeat_asked"] += 1
        self.log("repeat", "the changed-repair ask closes the last observation")

    def repeat_handoff(self) -> None:
        """Lane repeat1: the unit handed to lane fallback1's one-shot (`fallback.Resort.fire`, reason `fallback.REPEAT`,
        its run a loop run's at most, `run_cap`), the loop's output set aside beside work/ first (`REPEAT_ASIDE`) and
        put back after it; the one-shot's output kept beside work/ (`REPEAT_FLOOR`) where it holds a file to grade and
        passes the checks, for `repeat_end`. The loop goes on, and nothing of the one-shot is said to the House."""
        self.counts["repeat_fallback"] = 1
        if self.repeat_with is None:
            return
        aside, floor = self.work / REPEAT_ASIDE, self.work / REPEAT_FLOOR
        try:
            with guard.WRITING:
                shutil.rmtree(aside, ignore_errors=True)
                shutil.copytree(self.out, aside, symlinks=True)
        except OSError as exc:
            self.log("repeat", f"the one-shot not sent: the output not set aside ({type(exc).__name__})")
            return
        self.log("repeat", "the unit's third recurrence: lane fallback1's one-shot sent, the loop's output set aside")
        self.resort.fire(self, *self.repeat_with, fallback.REPEAT, ASK_S, run_s=self.run_cap())
        failing = self.ws.checks() if self.resort.used else []
        kept = self.resort.used and fallback.graded(self.out, self.ws.named) and not failing
        with guard.WRITING:
            if kept:
                shutil.rmtree(floor, ignore_errors=True)
                shutil.copytree(self.out, floor, symlinks=True)
                self.repeat_floor = floor
            refilled(aside, self.out)
            shutil.rmtree(aside, ignore_errors=True)
        self.log("repeat", "the one-shot's output kept aside: it passes the checks" if kept else
                 "the one-shot's output not kept: " + ("not sent" if not self.resort.used else
                                                       f"{len(failing)} check(s) failing" if failing else
                                                       "no file to grade") + "; the loop's put back")
        trace.event("repeat_fallback", sent=self.resort.used, kept=kept)

    def repeat_end(self, ended: str = "") -> None:
        """Lane repeat1, the caller holding guard.WRITING: where the one-shot's output was kept (`repeat_handoff`) and
        the unit ends on an output holding no file to grade or failing the checks, the one-shot's put back whole
        (`repeat_floor`); the loop's stands where it passes them. `ended`, the watchdog's close. Never raises, so the
        close that follows always runs."""
        if self.repeat_floor is None:
            return
        try:
            if fallback.graded(self.out, self.ws.named) and not self.ws.checks():
                self.log("repeat", "the loop's output stands: it passes the checks")
                return
            refilled(self.repeat_floor, self.out)
            self.counts["repeat_floor"] = 1
            self.log("repeat", "the one-shot's output put back: the loop's ended failing the checks"
                     + (f" ({ended})" if ended else ""))
            trace.event("repeat_floor", ended=ended or self.end_kind)
        except Exception as exc:  # noqa: BLE001 - the close follows whatever failed here
            self.log("repeat", f"the one-shot's output not put back ({type(exc).__name__})")

    def repeat_keys(self) -> dict:
        """Lane repeat1's keys (`REPEAT_KEYS`): the setting (`-` outside `REPEAT_FORMS`); the recurrences of a failing
        program state met, under either setting; the asks sent; whether the third handed the unit to the one-shot; and
        whether the one-shot's output was put back at the end."""
        return {"repeat_mode": self.repeat_setting if self.form in REPEAT_FORMS else "-",
                "repeat_states": self.repeats.states, "repeat_asked": self.counts["repeat_asked"],
                "repeat_fallback": self.counts["repeat_fallback"], "repeat_floor": self.counts["repeat_floor"]}

    def snapshot_keys(self) -> dict:
        """Lane snapshot1's keys (`SNAPSHOT_KEYS`): the setting (`-` outside `SNAPSHOT_FORMS`), the runs shown whose
        answer carried the state at their failure, those where it was dropped, and the said ones' median measure in
        `measure.UNIT` (`-` where none was said)."""
        return {"snapshot_mode": self.snapshot_setting if self.form in SNAPSHOT_FORMS else "-",
                "snapshot_shown": self.counts["snapshot_shown"], "snapshot_dropped": self.counts["snapshot_dropped"],
                "snapshot_tokens": round(statistics.median(self.snapshot_sizes)) if self.snapshot_sizes else "-"}

    def derive_keys(self) -> dict:
        """Lane derive1's keys (`DERIVE_KEYS`): the setting; whether the loop's first reply opened with a derivation,
        and the unit's first script's assertions and top-level functions (`-` where none was read), under either
        setting; the runs shown whose observation carried the harness's own checks' findings, those findings, and the
        checks that made them (`-` where none). Every key `-` outside `DERIVE_FORMS`."""
        if self.form not in DERIVE_FORMS:
            return {key: "-" for key in DERIVE_KEYS}
        return {"derive_mode": self.derive_setting,
                "derive_said": "-" if self.derive_said is None else int(self.derive_said),
                "derive_checks": self.derive_script[0] if self.derive_script else "-",
                "derive_functions": self.derive_script[1] if self.derive_script else "-",
                "witness_runs": self.counts["witness_runs"], "witness_findings": self.counts["witness_findings"],
                "witness_kinds": ",".join(sorted(self.witness_kinds)) or "-"}

    def contract_keys(self) -> dict:
        """Lane contract1's keys (`CONTRACT_KEYS`): the setting; the terms the instruction states (under any setting);
        the runs shown whose observation carried the contract's findings, those findings, the kinds they and a refusal
        held (`-` where none), and the finishes turned away. Every key `-` outside `CONTRACT_FORMS`."""
        if self.form not in CONTRACT_FORMS:
            return {key: "-" for key in CONTRACT_KEYS}
        return {"contract_mode": self.contract_setting, "contract_terms": self.counts["contract_terms"],
                "contract_runs": self.counts["contract_runs"], "contract_said": self.counts["contract_said"],
                "contract_kinds": ",".join(kind for kind in stated_contract.KINDS if self.contract_kinds[kind]) or "-",
                "contract_refused": self.counts["contract_refused"]}

    def join_keys(self) -> dict:
        """Lane join2 of 10-16's keys (`REJOIN_KEYS`): the setting (`-` outside `JOIN_FORMS`), the joins taken with the
        cut statement's received lines kept, the failing joins those readings were tried on, the rests read as
        restarts, and the writes staged beside a saved file."""
        return {"join_mode": self.join_setting if self.form in JOIN_FORMS else "-",
                **{key: self.counts[key] for key in REJOIN_KEYS[1:]}}

    def finishcheck_keys(self) -> dict:
        """Lane finishcheck1's keys (`FINISHCHECK_KEYS`): the setting (`-` outside `FINISHCHECK_FORMS`), the finishes
        the contract turned away, and the kinds its findings held over the unit's finishes (`-` where none)."""
        return {"finishcheck_mode": self.finishcheck_setting if self.form in FINISHCHECK_FORMS else "-",
                "finishcheck_turned": self.counts["finishcheck_turned"],
                "finishcheck_flags": ",".join(kind for kind in finishcheck.KINDS if kind in self.finishcheck_found)
                or "-"}

    def finishfail_keys(self) -> dict:
        """Lane finishfail1's keys (`FINISHFAIL_KEYS`): the setting (`-` outside `FINISHFAIL_FORMS`), the finishes on a
        failing run turned away, and whether the unit ended on one counted after `FINISHFAIL_MAX` (`finish`)."""
        return {"finishfail_mode": self.finishfail_setting if self.form in FINISHFAIL_FORMS else "-",
                "finishfail_turned": self.counts["finishfail_turned"],
                "finishfail_ended": self.counts["finishfail_ended"]}

    # ------------------------------------------------------------------------------------- the called finding

    @property
    def called_armed(self) -> bool:
        """Lane called1: `CALLED=on` in the loop's forms (`CALLED_FORMS`)."""
        return self.called_setting == "on" and self.form in CALLED_FORMS

    def card_reached(self, texts=(), candidates=()) -> None:
        """Lane called1: under `CALLED=on`, the functions whose card reached the House (`commands.Workspace.carded`,
        which the finding reads): each of `candidates` whose typed signature opens a line of a request's `texts` (a
        card's first line, as `show` reads the pointer's), or each where no text is given (`lib`'s answer); never one
        on an `UNPOINTED` shelf, whose conventions and I/O helpers are every task's words."""
        carded = self.ws.carded if self.ws is not None else None
        if carded is None:
            return
        fresh = [function for function in candidates
                 if function.name not in carded and function.shelf not in library.UNPOINTED]
        if fresh and texts:
            lines = {line for text in texts if text for line in str(text).split("\n")}
            fresh = [function for function in fresh if function.typed in lines]
        for function in fresh:
            carded[function.name] = function

    def called_ran(self, ran: commands.Ran) -> None:
        """Lane called1: after a run, a function a finding named before that this run's files no longer compute by hand
        and do call (`used`): `called_fixed`. The finding itself is counted where the run's observation says it
        (`tail`), so a run whose observation is never sent (the library pass's, the last call's) counts none."""
        if self.ws.carded is None or not self.called_named or self.counts["called_fixed"]:
            return
        found = {rewrite.function.name for rewrite in ran.called}
        gone = {name for name in self.called_named if name not in found}
        fixed = gone & self.used()[1] if gone else set()
        if fixed:
            self.counts["called_fixed"] = 1
            self.log("called", f"fixed: {', '.join(sorted(fixed))} called, no longer computed by hand")

    def called_finish(self, findings: list[str], failures) -> list[str]:
        """Lane called1: the finish's `findings` with the called finding's lines among them (`commands.beside`), so its
        one refusal fires on them as on any finding; whether they stand alone with the checks passing kept for
        `called_refused`. Under `off` the findings as they are."""
        found = self.ws.called()
        self.called_found(found)
        self.called_alone = bool(found) and not findings and not failures
        return commands.beside(findings, found, self.ws.ruled if self.rules_armed else ())

    def called_found(self, found) -> None:
        """A called finding said (`called_findings`), each function kept by the turn it was first said, and each def
        or routine name logged once per unit."""
        for rewrite in found:
            self.counts["called_findings"] = 1
            if rewrite.function.name not in self.called_named:
                self.called_named[rewrite.function.name] = self.counts["turns"]
            identity = (rewrite.by, rewrite.name)
            if identity not in self.called_logged:
                self.called_logged.add(identity)
                self.log("called", f"{rewrite.function.call} by hand: `{rewrite.name}` ({rewrite.by}) in "
                                   f"{rewrite.file}")

    # ------------------------------------------------------------------------------------------ the messages

    def system(self, index: str) -> str:
        """The loop's system prompt: `SYSTEM` in the mode's form, the index in it, the cards after it; the library rule
        without its lookup clause under lane libnever1's `LIB=never` (`LIBRARY_USE_NEVER`); the form's paragraph saying
        a fix by definition under lane fixdef1's `WRITE=defs` (`PART_DEFS`); the task's first output file asked alone,
        each later one in a reply of its own, under lane firstout1's `FIRSTOUT=on` (`FIRSTOUT_SAID`); the conventions
        operation by operation and each one the task pins stated once, under lane conventions1's `STATED=on`
        (`STATED_LINES`, `SETTLE_STATED`); the fifth step said the audit's way, under lane audit1's `AUDIT=on`
        (`audit.STEP` for `audit.TODAY`, a replacement in the rendered text, whether it met the step logged once)."""
        tools = self.mode == "tools"
        part = PART_DEFS[self.mode] if self.write_defs else TOOLS_PART if tools else FENCED_PART
        system = SYSTEM.format(tools=part, libraries=LIBRARIES, run=RUN_S,
                             example=TOOLS_EXAMPLE if tools else FENCED_EXAMPLE, index=index, api=api_card(),
                             conventions="\n".join(STATED_LINES if self.stated_armed else CONVENTION_LINES),
                             settle=SETTLE_STATED if self.stated_armed else SETTLE,
                             library=LIBRARY_USE_NEVER if self.lib_never else LIBRARY_USE)
        for today, said in FIRSTOUT_SAID.items() if self.firstout_armed else ():
            system = system.replace(today, said, 1)
        if self.audit_armed:            # lane audit1: the fifth step said the audit's way, the template's text kept
            stepped = system.replace(audit.TODAY, audit.STEP[self.mode], 1)
            if self.audit_step is None:
                self.audit_step = stepped != system
                self.log("audit", "the system prompt's fifth step said the audit's way" if self.audit_step else
                         "the system prompt holds no fifth step of today's: said as it stands")
            system = stepped
        system += f"\n\n{self.cards}" if self.cards else ""
        return system.replace(FIX_WHOLE, FIX_DEFS, 1) if self.write_defs else system

    @property
    def prompt_word(self) -> str:
        """Lane prompt1: the prompt's word this unit runs: its setting where armed (form `loop`, `PROMPT_FORMS`, the
        loop started in the fenced form, `prompts.MODE`) and today's text in its paragraphs (`control`), else `full`."""
        armed = self.form in PROMPT_FORMS and self.mode_setting == prompts.MODE and not self.prompt_fell
        return self.prompt_setting if armed else "full"

    def control(self, index: str) -> prompts.Control | None:
        """Lane prompt1: today's system prompt (`system`) in its paragraphs (`prompts.Control`); None where it is not,
        and from then the unit runs today's text (`prompt_fell`), logged once."""
        found = prompts.Control.read(self.system(index), self.cards)
        if found is None and not self.prompt_fell:
            self.prompt_fell = True
            self.log("prompt", f"{self.prompt_setting}: today's system prompt is not in its paragraphs; today's text")
        return found

    def clean_armed(self, index: str) -> bool:
        """Lane clean1: whether `clean` runs, its word armed (`prompt_word`) and today's render in its paragraphs
        (`control`, which turns the unit to today's text where it is not): read before the selection call and the
        first message, so that neither is said as `clean` where the system prompt would not be."""
        return self.prompt_word == "clean" and self.control(index) is not None

    def worded(self, index: str) -> str:
        """Lane prompt1: the loop's system prompt as the prompt's word says it (`prompt_word`; agent/n11/prompts.py):
        `full`, today's (`system`); `lean`, design A's, its paths the path map's and its cards of the functions the
        first message cards (`carded_order`) in the form lane lean2's setting names (`prompts.lean_card_as`, design
        A's under `cut`), their bytes kept (`lean_cards_bytes`); `leanconst`, `lean`'s with design C's parts (lane
        leanconst1, `prompts.LEANCONST`), its whole cards' conventions as stubs; `taskfirst`, today's head, reply
        form, cut, must list
        and example, the rest in the first message (`task_part`); `constants`, today's with design C's edits, an edit
        whose anchor is not met logged once; `norules` and `rulesafter`, today's without its runtime paragraph and
        defaults list (`prompts.moved_out`; `rulesafter` says them in the first message, `task_part`); `minimal`,
        lane prompt2's whole (`prompts.minimal`: the cut as the unit's `CUT` runs it, the one-shot's libraries), no
        card; `clean`, today's with the static audit's edits in `prompts.CLEAN_ORDER` (lane clean1,
        `prompts.clean_parts`), an edit whose anchor is not met logged once. The functions its cards carry and its
        bytes before them kept."""
        word = self.prompt_word
        control = self.control(index) if word in prompts.CONTROLLED else None
        word = self.prompt_word
        paths = self.ws.pathmap if self.ws is not None else None
        dirs = (str(paths.data_dir), str(paths.out)) if paths is not None else ("", "")
        if word == "minimal":
            text = prompts.minimal(self.cut_setting, n0.LIBRARIES)
            self.prompt_carded, self.prompt_head = [], prompts.size(text)
            return text
        if word in prompts.LEAN_WORDS:
            limit = selection.card_bound() if self.selection.result is not None else library.card_bound()
            cards, carded = prompts.lean_cards(self.carded_order, limit, self.lean_cards_setting, word)
            versions = prompts.versions_said(platform.python_version(), installed())
            self.prompt_carded = carded
            self.prompt_head = prompts.size(prompts.lean(*dirs, versions, RUN_S, "", word))
            self.lean_cards_bytes = prompts.size(cards)
            return prompts.lean(*dirs, versions, RUN_S, cards, word)
        if word == "taskfirst":
            text = prompts.taskfirst_system(control)
            self.prompt_carded, self.prompt_head = self.carried_functions(), prompts.size(text)
            return text
        if word == "clean":
            said, missed = prompts.clean_parts(control, self.cut_setting, TIMING_LINES, FENCED_EXAMPLE)
            if missed and self.prompt_missed is None:
                self.log("prompt", f"clean: {', '.join(missed)} left as today's (the anchor not met)")
            self.prompt_missed = missed
            self.prompt_carded = self.carried_functions()
            self.prompt_head = prompts.size(prompts.clean_text(said, library_=False))
            return prompts.clean_text(said)
        system = self.system(index)
        parts: dict[str, str] = {}
        if word == "constants":
            parts, missed = prompts.constants_parts(control, *dirs, self.carded_order, FENCED_EXAMPLE)
            system = control.text(**parts)
            if missed and self.prompt_missed is None:
                self.log("prompt", f"constants: {', '.join(missed)} left as today's (the anchor not met)")
            self.prompt_missed = missed
        elif word in ("norules", "rulesafter"):
            parts = prompts.moved_out()
            system = control.text(**parts)
        read = control or prompts.Control.read(system, self.cards)
        self.prompt_carded = self.carried_functions()
        self.prompt_head = prompts.size(read.rules(**parts)) if read is not None else prompts.size(system)
        return system

    def carried_functions(self) -> list[library.Function]:
        """The functions whose card today's cards carry (`carried_cards`), in the order carded."""
        names = self.carried_cards()
        return [function for function in self.carded_order if function.name in names]

    def selected(self, task, lib: library.Library, index: str, system=None) -> str:
        """Lane select2 (`SELECT=call`, forms `loop`, `sketch` and `state`; agent/n11/selection.py): the selection
        call before the loop's first message (form `state`'s first state, lane state4), bounded as a loop request is
        while the last resort's reserve is held (lane fallback2). Where it picked, its shelves' section takes the
        index's place (`self.index` too, from which a route's refusal of the tools rebuilds the prompt) and is
        returned, else `index`, today's; in the loop's forms the cards of the pointer's functions and its picks take
        the pointer's cards' place, and in form `state` they are its CARDS section's base (`state_base`). The system
        prompt (`system`, the loop's `system` by default) is counted as sent and as today's where the call was sent
        (`select_system_tokens`, `select_system_today`). Under `SELECT=given` (lane given1, the desk's) the unit's
        given list in the call's place, no request sent, counted alike. Under lane prompt2's `minimal` nothing is
        sent or read, whatever the setting: its request stays the loop's, logged. Under lane clean1's `clean` the
        call's index is said without the lookup (`library_unsaid`), which the call never offers."""
        if self.prompt_word == "minimal":
            if self.selection.armed:
                self.log("select", f"{self.selection.setting} ({self.selection.source}): not consulted under prompt "
                                   "minimal")
            return index
        system = system or self.system
        clean = self.clean_armed(index)     # lane clean1: the call's index without the lookup it never offers
        shaped = self.selection.call(self, task, lib, sketch_names, ASK_S, self.resort.bounded(self.left()),
                                     library_unsaid if clean else None)
        today = measure.size(system(index)) if self.selection.consulted else None
        if shaped is not None:
            self.index, index = shaped.index, shaped.index
            if self.form != "state":    # form state's cards are its CARDS section's, trimmed and logged there
                self.cards, self.cuts = shaped.cards, list(shaped.cuts)
                for cut in shaped.cuts if not self.whole_armed else ():    # lane first1: fitted again under `whole`
                    self.log("cards", f"the selection's {cut}")
        if today is not None:
            self.selection.measured(measure.size(system(index)), today)
        return index

    @property
    def whole_armed(self) -> bool:
        """Lane first1: `CARDS=whole` in the loop's forms (`CARDS_FORMS`), and lane libnever1's `LIB=never` there
        whatever `CARDS` says."""
        return (self.cards_setting == "whole" or self.lib_never) and self.form in CARDS_FORMS

    @property
    def lib_never(self) -> bool:
        """Lane libnever1: `LIB=never` in the loop's forms (`LIB_FORMS`): `lib` neither offered nor said."""
        return self.lib_setting == "never" and self.form in LIB_FORMS

    @property
    def rules_armed(self) -> bool:
        """Lane rules1: `RULES=on` (or lane rules2's `wide`) in the loop's forms (`RULES_FORMS`)."""
        return self.rules_setting in ("on", "wide") and self.form in RULES_FORMS

    @property
    def rules_wide(self) -> bool:
        """Lane rules2: `RULES=wide` in the loop's forms: the six rules after the eight."""
        return self.rules_setting == "wide" and self.form in RULES_FORMS

    @property
    def resend_armed(self) -> bool:
        """Lane resend1: `RESEND=finish` in the loop's forms (`RESEND_FORMS`)."""
        return self.resend_setting == "finish" and self.form in RESEND_FORMS

    def through(self, named: str) -> str:
        """What the rest of `named`'s held head is asked through (`CUT_THROUGH`): work/main.py's run and its writes,
        under lane firstout1's `FIRSTOUT=on` its output whatever it writes (`CUT_THROUGH_FIRSTOUT`); another file's
        end."""
        if named != MAIN_FILE:
            return CUT_THROUGH[False]
        return CUT_THROUGH_FIRSTOUT if self.firstout_armed else CUT_THROUGH[True]

    def split_words(self, said: str) -> str:
        """Under FIRSTOUT, a cut script's modules belong to that output's script, including a later script."""
        return said.replace(commands.SPLIT, FIRSTOUT_SPLIT) if self.firstout_armed else said

    @property
    def firstout_armed(self) -> bool:
        """Lane firstout1: `FIRSTOUT=on` in the loop's forms (`FIRSTOUT_FORMS`)."""
        return self.firstout_setting == "on" and self.form in FIRSTOUT_FORMS

    @property
    def clause_armed(self) -> bool:
        """Lane clauses1: a `CLAUSE` word other than `off` in the loop's forms (`CLAUSE_FORMS`)."""
        return self.clause_setting != "off" and self.form in CLAUSE_FORMS

    @property
    def stated_armed(self) -> bool:
        """Lane conventions1: `STATED=on` in the loop's forms (`STATED_FORMS`)."""
        return self.stated_setting == "on" and self.form in STATED_FORMS

    @property
    def drafts_armed(self) -> bool:
        """Lane drafts1: `DRAFTS=one` or `three` in `DRAFTS_FORMS`."""
        return self.drafts_setting in drafts.COUNTS and self.form in DRAFTS_FORMS

    @property
    def recon_armed(self) -> bool:
        """Lane recon1: `RECON=on` or `refuse` in the loop's forms (`RECON_FORMS`)."""
        return self.recon_setting in ("on", "refuse") and self.form in RECON_FORMS

    @property
    def recon_refusing(self) -> bool:
        """Lane recon1: `RECON=refuse` in the loop's forms."""
        return self.recon_setting == "refuse" and self.recon_armed

    @property
    def audit_armed(self) -> bool:
        """Lane audit1: `AUDIT=on` in the loop's forms (`AUDIT_FORMS`)."""
        return self.audit_setting == "on" and self.form in AUDIT_FORMS

    @property
    def audit_holding(self) -> bool:
        """Lane audit1: the audit said after a run of this reply's and no finish held before in the unit
        (`AUDIT_MAX`): a finish in this reply is held."""
        return self.audit_at is not None and self.audit_at == self.counts["turns"] \
            and self.counts["audit_held"] < AUDIT_MAX

    @property
    def cardfail_armed(self) -> bool:
        """Lane cardfail1: `CARDFAIL=on` or `again` in the loop's forms, unless `minimal` withholds cards."""
        return self.cardfail_setting != "off" and self.form in CARDFAIL_FORMS and self.prompt_word != "minimal"

    @property
    def finishrun_armed(self) -> bool:
        """Lane finishrun1: `FINISHRUN=on` in the loop's forms (`FINISHRUN_FORMS`)."""
        return self.finishrun_setting == "on" and self.form in FINISHRUN_FORMS

    @property
    def finishfail_armed(self) -> bool:
        """Lane finishfail1: `FINISHFAIL=on` in the loop's forms (`FINISHFAIL_FORMS`)."""
        return self.finishfail_setting == "on" and self.form in FINISHFAIL_FORMS

    @property
    def snapshot_armed(self) -> bool:
        """Lane snapshot1: `SNAPSHOT=on` in the loop's forms (`SNAPSHOT_FORMS`)."""
        return self.snapshot_setting == "on" and self.form in SNAPSHOT_FORMS

    @property
    def derive_armed(self) -> bool:
        """Lane derive1: `DERIVE=on` in the loop's forms (`DERIVE_FORMS`)."""
        return self.derive_setting == "on" and self.form in DERIVE_FORMS

    @property
    def contract_armed(self) -> bool:
        """Lane contract1: `CONTRACT=on` or `refuse` in the loop's forms (`CONTRACT_FORMS`)."""
        return self.contract_setting in ("on", "refuse") and self.form in CONTRACT_FORMS

    @property
    def join_armed(self) -> bool:
        """Lane join2 of 10-16: `JOIN=repair` in the loop's forms (`JOIN_FORMS`)."""
        return self.join_setting == "repair" and self.form in JOIN_FORMS

    @property
    def finishcheck_armed(self) -> bool:
        """Lane finishcheck1: `FINISHCHECK=on` in the loop's forms (`FINISHCHECK_FORMS`)."""
        return self.finishcheck_setting == "on" and self.form in FINISHCHECK_FORMS

    def made_whole(self, lib: library.Library, index: str) -> str:
        """Lane first1 (`CARDS=whole`): the loop's first message's cards, the selection call's (its picks' and the
        pointer's, `selection.Shaped.order`) or, where it shaped nothing, the pointer's, each whole after the index
        within `library.WHOLE_TOKENS` in all (`library.made_whole`): where they do not fit, the index trimmed for them
        (the call's section by `selection.trims`, else `Library.trims`, the shelves no carded function lies on reduced
        first), and only where that leaves too little, the cards cut as today; logged, counted (`CARDS_KEYS`), and the
        system prompt measured again where the call was consulted. The index returned (`self.index` too); unarmed,
        `index` as it came, the functions carded kept for `cards_whole`. Lane libnever1's `LIB=never` arms it whatever
        `CARDS` says, and after the fit takes the lookup out of the index's head and a cut card's last line
        (`library_unsaid`, `cards_unsaid`)."""
        shaped = self.selection.result if self.form != "state" else None
        self.carded_order = list(shaped.order) if shaped is not None else [function for function, _ in self.fired]
        if not self.whole_armed:
            return index
        head = selection.CARDS_HEAD if shaped is not None else library.CARDS_HEAD
        trims = selection.trims(index) if shaped is not None else \
            lib.trims(index, {function.shelf for function in self.carded_order})
        made = self.whole = library.made_whole(self.carded_order, index, trims, head=head)
        self.index, self.cards, self.cuts = made.index, made.cards, list(made.cuts)
        if self.lib_never:      # lane libnever1: the lookup named nowhere, the fit as `whole` made it
            self.index, self.cards = library_unsaid(made.index), cards_unsaid(made.cards)
        source = self.cards_source if self.cards_setting == "whole" else f"{self.lib_source} never"
        self.log("cards", f"whole ({source}): {self.cards_whole()} of {len(self.carded_order)} card(s) "
                          f"whole, {library.measured(made.cards):,} {measure.UNIT.name} with their head"
                          + (f"; the index trimmed for them by {made.given_up:,} B in {made.steps} step(s)"
                             if made.steps else "")
                          + (f"; at the bound ({library.whole_bound():,} {measure.UNIT.name} with the index) the "
                             f"cards cut as today" if made.cuts else ""))
        for cut in made.cuts:
            self.log("cards", f"past the bound: {cut}")
        if self.selection.consulted and self.selection.system_today is not None:
            self.selection.measured(measure.size(self.system(self.index)), self.selection.system_today)
        return self.index

    def cards_whole(self) -> int:
        """The functions the first message cards whole (`CARDS_KEYS`), under either setting: each whose card, nothing
        cut, the cards carry."""
        return sum(function.whole_card() in self.cards for function in self.carded_order)

    @property
    def describing(self) -> bool:
        """Lane first1: `INSPECT=described` in the loop's forms (`INSPECT_FORMS`), and lane putaway1's `never`, whose
        look still asked is answered alike (`inspected`)."""
        return self.inspect_setting in ("described", "never") and self.form in INSPECT_FORMS

    @property
    def never_armed(self) -> bool:
        """Lane putaway1: `INSPECT=never` in the loop's forms, `inspect` withheld from every request."""
        return self.inspect_setting == "never" and self.form in INSPECT_FORMS

    def described_parts(self) -> dict[str, str]:
        """The input files the loop's first message describes, each with its description as carried there
        (`commands.carried`), read once the description is built; {} before."""
        if self.parts is None and self.inputs_text is not None and self.unit is not None:
            self.parts = commands.carried(self.inputs_text, self.unit.data_dir)
        return self.parts or {}

    def inspected(self, path: str, rows, tail: str) -> str:
        """`inspect`'s observation: under `INSPECT=described` (lane first1) an input file the loop's first message
        describes answered from that description (`commands.Workspace.answered`: the same observation, from the desk;
        `inspect_local`), any other as before (`inspect_sent`). No tool call is a request: the House is asked only by
        the loop's next request, as after any call."""
        local = self.ws.answered(path, self.described_parts(), rows, tail) if self.describing else None
        self.counts["inspect_local" if local is not None else "inspect_sent"] += 1
        return local if local is not None else self.ws.inspect(path, rows, tail)

    def offered(self) -> list[dict]:
        """The tools a request offers: all of them, but under `INSPECT=described` (lane first1) without `inspect` while
        it could show nothing the loop's first message lacks: every input file described there and no file yet under
        work/ or in the output (`inspect_withheld` counts the requests sent so), so the request goes to the script;
        under `INSPECT=never` (lane putaway1) without `inspect` on every request; and under `REDIRECT=write` (lane
        redirect1), from the waste state's declaration until a file is saved, without the tools it puts away
        (`put_away`); under lane libnever1's `LIB=never` without `lib` on every request. Under lane fixdef1's
        `WRITE=defs` the write tool's words say a fix by definition (commands.TOOLS_DEFS)."""
        tools = commands.TOOLS_DEFS if self.write_defs else commands.TOOLS
        withheld = set(self.put_away()) if self.redirecting else set()
        if self.lib_never:
            withheld.add("lib")
        if self.never_armed:
            self.counts["inspect_withheld"] += 1
            withheld.add("inspect")
        elif self.describing and self.ws is not None and self.inputs_text is not None:
            made = any(path.is_file() for root in (self.ws.dir, self.out) if root.is_dir() for path in root.rglob("*"))
            if not made and self.every_described():
                self.counts["inspect_withheld"] += 1
                withheld.add("inspect")
        if not withheld:
            return tools
        return [tool for tool in tools if tool["function"]["name"] not in withheld]

    def every_described(self) -> bool:
        """Whether the loop's first message describes every input file (`described_parts`: a file omitted past the
        description's bound, or a portrait cut at it, is not described); False before the description is built."""
        if self.unit is None or self.inputs_text is None:
            return False
        files = [path.relative_to(self.unit.data_dir).as_posix() for path in sorted(self.unit.data_dir.rglob("*"))
                 if path.is_file()] if self.unit.data_dir.is_dir() else []
        parts = self.described_parts()
        return all(name in parts for name in files)

    def short_clock(self) -> tuple[bool, str]:
        """Lane first1 (`SELECT_BOUND=short`, agent/n11/selection.py): whether the clock now buys fewer than
        `SHORT_CLOCK_REQUESTS` requests at the measured pace, else the platform's (`PACE_PRIOR_S`: lane pace1's rule,
        read as the selection call is sent), and the log's words."""
        measured = self.pace()
        pace = measured if measured is not None else PACE_PRIOR_S
        bought = self.bought(pace)
        short = bought < SHORT_CLOCK_REQUESTS
        return short, (f"{max(self.left(), 0):.0f} s buy {bought:.1f} request(s) at "
                       f"{'the measured' if measured is not None else 'the prior'} {pace:.0f} s a request and "
                       f"{self.run_expected():.0f} s a run, " + (
                           f"under {SHORT_CLOCK_REQUESTS}: {selection.MAX_TOKENS:,} tokens, thinking off" if short
                           else f"not under {SHORT_CLOCK_REQUESTS}: the call as the route's"))

    def task_part(self, task, pathmap, data: str, named: tuple[str, ...]) -> str:
        """The loop's first message up to its clock (`task_text`, the input files described now); under lane prompt1's
        `taskfirst` in design B's order, How to work and Library after it (`prompts.taskfirst_sections`), step 1's
        `inspect` clause left out where `inspect` is withheld from every request (`STEP_ONE`, as `told` says it); under
        `rulesafter` today's order, then the runtime paragraph and the defaults list (`prompts.rules_after`); under
        lane clean1's `clean` its sections as `cleaned` says them."""
        word = self.prompt_word
        control = self.control(self.index) if word in ("taskfirst", "rulesafter") else None
        first = control is not None and word == "taskfirst"
        said = task_text(task, pathmap, data, self.described() if self.ws is not None else "", named,
                         self.firstout_armed, self.contract_armed, prompts.TASKFIRST_ORDER if first else None,
                         self.cleaned if word == "clean" else None)
        if control is None:
            return said
        return said + (prompts.taskfirst_sections(control, STEP_ONE if self.never_armed else None) if first
                       else prompts.rules_after(control))

    def described(self) -> str:
        """The input files described (lane inputs1): `standard`, the workspace's (commands.Workspace.described), else
        agent/n11/inputs.py's under its bound; built once, its bytes counted (`inputs_bytes`)."""
        if self.inputs_text is None:
            task = self.unit
            self.inputs_text = self.ws.described() if self.inputs_setting != "rich" else \
                inputs.described(task.data_dir, task.instruction, task.canaries)
            if self.never_armed:
                self.inputs_text = self.inputs_text.replace("- not described here (`inspect` shows them): ",
                                                          "- not described here (past the description's bound): ", 1)
        return self.inputs_text

    def carried_inputs(self) -> str:
        """The input files described as the form's requests carry them (`inputs_bytes`): form `state`'s INPUT FILES
        under `standard` (the selection call's own description, the loop's, is not the state's: lane state4), else
        `described`'s as built ("" where none was)."""
        if self.form == "state" and self.inputs_setting != "rich" or self.inputs_text is None:
            return self.state_inputs
        return self.inputs_text

    def first_turn(self, task, pathmap, data: str, named: tuple[str, ...], pointer: str) -> str:
        """The loop's first message: `task_part`, the conventions card where it is on (`card_part`), the worked example
        where it is armed (`example_part`), the clock, the budget, the conventions' sentence and the pointer's line;
        pace, sketch and clause text follow at assembly. Under lane clean1's `clean` no clock (the system prompt says
        the run's limit, the budget the unit's), no conventions' sentence where step 1 says the lags it adds
        (`prompts.clean_parts`), and the pointer's line without its call imperative (`prompts.CLEAN_POINTER`)."""
        control = self.control(self.index) if self.prompt_word == "clean" else None
        said = self.task_part(task, pathmap, data, named) + self.card_part(task) + self.example_part(task, named)
        if control is None:
            return (said + f"# Time\n\nEach run is stopped after {RUN_S:.0f} seconds; the unit ends "
                    f"{max(self.left(), 0):.0f} seconds from now.\n\n# Budget\n\n"
                    f"{self.budget()}\n\n{CONVENTIONS}\n" + (f"\n{pointer}\n" if pointer else ""))
        lags = "lags" not in prompts.clean_parts(control, self.cut_setting, TIMING_LINES, FENCED_EXAMPLE)[1]
        if pointer and pointer.count(prompts.CLEAN_POINTER[0]) == 1:
            pointer = pointer.replace(*prompts.CLEAN_POINTER, 1)
        elif pointer:
            self.log("prompt", "clean: pointer left as today's (the anchor not met)")
        return (said + f"# Budget\n\n{self.budget()}\n" + ("" if lags else f"\n{CONVENTIONS}\n")
                + (f"\n{pointer}\n" if pointer else ""))

    def cleaned(self, sections: dict[str, str]) -> dict[str, str]:
        """Lane clean1: the first message's sections as `clean` says them: Paths without its "nothing anywhere else
        but /tmp" tail (`prompts.CLEAN_PATHS`; the system prompt's must list allows work/ beside it), and the input
        files described without a file's portrait that repeats the tree its input data shows
        (`prompts.repeated_portraits`); an edit not made logged."""
        said = dict(sections)
        if said["Paths"].count(prompts.CLEAN_PATHS[0]) == 1:
            said["Paths"] = said["Paths"].replace(*prompts.CLEAN_PATHS, 1)
        else:
            self.log("prompt", "clean: paths left as today's (the anchor not met)")
        described = said["Input files, described"]
        if described and self.unit is not None:
            said["Input files, described"], dropped = prompts.repeated_portraits(
                described, said["Input data"], commands.portraits(described, self.unit.data_dir))
            if dropped:
                self.log("prompt", f"clean: the described portrait of {', '.join(dropped)} left out (the input data "
                                   "shows its tree)")
        return said

    @property
    def example_armed(self) -> bool:
        """Lane example1: `EXAMPLE=nearest` in the loop's forms (`EXAMPLE_FORMS`)."""
        return self.example_setting != "off" and self.form in EXAMPLE_FORMS

    def example_part(self, task, named: tuple[str, ...]) -> str:
        """Lane example1 (`EXAMPLE=nearest`, agent/n11/example.py): the worked example's section of the loop's first
        message, after the output files and the conventions card and before the clock: the corpus's entry nearest this
        task (`example.nearest`, never this task's kin nor `EXAMPLE_EXCLUDE`'s), within its own bound
        (`example.shown`); built once and logged; "" where the switch is off or not armed, or no entry overlaps."""
        if not self.example_armed:
            return ""
        if self.example_shown is None:
            entries, malformed = example.load(EXAMPLES)
            meta = task.card.get("metadata") if isinstance(task.card, dict) else None
            name = str((task.card.get("task") or {}).get("id", "")) if isinstance(task.card, dict) else ""
            category = str(meta.get("category", "")) if isinstance(meta, dict) else ""
            entry, score = example.nearest(name, task.instruction, category, entries, EXAMPLE_EXCLUDE, outputs=named)
            self.example_shown = example.shown(entry, score)
            shown = self.example_shown
            self.log("example", (f"{shown.unit} at overlap {shown.overlap:.3f}, {shown.tokens:,} "
                                 f"{measure.UNIT.name}" + (f", cut: {', '.join(shown.cut)}" if shown.cut else "")
                                 if shown.unit else "none: " + ("no corpus" if not entries else "no entry overlaps"))
                     + f"; {len(entries)} entries" + (f", {malformed} malformed line(s)" if malformed else "")
                     + (f"; excluding {','.join(sorted(EXAMPLE_EXCLUDE))} ({EXAMPLE_EXCLUDE_SOURCE})"
                        if EXAMPLE_EXCLUDE else ""))
        return f"{self.example_shown.text}\n\n" if self.example_shown.text else ""

    def card_part(self, task) -> str:
        """Lane convcard1 (`CONVCARD=on`): the conventions card's section of the loop's first message, after the
        output files and before the clock (agent/n11/conventions.py `card`: the instruction's convention words, each a
        question, and the parameters files' keys), built once, logged and counted; "" where the switch is off or the
        instruction holds none of the words and no parameters file is there. The loop's first message is the forms of
        `CONVCARD_FORMS` alone: form `state` builds its own."""
        if self.convcard_setting != "on":
            return ""
        if self.convcard is None:
            self.convcard = conventions.card(task.instruction, conventions.parameters(task.data_dir))
            self.convcard_size = measure.size(self.convcard.text)
            self.log("conventions", f"{len(self.convcard.terms)} line(s), {self.convcard_size:,} "
                                    f"{measure.UNIT.name}" + (f", {self.convcard.left} past the bound"
                                                              if self.convcard.left else "")
                     + (f": {', '.join(self.convcard.terms)}"[:300] if self.convcard.terms else ""))
        return f"{self.convcard.text}\n\n" if self.convcard.text else ""

    def outgoing(self) -> list[dict]:
        """The messages as sent: tool results and observations older than the last `KEEP_TURNS` turns cut to their
        first line (the first user turn whole)."""
        turns = [k for k, message in enumerate(self.messages) if message["role"] == "assistant"]
        if len(turns) <= KEEP_TURNS:
            return list(self.messages)
        edge = turns[-KEEP_TURNS]
        return [{**message, "content": first_line(message["content"])}
                if 1 < k < edge and message["role"] in ("tool", "user") and message["content"] != self.sketch_kept
                else message for k, message in enumerate(self.messages)]

    def record(self, text: str, calls: list[Call], results: list[str], native: bool) -> None:
        """A turn's reply and its observations into the conversation, in the form the mode sends."""
        if native and calls and all(call.id and call.name in commands.NAMES + ("unparsed",) for call in calls) \
                and not any(call.name in ("cut", "unparsed") for call in calls):
            self.messages.append({"role": "assistant", "content": text or None, "tool_calls": [
                {"id": call.id, "type": "function", "function": {"name": call.name, "arguments": call.raw}}
                for call in calls]})
            self.messages += [{"role": "tool", "tool_call_id": call.id, "content": result}
                              for call, result in zip(calls, results)]
            return
        self.messages.append({"role": "assistant", "content": text or "(no text)"})
        self.messages.append({"role": "user", "content": "\n\n".join(
            f"[{call.describe()}]\n{result}" for call, result in zip(calls, results)) or "(nothing ran)"})

    # ------------------------------------------------------------------------------------------- the request

    def ask(self, forced: bool = False, pair: bool = False) -> object:
        """The loop's request; `forced` (lane clock1's forced ask) requires the write tool where the tools are sent;
        `pair` (lane repairpair1) sends B beside it, its system prompt this request's (`repair_start`). Its
        seconds on the unit's clock, from the send to the reply, are kept (lane pace1: the pace's sample). While lane
        fallback1's reserve is held, it is bounded by the time left before the reserve, and its reply's tokens and
        seconds size the reserve (lane fallback2: `fallback.Resort.bounded`, `observe`)."""
        tools = self.mode == "tools" and not self.client.usage.tool_refused
        if self.mode == "tools" and not tools:
            self.mode, self.mode_refused = "fenced", True
            self.messages = [{"role": "system", "content": self.worded(self.index)}, *textual(self.messages[1:])]
            self.log("mode", "the route refused the tools: the fenced form from here")
        self.client.kind = "review" if self.review_from is not None else \
            "sketch" if self.sketch_due and self.sketch_mode == "turn" else "loop"
        self.show(self.messages[0]["content"])
        own = self.review_from is not None and not self.review_asked or self.sketch_due and self.sketch_mode == "turn" \
            or bool(self.sketch_think())    # lane thinkwrite1: the review's request and the sketch's keep their own
        kind = self.request_kind()          # switches; any other request, the write replies' switch by its kind
        think = {} if own else self.write_think(kind)
        self.repeats_asked = self.counts["write_repeats"]
        asked = {}                      # lane inputs1: the review request alone asks thinking, and only where set
        if self.review_from is not None and not self.review_asked:
            self.review_asked = self.review_reply = True
            if self.review_thinking != "off":
                asked = {"thinking": "low"}
                self.log("review", f"the review request asks thinking {self.review_thinking}")
        self.counts["redirect_after"] += self.redirect_at is not None     # lane redirect1: a request after it
        offered = self.offered() if tools else None
        messages = self.said(offered)                   # lane putaway1: the system prompt follows the tools
        if pair:
            try:
                self.repair_start(messages[0]["content"], offered, think)
            except Exception as exc:  # noqa: BLE001 - preparation of B cannot cost A its request
                with guard.WRITING:
                    self.repair.discard(self.client)
                    shutil.rmtree(self.work / repair.WORK / str(self.repair.number), ignore_errors=True)
                self.log("repair", f"not sent ({type(exc).__name__}): the loop's request stands")
        began, moved = time.time(), self.moved()
        bounded = self.resort.bounded(self.left())      # lane fallback2: while the reserve is held, none runs into it
        if forced and tools:
            reply = self.client.converse(messages, tools=offered, tool_choice=WRITE_CHOICE, **asked, **think,
                                         **bounded)
        else:
            reply = self.client.converse(messages, tools=offered, **{**self.sketch_think(), **asked, **think,
                                                                      **bounded})
        if think:
            self.write_thought(kind, reply)
        seconds = max(time.time() - began - (self.moved() - moved), 0.0)   # an untimed wait left out
        self.request_seconds.append(seconds)
        if resized := self.resort.observe(reply.completion_tokens, seconds):    # lane fallback2: the reserve sized
            self.log("fallback", resized)
        return reply

    def said(self, offered: list[dict] | None) -> list[dict]:
        """Lane putaway1: the messages a loop request sends (`outgoing`), its system prompt saying the tools the request
        offers (`told`, by `said_tools`); a change of what it says logged, and the keys' state kept (`SYSTEM_KEYS`).
        The first one's bytes as sent are kept (lane clean1's `prompt_bytes_first`)."""
        messages = self.outgoing()
        unsaid, away = self.said_tools(offered)
        system = told(messages[0]["content"], self.mode, unsaid, away, self.write_defs, self.prompt_word == "clean")
        if system != messages[0]["content"]:
            messages[0] = {**messages[0], "content": system}
        if self.prompt_first_bytes is None:     # lane clean1: the first loop request's size, as sent
            self.prompt_first_bytes = sum(prompts.size(message["content"]) for message in messages
                                          if isinstance(message.get("content"), str))
        if (unsaid, away) != (self.system_said or ((), ())):
            self.log("system", "the system prompt describes "
                               + ", ".join(name for name in commands.NAMES if name not in unsaid + away)
                               + (f"; {' and '.join(unsaid)} left unsaid" if unsaid else "")
                               + (f"; {' and '.join(away)} said put away" if away else ""))
        self.system_said = (unsaid, away)
        self.system_first = system if self.system_first is None else self.system_first
        self.counts["system_changed"] += system != self.system_first
        return messages

    def said_tools(self, offered: list[dict] | None) -> tuple[tuple[str, ...], tuple[str, ...]]:
        """Lane putaway1: the tools a request's system prompt leaves unsaid and those it says are put away, in
        `commands.NAMES`' order, from the tools the request offers (`offered`; None in the fenced form, whose commands
        the prompt alone offers): a tool withheld (`INSPECT=described` or `never`) unsaid, and from the waste state's
        declaration until a file is saved the tools put away said put away, `inspect` among them where `described`
        withholds it too, but never under `never`. Under `never` a fenced request is counted here as sent without
        `inspect` (`offered` counts the tools form's). Lane libnever1's `LIB=never` leaves `lib` unsaid on every
        request, never said put away."""
        gone = ({"inspect"} if self.never_armed else set()) | ({"lib"} if self.lib_never else set())
        if offered is None:
            self.counts["inspect_withheld"] += "inspect" in gone
            missing = gone | (set(self.put_away()) if self.redirecting else set())
        else:
            missing = set(commands.NAMES) - {tool["function"]["name"] for tool in offered}
        away = tuple(name for name in commands.NAMES if name in missing - gone) if self.redirecting else ()
        return tuple(name for name in commands.NAMES if name in missing and name not in away), away

    def sketch_think(self) -> dict:
        """The request's thinking where it carries the sketch's ask and the setting is `low` (addendum 4): the
        client's `thinking="low"` (model.THINKING_LOW); else nothing, every other request off as before."""
        return {"thinking": "low"} if self.sketch_due and self.sketch_thinking == "low" else {}

    def state_kind(self) -> str:
        """What the last run left, as the short clock's ask reads it (lane pace1): `finish` where it was clean (exit 0,
        the checks passing, every named output file among them, and no finding), `write` where work/main.py is not
        written, else `fix`."""
        return "finish" if self.last_clean else "fix" if (self.ws.dir / "main.py").is_file() else "write"

    def request_kind(self) -> str:
        """Lane thinkwrite1: the loop request's kind by the harness's state at its send (`thinking.kind_of`): a held
        head's rest, the answer to an unchanged re-send, the answer to a finish not taken, else `state_kind`."""
        return thinking.kind_of(self.held is not None, self.counts["write_repeats"] > self.repeats_asked,
                                "finish" in self.turn_calls, self.state_kind())

    def write_think(self, kind: str) -> dict:
        """Lane thinkwrite1: the request's `thinking` argument where the switch asks it (`thinking.Ledger.asks`), the
        unit's first write still to come (`first_write`) where it says `first`; else nothing, thinking off."""
        return self.thought.asks(kind, self.first_write is None)

    def write_thought(self, kind: str, reply) -> None:
        """Lane thinkwrite1: the reply to a request that asked, counted (`thinking.Ledger.observe`), logged and
        traced."""
        reasoning, completion = int(reply.reasoning_tokens or 0), int(reply.completion_tokens or 0)
        said = self.thought.observe(reasoning, completion, reply.finish_reason == "length")
        self.log("thinking", f"the {kind} request asked {self.thought.budget}: {said}")
        trace.event("write_thinking", kind=kind, budget=self.thought.budget, reasoning=reasoning,
                    completion=completion, finish=reply.finish_reason, dropped=self.thought.dropped)

    def calls(self, reply, turn: int) -> tuple[list[Call], bool]:
        """A reply's calls and whether they are native; a native reply with none read in the fenced form, under
        `CUT=continue` leniently (lane join2: `commands.from_fenced`'s `lenient`, a block holding one directive alone
        counted `directive_in_block`). Whole file blocks discarded by the reader count as `blocks_unsaved`;
        directives read as calls and cut blocks do not."""
        cap = self.max_output() if self.redirect_armed else None    # lane redirect1: a reply at the cap is a cut
        if self.mode == "tools" and reply.tool_calls:
            return commands.from_tools(reply, turn, cap), True
        if self.mode == "tools" and "<tool_call>" in (reply.text or ""):
            return commands.from_tools(reply, turn, cap), True
        found = commands.from_fenced(reply.text, turn, self.continuing)
        items = [item for item in commands.fenced(runner.THINK.sub("", reply.text or ""))
                 if isinstance(item, commands.Fenced)]
        self.named_counted([item.path for item in items if item.named and item.closed and not item.void])
        self.counts["blocks_unsaved"] += sum(
            item.closed and bool(item.void) and (item.path is not None or item.tag in commands.PYTHON or not item.tag)
            and not (self.continuing and commands.block_directive(item) is not None) for item in items)
        if self.continuing:
            self.counts["directive_in_block"] += sum(commands.block_directive(item) is not None for item in items)
            blocks = [item for item in items if not item.closed]
            for call in found:
                if call.name == "write" and call.cut:
                    block = next((item for item in blocks if self.ws.local(item.path or MAIN_FILE)[1] ==
                                  self.ws.local(call.arguments["path"])[1]), None)
                    if block is not None:
                        call.arguments["text"] = "\n".join(block.lines)
        if found and self.at_cap(reply) and not any(call.cut for call in found):
            found[-1].cut, found[-1].problem = True, None
        if not found and self.at_cap(reply):
            found = [Call("cut", {}, f"call_{turn}_1", "{}", None, True)]
        return found, False

    def named_counted(self, paths: list[str]) -> None:
        """Lane fencedpath1: the whole blocks a labelled line named another file by (commands.labelled), each counted
        `blocks_named` where the workspace path check accepts it, `blocks_named_outside` where it refuses it
        (including a dotfile under work/). These count dispatches, not successful writes."""
        for path in paths:
            self.counts["blocks_named" if self.ws.local(path)[0] is not None else "blocks_named_outside"] += 1

    def max_output(self) -> int:
        """The client's cap on a reply's completion tokens, as `wrote_first` reads it."""
        return getattr(self.client, "max_output", model.MAX_OUTPUT_TOKENS)

    def cut_shape(self, reply) -> None:
        """Lane redirect1: a reply at the cap (its finish `length`, or its completion at the client's cap) counted by
        what arrived of its call, under either setting (`CUT_SHAPES`): native calls or a call written as text, the
        route saying it finished (`calls`) or cut (`length`); a fenced block (`fenced`); or nothing of either
        (`empty`: no head to keep). Under `REDIRECT=write` the trace's `cut_shape` event says what the route's message
        held, its fields and sizes, never its text, and whether its last fenced block was closed (lane fenced1: a cut
        that fell after a whole block, whose write the loop reads as cut all the same; None where it held no block)."""
        text, tokens = reply.text or "", int(reply.completion_tokens or 0)
        if reply.finish_reason != "length" and tokens < self.max_output():
            return
        called = bool(reply.tool_calls) or any(mark in text for mark in TEXT_CALL)
        shape = ("length" if reply.finish_reason == "length" else "calls") if called else \
            "fenced" if "```" in text else "empty"
        self.counts[f"cut_shape_{shape}"] += 1
        if self.redirect_armed:
            message = reply.message if isinstance(reply.message, dict) else {}
            blocks = [item for item in commands.fenced(runner.THINK.sub("", text)) if isinstance(item, commands.Fenced)]
            trace.event("cut_shape", shape=shape, finish=reply.finish_reason, completion_tokens=tokens,
                        text_bytes=len(text.encode()), calls=[[call.name, len(call.raw or "")]
                                                               for call in reply.tool_calls],
                        message={key: len(json.dumps(value, default=str)) for key, value in message.items()},
                        block_closed=blocks[-1].closed if blocks else None)

    def at_cap(self, reply) -> bool:
        """Whether the cap cut a reply: its finish `length`, and under `REDIRECT=write` (lane redirect1) its completion
        at the client's cap whatever finish the route said (the pooled route's `tool_calls`, its last call's arguments
        cut: docs/research/requests-1013.md, section 4)."""
        return reply.finish_reason == "length" or (self.redirect_armed and
                                                   int(reply.completion_tokens or 0) >= self.max_output())

    def loop(self) -> str:
        """The loop's turns until a stop: no request or `ASK_S` left; a finish that ends the unit; a run past its
        reserve (`past_reserve`); two replies in a row with nothing runnable; `NO_CALL_CAP` replies in a row with no
        call, but while two replies remain the forced ask first (`FORCED`, the write tool required: lane clock1), and
        where a last call is due that instead; and the last call (`LAST_CALL`, lane clock1), whose reply's work/main.py
        runs with all the time left and ends the unit."""
        streak, began, forced = 0, False, False
        while True:
            if self.repair.pending is not None and self.repair_settle():    # lane repairpair1: B kept, its turn the
                streak, forced = 0, False                                     # loop's last
            if self.requests_left() < 1:
                return self.stop(self.spent(), "requests")
            if self.left() < ASK_S:
                return self.stop(f"{max(self.left(), 0):.0f} s left, under a request's {ASK_S:.0f} s", "clock")
            if not began:
                self.carried()          # the loop's first request: the loop band's request 1, or after N0's
                began = True
            if self.sketch_note:        # form `sketch`'s picks (addendum 2): after the message the sketch's reply got,
                if self.sketch_inside:  # in `turn`'s line's message, else a message of their own; whole for the
                    self.messages[-1] = {**self.messages[-1],    # unit's life (addendum 3)
                                         "content": f"{self.messages[-1]['content']}\n\n{self.sketch_note}"}
                else:
                    self.messages.append({"role": "user", "content": self.sketch_note})
                self.sketch_kept, self.sketch_note = self.messages[-1]["content"], ""
            self.repair_before = self.repair_state() if self.repair.armed else None
            last = self.last_call_due()
            if last:
                self.call_last()
            elif self.short_due:        # lane pace1: the short-clock mode's ask after the last turn's observations
                self.short_ask()
            self.short_due = False
            if self.redirect_due and not last:      # lane redirect1: the demand closes the last observation
                self.redirect_ask()
            self.redirect_due = False
            if self.repeat_due and not last:        # lane repeat1: the changed-repair ask closes it
                self.repeat_ask()
            self.repeat_due = False
            pair = self.repair_due(last)        # lane repairpair1: a repair turn, the rewrite sent beside it
            try:
                reply = self.ask(forced, pair)
            except model.ModelError as exc:
                self.log("request", f"no reply ({type(exc).__name__}: {logged(str(exc))[:200]})")
                if last:
                    self.last_sent(None)
                if isinstance(exc, model.BoundReached):     # lane fallback2: cut where the reserve begins
                    self.resort.cuts += 1
                    return self.stop("the request ran to the reserve's line, cut there", "clock")
                return self.stop(f"the House did not answer ({type(exc).__name__})")
            if last:
                self.last_sent(reply)
            forced = False
            self.counts["turns"] += 1
            self.paced_turn()           # lane pace1: the pace, the requests bought, the mode's entry
            self.replied("review" if self.review_from is not None else "loop", reply)
            if self.review_reply:       # the review request's reply (lane inputs1)
                self.review_reply = False
                self.counts["review_reasoning_tokens"] = int(reply.reasoning_tokens or 0)
                self.counts["review_cut"] = int(reply.finish_reason == "length")
            turn = self.counts["turns"]
            self.reply_text = reply.text or ""
            found, native = self.calls(reply, turn)
            if self.held is not None:   # lane review2, `CUT=continue`: the rest of the file whose head was kept
                found, native = self.resumed(reply, found, native, turn)
            self.turn_calls = tuple(call.name for call in found)     # lane thinkwrite1: `request_kind` reads it
            if self.audit_at is not None and turn == self.audit_at + 1:     # lane audit1: the reply after the audit
                self.audit_read(reply, found)
            self.log("request", f"turn {turn}: {reply.prompt_tokens:,} prompt and {reply.completion_tokens:,} "
                                f"completion tokens, finish {reply.finish_reason}; "
                                + (", ".join(call.describe() for call in found) or "no call"))
            if self.nocall_due:         # lane cap1: the reply to an ask (`ask`)
                self.nocall_due = False
                self.counts["nocall_recovered"] += bool(found) and not self.lost(found)
            if self.sketch_due:
                self.sketch_due = False
                if self.sketched(reply, found, native) and not last:
                    self.replies[-1] = ("sketch", *self.replies[-1][1:])
                    trace.event("turn", turn=turn, calls=[], ended=False)
                    self.messages.append({"role": "assistant", "content": reply.text or "(no text)"})
                    self.messages.append({"role": "user", "content": f"{SKETCH_NEXT[self.mode]}\n{self.budget()}"})
                    self.short_due = self.short
                    continue
            self.wrote_first(reply, any(call.name in ("write", "cut", "unparsed") for call in found))
            self.derive_read(reply, found)
            self.cut_shape(reply)
            if self.redirect_at is not None:    # lane redirect1: the replies after the declaration
                self.counts["redirect_cut"] += self.at_cap(reply)
                self.counts["redirect_nocall"] += not found
            if not found or last:
                self.redirected(found)
            if last:
                return self.last_turn(found, turn)
            if self.asking and (not found or self.lost(found)):    # lane cap1: a call asked after every one
                ended, forced = self.ask_call(reply, found, turn)
                if ended:
                    return ended
                streak = 0
                self.redirect_due = self.redirecting and not found  # a cut's ask stands alone
                continue
            if not found:                       # only finish ends a unit (lane n11d), or a third reply with no call
                self.counts["no_call"] += 1
                self.no_call_streak += 1
                self.gave[turn] = "no call"
                capped = self.no_call_streak >= NO_CALL_CAP
                forced = capped and self.can_force()
                onward = forced or (capped and self.no_call_streak == NO_CALL_CAP and self.last_call_due())
                trace.event("turn", turn=turn, calls=[], ended=capped and not onward)
                if capped and not onward:
                    self.counts["no_call_capped"] = 1
                    return self.stop(f"{self.no_call_streak} replies in a row with no tool call"
                                     + (", the forced ask among them" if self.no_call_streak > NO_CALL_CAP else "")
                                     + ": the unit ends with the output as it stands", "no_call")
                if forced:
                    self.counts["prose_forced"] = 1
                    self.log("request", "the forced ask: the write tool required while two replies remain")
                self.messages.append({"role": "assistant", "content": reply.text or "(no text)"})
                self.messages.append({"role": "user", "content": f"{(FORCED if forced else NO_CALL)[self.mode]}\n"
                                                                 f"{self.budget()}"})
                streak = 0
                self.redirect_due = self.redirecting
                continue
            self.no_call_streak = 0
            cut = any(call.cut for call in found)
            self.counts["cut_replies"] += cut
            self.cut_streak = self.cut_streak + 1 if cut else 0
            bad = [call for call in found if call.problem and not call.cut]
            self.counts["tool_malformed"] += len(bad)
            self.counts["calls"] += len(found)
            then = self.implicit(found, not native)
            before = self.finish_writes(found, not native)  # lane finishrun1: the write each finish follows, run
            #                                                  before it
            results, ended = [], False
            for number, call in enumerate(found):
                if number in before and not ended:
                    results[before[number]] = self.after_write(
                        results[before[number]], str(found[before[number]].arguments.get("path", "")), finishing=True)
                    ended = bool(self.ending)
                if ended:
                    results.append("not run: the task ended at finish" if not self.ending else
                                   "not run: the task ended on the last run's output")
                    continue
                said, ended = self.execute(call, number == len(found) - 1 and then is None, number == then)
                results.append(said)
            if then is not None and not ended:
                results[then] = self.after_write(results[then], str(found[then].arguments.get("path", "")))
                ended = bool(self.ending)
            if self.stitching is not None:     # lane review2: the stitched write said as such
                results = [STITCHED.format(named=call.arguments["path"]) + said if call is self.stitching else said
                           for call, said in zip(found, results)]
                self.stitching = None
            trace.event("turn", turn=turn, calls=[call.describe() for call in found], ended=ended)
            if ended:
                return self.stop(self.ending or self.finish_said, "clock" if self.ending else "finish")
            self.record(reply.text, found, results, native)
            self.short_due = self.short and not cut     # a cut's ask stands alone
            self.redirected(found)
            self.redirect_due = self.redirecting and not cut
            self.repeat_turn(cut)                       # lane repeat1: the turn's recurrences answered
            runnable = [call for call in found if not call.problem and not call.cut]
            streak = streak + 1 if bad and not runnable else 0
            if streak > 1:
                return self.stop("two replies in a row with no call that could be run", "no_call")

    # ------------------------------------------------------------------------------- lane redirect1: the waste state

    @property
    def redirect_armed(self) -> bool:
        """`REDIRECT=write` in the loop's forms (`REDIRECT_FORMS`)."""
        return self.redirect_setting == "write" and self.form in REDIRECT_FORMS

    @property
    def redirecting(self) -> bool:
        """The waste state stands: declared (`redirected`) and no file saved since, nor before (it is declared with
        nothing written)."""
        return self.redirect_at is not None and not self.wrote

    def put_away(self) -> tuple[str, ...]:
        """The tools the waste state puts away: `inspect` and `lib`, but `lib` alone where an input file is not
        described (`every_described`), where `inspect` shows what the first message lacks. Under lane libnever1's
        `LIB=never`, which withholds `lib` all the way, `inspect` alone where every input is described, and nothing
        where an input is not or under `INSPECT=never`, which withholds `inspect` all the way."""
        away = ("inspect", "lib") if self.every_described() else ("lib",)
        if self.lib_never:
            return tuple(name for name in away if name == "inspect" and not self.never_armed)
        return away

    def redirect_shape(self) -> tuple[bool, bool]:
        """The demand's and the one-line answer's shape: whether `inspect` is put away with `lib`, and whether the
        system prompt carries cards."""
        return "inspect" in self.put_away(), bool(self.cards)

    def redirect_text(self, texts: dict[tuple[bool, bool], str]) -> str:
        """The demand or held answer, with only `lib` named under `INSPECT=never`."""
        text = texts[self.redirect_shape()]
        return text.replace("`inspect` and `lib` are", "`lib` is", 1) if self.never_armed else text

    def redirect_why(self) -> str:
        """The demand's first clause (`REDIRECT_WHY`, `redirect_text`); under lane libnever1's `LIB=never` the tools it
        names those put away (`LIB_NEVER_WHY`: `inspect`, or none, the clause then saying what the first message and
        the system prompt hold, or nothing)."""
        if self.lib_never:
            return LIB_NEVER_WHY[(self.every_described(), bool(self.cards), "inspect" in self.put_away())]
        return self.redirect_text(REDIRECT_WHY)

    def redirect_held(self) -> str:
        """The one-line answer to a look put away (`REDIRECT_HELD`, `redirect_text`); under lane libnever1's `LIB=never`
        `inspect`'s alone (`LIB_NEVER_HELD`), a lookup answered as not offered (`execute`)."""
        return LIB_NEVER_HELD[bool(self.cards)] if self.lib_never else self.redirect_text(REDIRECT_HELD)

    def redirected(self, found: list[Call]) -> None:
        """After a loop reply's calls ran: the waste state declared, once, where all hold under `REDIRECT=write`
        (docs/research/requests-1013.md, section 6, I2@4): nothing written yet (no `write` call, refused or not, and no
        reply cut at the cap: `first_write`), at least `REDIRECT_LOOKS` inspect-only replies so far (every call
        `inspect`), and it answers request `REDIRECT_AT` or a later one, whatever this reply's calls, the
        selection call's place counted whether it was sent or not. From here until a file is saved, `offered` puts
        `inspect` and `lib` away, a look still asked is answered in one line (`execute`) and each observation closes
        on the demand (`redirect_ask`); a reply at the cap is a cut under the switch throughout (`at_cap`)."""
        if found and all(call.name == "inspect" for call in found):
            self.redirect_inspects += 1
        if not self.redirect_armed or self.redirect_at is not None or self.first_write is not None or \
                self.redirect_inspects < REDIRECT_LOOKS or self.counts["turns"] + 1 < REDIRECT_AT:
            return
        self.redirect_at = self.counts["turns"] + int(self.selection.sent)
        self.redirect_left = self.left()
        away = self.put_away()
        self.log("redirect", f"declared at request {self.redirect_at} (reply {self.counts['turns']}), "
                             f"{max(self.redirect_left, 0):.0f} s left ({self.redirect_source}): inspect-only replies "
                             f"{self.redirect_inspects} with nothing written; "
                             + (f"{' and '.join(away)} put away" if away else "nothing to put away")
                             + " and the script asked until a file is saved")
        trace.event("redirect", at=self.redirect_at, reply=self.counts["turns"], left_s=round(self.redirect_left, 1),
                    inspects=self.redirect_inspects, put_away=list(away))

    def redirect_ask(self) -> None:
        """The demand (`REDIRECT_ASK` in the reply form, `REDIRECT_ASK_FIRSTOUT` under lane firstout1's `FIRSTOUT=on`,
        its first clause by `redirect_shape`, `redirect_why`) closing
        the last observation before the next request while the waste state stands; never after a cut, whose ask stands
        alone, nor with the last call's. Where the clause says nothing (lane libnever1) the demand opens the line."""
        kept = self.messages[-1]["content"] == self.sketch_kept
        said = (REDIRECT_ASK_FIRSTOUT if self.firstout_armed else REDIRECT_ASK)[self.mode].format(
            why=self.redirect_why()).lstrip(" ")
        self.messages[-1] = {**self.messages[-1], "content": f"{self.messages[-1]['content']}\n{said}"}
        if kept:
            self.sketch_kept = self.messages[-1]["content"]

    def saved(self, named: str) -> None:
        """A file the House's save wrote under work/ (`wrote`, lane files1): the unit's first saved file's seconds
        (`redirect_first_saved_s`), and after the declaration the files saved (`redirect_writes`), the first of which
        ends the waste state."""
        if self.first_saved is None:
            self.first_saved = time.time() - self.started - self.moved()
        if named == MAIN_FILE and self.prompt_first is None and self.ws is not None:    # lane prompt1's reached read
            self.prompt_first = (self.ws.dir / "main.py").read_text(encoding="utf-8", errors="replace") \
                if (self.ws.dir / "main.py").is_file() else None
            if self.prompt_attempt is None:     # saved with no write call before it (a continuation's join)
                self.prompt_attempt = self.prompt_first
                self.prompt_attempt_saved = self.prompt_first is not None
        if self.redirect_at is not None:
            if not self.wrote:
                self.log("redirect", f"{named} saved, {self.counts['redirect_after']} request(s) after the "
                                     "declaration: the tools offered again and the demand stopped")
                trace.event("redirect_end", path=named, after=self.counts["redirect_after"])
            self.redirect_saved[named] = None
        self.wrote[named] = None

    def redirect_keys(self) -> dict:
        """Lane redirect1's keys (`REDIRECT_KEYS`): the setting (`-` outside `REDIRECT_FORMS`); the request the waste
        state was declared at and the seconds left then (`-` where never); the loop's requests after it, the files
        saved after it, its replies at the cap, the looks answered in one line and the replies with no call; and the
        unit's first saved file's seconds from its start, under either setting (`-` where none)."""
        at = self.redirect_at
        return {"redirect_mode": self.redirect_setting if self.form in REDIRECT_FORMS else "-",
                "redirect_at": at if at is not None else "-",
                "redirect_left_s": f"{max(self.redirect_left, 0):.0f}" if at is not None else "-",
                "redirect_after": self.counts["redirect_after"], "redirect_writes": len(self.redirect_saved),
                "redirect_cut": self.counts["redirect_cut"], "redirect_looks": self.counts["redirect_looks"],
                "redirect_nocall": self.counts["redirect_nocall"],
                "redirect_first_saved_s": f"{self.first_saved:.0f}" if self.first_saved is not None else "-",
                **{key: self.counts[key] for key in CUT_READ_KEYS}}

    # ------------------------------------------------------------------------------------- lane cap1: the no-call ask

    @property
    def asking(self) -> bool:
        """`NOCALL=ask`, read by the loop alone (form `state` keeps its own reading of a reply with nothing in it)."""
        return self.nocall_setting == "ask"

    def lost(self, found: list[Call]) -> bool:
        """A reply whose one call the cap cut with nothing of it left to keep (`salvage` would say `CUT_CALL`): the
        NVIDIA route's cut tool call, which comes back empty (docs/research/failure-modes-1010.md, section 1)."""
        if len(found) != 1 or not found[0].cut:
            return False
        return found[0].name not in ("write", "cut", "unparsed") or not self.recovered(found[0]).get("text")

    def prose_file(self, text: str) -> str:
        """The file a reply's prose names (`prose_files`) for the ask: the first not yet under work/, else the last
        named; "" where it names none."""
        named = prose_files(text)
        fresh = [each for each in named if not (self.ws.dir / each.split("/", 1)[1]).is_file()]
        return fresh[0] if fresh else named[-1] if named else ""

    def ask_call(self, reply, found: list[Call], turn: int) -> tuple[str, bool]:
        """Lane cap1, `NOCALL=ask`: a reply with no call, or one whose cut call left nothing (`lost`), answered every
        time by one short ask for the call (`NOCALL_ASK`): the file its prose named, else after a cut the cut said
        (`NOCALL_CUT`, the `SHORT_STREAK`-th in a row asking one small file) and the first file of the split, else the
        next file or finish; on the tool form the write tool required (`WRITE_CHOICE`) while an output file is
        missing. The backstop: the `NO_CALL_CAP`-th ask in a row met by prose ends the unit (a cut call is a call
        tried: it neither counts nor ends the streak). The stop's reason ("" where the loop goes on) and whether the
        next request requires the write tool."""
        cut = bool(found)
        if cut:
            self.counts["cut_replies"] += 1
            self.cut_streak += 1
            self.no_call_streak = 0
        else:
            self.counts["no_call"] += 1
            self.no_call_streak += 1
        self.gave[turn] = "no call"
        ended = self.no_call_streak > NO_CALL_CAP
        trace.event("turn", turn=turn, calls=[call.describe() for call in found], ended=ended)
        if ended:
            self.counts["no_call_capped"] = 1
            return self.stop(f"{self.no_call_streak} replies in a row with no tool call, {self.no_call_streak - 1} of "
                             "them after an ask for one: the unit ends with the output as it stands", "no_call"), False
        named = "" if cut else self.prose_file(reply.text)
        call = NOCALL_CALL["cut" if cut else "named" if named else "next"][self.mode].format(named=named)
        short = (CUT_SHORT_FIRSTOUT if self.firstout_armed else CUT_SHORT.format(n=self.cut_streak)) \
            if self.cut_streak >= SHORT_STREAK else ""
        said = (self.split_words(NOCALL_CUT) + short if cut
                else NOCALL_ASK[self.mode]) + " " + NOCALL_NOW[self.mode].format(call=call)
        required = self.mode == "tools" and not self.client.usage.tool_refused and bool(self.ws.missing() or
                                                                                         self.unwritten())
        self.counts["nocall_asked"] += 1
        self.nocall_due = True
        self.log("request", "the call asked" + (f": {named}" if named else ", the cut said" if cut else "")
                 + ("; the write tool required" if required else ""))
        self.messages.append({"role": "assistant", "content": reply.text or "(no text)"})
        self.messages.append({"role": "user", "content": f"{said}\n{self.budget()}"})
        return "", required

    def sketch_ask(self) -> str:
        """Form `sketch`'s ask, before an armed clause (lane plan4; the owner's in316), in shapes, never a
        count: `SKETCH_ASK` in the switch's words, naming the cards where the pointer named functions, else the index;
        `turn`'s only with `SKETCH_FLOOR` requests and as many `ASK_S` left, since its reply is a request of the 20.
        "" where not asked."""
        if self.sketch_short():
            return ""
        if self.counts["short_first"] and self.sketch_mode == "turn":     # lane pace1: no turn of its own
            self.log("sketch", f"not asked ({self.sketch_mode}): the clock is short, the whole script first")
            return ""
        ask = SKETCH_ASK[self.sketch_mode].format(source=SKETCH_SOURCES[0] if self.cards else SKETCH_SOURCES[1],
                                                  how=SKETCH_HOW[self.mode])
        self.counts["sketch_asked"], self.sketch_due, self.sketch_thought = 1, True, self.sketch_thinking
        self.log("sketch", f"asked ({self.sketch_mode}, {self.sketch_source}): the loop's first message carries "
                           f"its ask, {len(ask.encode('utf-8')):,} B, naming "
                           + (SKETCH_SOURCES[0] if self.cards else SKETCH_SOURCES[1])
                           + f"; thinking {self.sketch_thinking} on its request")
        return f"\n{ask}\n"

    def sketch_short(self) -> str:
        """Why `turn`'s sketch is not asked, logged, "" where it is: its reply is a request of the budget, so not with
        fewer than `SKETCH_FLOOR` requests or as many `ASK_S` left; `inline`'s costs none."""
        short = "" if self.sketch_mode != "turn" else f"{self.requests_left()} request(s) left" \
            if self.requests_left() < SKETCH_FLOOR else f"{max(self.left(), 0):.0f} s left" \
            if self.left() < SKETCH_FLOOR * ASK_S else ""
        if short:
            self.log("sketch", f"not asked ({self.sketch_mode}): {short}")
        return short

    def sketched(self, reply, found: list[Call], native: bool) -> bool:
        """Form `sketch`'s first reply after its ask, read and never acted on (lane plan4): True where it is `turn`'s
        sketch, a reply with no native call and no file block (a directive line in its prose is not run), which the
        loop keeps as the House's first turn and answers `SKETCH_NEXT`; any other reply runs as the loop's
        (`sketch_skipped` under `turn`). Its picks' cards and unknown names are the next message's (`sketch_cards`);
        the keys `sketch_read`'s."""
        sketch = self.sketch_mode == "turn" and not native and not any(call.name == "write" for call in found)
        text = (reply.text or "") if sketch or native else (reply.text or "").split("```", 1)[0]
        picks, unknown = self.sketch_read(reply, text.strip(), sketch)
        self.sketch_note, self.sketch_inside = self.sketch_cards(picks, unknown), sketch
        self.sketch_said(reply, text.strip(), sketch, picks, unknown, "the sketch, kept as the first turn: " if sketch
                         else "a call came first: the reply runs as the loop's; " if self.sketch_mode == "turn" else
                         "the text before the call: ")
        return sketch

    def sketch_read(self, reply, text: str, sketch: bool) -> tuple[list[library.Function], list[str]]:
        """The sketch's keys under either form (`text` the sketch's, or the reply's text before its first call or
        block): `sketch_tokens` (the sketch's completion tokens; else the text at `library.BYTES_PER_TOKEN` a token,
        `inline`'s paragraph), `sketch_inline_bytes` (the text's bytes, under `inline`), `sketch_cut` (the reply cut
        at the cap), `sketch_names_lib`, `sketch_picks` and `sketch_picks_unknown` (`sketch_names`) and
        `sketch_skipped` (`turn`: a call or a file block came first; `inline`: no text before it). Its picks and the
        `qf.` names the index lacks. Lane cap1's `SKETCH_PICKS=named` (the loop's forms) reads the picks from `qf.`
        names and code spans alone; under either setting `sketch_picks_named` counts those and `sketch_picks_dropped`
        the picks a bare word alone named, which `named` drops."""
        size = len(text.encode("utf-8"))
        every, unknown = sketch_names(text, self.library)
        named = sketch_names(text, self.library, named=True)[0]
        picks = named if self.picks_setting == "named" and self.form != "state" else every
        self.counts.update(sketch_picks_named=len(named), sketch_picks_dropped=len(every) - len(named))
        self.counts.update(sketch_picks=len(picks), sketch_picks_unknown=len(unknown),
                           sketch_reasoning_tokens=int(reply.reasoning_tokens or 0),
                           sketch_cut=int(reply.finish_reason == "length"), sketch_names_lib=int(bool(picks)),
                           sketch_skipped=int(not sketch) if self.sketch_mode == "turn" else int(not text),
                           sketch_tokens=int(reply.completion_tokens or 0) if sketch else
                           -(-size // library.BYTES_PER_TOKEN))
        if self.sketch_mode == "inline":
            self.counts["sketch_inline_bytes"] = size
        return picks, unknown

    def sketch_said(self, reply, text: str, sketch: bool, picks: list[library.Function], unknown: list[str],
                    how: str) -> None:
        """The sketch's log line and trace event, after its cards are counted."""
        named = [function.name for function in picks]
        self.log("sketch", how + f"{len(text.encode('utf-8')):,} B, {self.counts['sketch_tokens']:,} tokens"
                 + (f" ({self.counts['sketch_reasoning_tokens']:,} reasoning)" if self.sketch_thought == "low" else "")
                 + ("; cut at the cap" if reply.finish_reason == "length" else "")
                 + ("; it names " + ", ".join(named) if named else "; it names no index function")
                 + ("; not in the index: " + ", ".join(unknown) if unknown else "")
                 + f"; {self.counts['sketch_cards_added']} card(s) added")
        trace.event("sketch", mode=self.sketch_mode, kept=sketch, bytes=len(text.encode("utf-8")), named=named,
                    unknown=unknown, added=self.counts["sketch_cards_added"])

    def unknown_lines(self, unknown: list[str]) -> list[str]:
        """Each `qf.` name the index lacks on its line (`SKETCH_UNKNOWN`), with the index's nearest names."""
        names = sorted(function.name for function in self.library.functions) if self.library is not None else []
        lines = []
        for name in unknown:
            near = difflib.get_close_matches(name.rsplit(".", 1)[-1], names, SKETCH_NEAREST, 0.5)
            lines.append(SKETCH_UNKNOWN.format(name=name, near="nearest there: " + ", ".join(
                f"`{each}`" for each in near) if near else "nothing near it there"))
        return lines

    @property
    def sketch_on(self) -> bool:
        """Whether this unit asks the sketch: form `sketch` always; form `state` where the switch is named, by the
        desk's variable or an image's file (addendum 3), so its bodies without the switch stay as they were."""
        return self.form == "sketch" or (self.form == "state" and self.sketch_source in (SKETCH_ENV, SKETCH_FILE))

    def sketch_cards(self, picks: list[library.Function], unknown: list[str]) -> str:
        """Addendum 2 (the owner's in330): what the loop's next message carries of the sketch's names, "" where
        nothing: the cards of its picks the pointer did not point to (`SKETCH_CARDS_HEAD`; `Library.carded`'s trim)
        within what the cards' bound (`library.card_bound()`) leaves after the pointer's cards, which stand first and
        whole in the system prompt; a pick whose card does not fit named on one line (`SKETCH_LEFT`); each name the
        index lacks on a line with the index's nearest names (`SKETCH_UNKNOWN`, `SKETCH_NEAREST` of them), so it is
        never called. Under lane libnever1's `LIB=never` no lookup is named (`cards_unsaid`, `SKETCH_LEFT_NEVER`)."""
        pointed = {function.name for function, _ in self.fired} | self.selection.carded   # lane select2's too
        new = [function for function in picks if function.name not in pointed]
        bound = library.card_bound()
        room = max(bound - library.measured(self.cards), 0) if self.cards else bound
        text, cuts = library.Library.carded([(function, function.name) for function in new], room,
                                            head=SKETCH_CARDS_HEAD) if new else ("", [])
        left = [function for function in new if any(cut.startswith(f"{function.call}: left out") for cut in cuts)]
        for cut in cuts:
            self.log("cards", f"the sketch's {cut}")
        self.card_reached((text,), new)         # lane called1: carried by the next message
        self.counts["sketch_cards_added"] = len(new) - len(left)
        lines = [cards_unsaid(text) if self.lib_never else text] if text else []    # lane libnever1: no lookup named
        if left:
            lines.append((SKETCH_LEFT_NEVER if self.lib_never else SKETCH_LEFT).format(
                named=", ".join(f"`{function.call}`" for function in left)))
        return "\n".join(lines + self.unknown_lines(unknown))

    def stop(self, reason: str, kind: str = "") -> str:
        """The unit's end said and logged; `kind`, the loop's, one of lane fallback1's `fallback.REASONS`. A pair still
        out is settled first, B's files kept where the keep rule takes them (lane repairpair1)."""
        if self.repair.pending is not None:
            self.repair_settle(ending=True)
        self.stopped, self.end_kind = reason, kind
        self.log("stop", reason)
        trace.event("stop", reason=reason)
        return reason

    # -------------------------------------------------------------------------------------------- request 1

    def show(self, *texts: str) -> None:
        """`pointer_shown` (lane lib4): a request about to be sent carries a pointed function's card, its typed
        signature opening a line (`library.Card`), wherever the form put it."""
        if self.fired and not self.counts["pointer_shown"]:
            heads = {function.typed for function, _ in self.fired}
            if any(line in heads for text in texts if text for line in str(text).split("\n")):
                self.counts["pointer_shown"] = 1
        self.card_reached(texts, [*(function for function, _ in self.fired), *self.carded_order])   # lane called1

    def opened(self, name: str | None) -> None:
        """`pointer_shown` too where `lib` opened a pointed function's card (its whole docstring)."""
        found = self.library.find(name) if name and self.library is not None else None
        if found is not None:
            self.cardfail_looked.add(found.name)    # lane cardfail1: its card in a `lib` answer
        if found is not None and found.name in {function.name for function, _ in self.fired}:
            self.counts["pointer_shown"] = 1
        self.card_reached(candidates=[found] if found is not None else [])     # lane called1

    def carried(self) -> None:
        """The loop's first request is about to be sent: whether it carries cards (`pointer_request1`, request 1's in
        the loop band), their bytes and the cards cut (`cards_cut`)."""
        if self.band == "loop":
            self.counts["pointer_request1"] = int(bool(self.cards))
            self.counts["cards_request1_bytes"] = len(self.cards.encode("utf-8"))
        if self.cards:
            self.counts["cards_cut"] = len(self.cuts)

    def first(self, task, pathmap, data: str) -> bool:
        """Form `review`'s request 1 under `LOOP_BYTES`: N0's, its system prompt and user turn N0's byte for byte, and
        its first run, the script laid in as work/main.py; True where the unit ends there (the run clean and the review
        not opened, after the library pass where the pointer named functions; or the House never answered).
        Otherwise the conversation holds what came of it."""
        user = n0_prompt(task, pathmap, data, max(0.0, self.left()) * RUN_SHARE)
        self.client.kind = "request1"
        think = self.write_think("write")     # lane thinkwrite1: the unit's first write
        try:
            self.show(N0_SYSTEM)
            reply = self.client.chat(N0_SYSTEM, self.redact(user), **think)
        except model.ModelError as exc:
            self.log("request", f"no reply ({type(exc).__name__}: {logged(str(exc))[:200]})")
            self.stop(f"the House did not answer request 1 ({type(exc).__name__})")
            return True
        if think:
            self.write_thought("write", reply)
        self.counts["turns"] += 1
        self.replied("request1", reply)
        code = runner.extract(reply.text)
        self.wrote_first(reply, code is not None)
        cut = reply.finish_reason == "length" or (code is not None and not code.closed)
        self.log("request", f"{reply.prompt_tokens:,} prompt and {reply.completion_tokens:,} completion tokens, "
                            f"finish {reply.finish_reason}; " + (f"a script of {len(code.text):,} bytes" if code else
                                                                  "no script") + ("; cut" if cut else ""))
        if code is None:
            self.messages.append({"role": "user", "content": NO_SCRIPT})
            return False
        if cut:
            self.counts["cut_replies"] += 1
            whole, best = self.continued(user, code.text)
            if whole is None:
                return self.drafted(best)
            code = runner.Code(whole, True)
        return self.ran_first(code)

    def continued(self, user: str, script: str) -> tuple[str | None, str]:
        """Lane n11e: request 1's script the cap cut, continued. Kept to its last complete top-level statement before
        the definition the cut fell in (`resume`), it is shown under request 1's own user turn and the rest asked for,
        code only and no prose (`CONTINUE_ASK`, under N0's system prompt: a new request after the cut, request 1
        unchanged); the reply's code, every fence line dropped (`continuation_code`),
        joined to it (`runner.stitch`: a restart taken as the script, lines repeating the kept end dropped). A join
        that compiles from a reply not cut is whole: its entry is checked (a `main` defined and never called is said
        at its run, `ENTRY`). A reply cut again is continued while `CONTINUATIONS` and its join grew; else the best
        text so far goes to the draft. Asked only with a request and `ASK_S` left. (the whole script or None, the
        best text)."""
        for _ in range(CONTINUATIONS):
            kept, opened = resume(script)
            if not kept.strip():
                self.log("continue", "not asked: nothing of the script is complete")
                return None, script
            if self.requests_left() < 1 or self.left() < ASK_S:
                self.log("continue", "not asked: " + ("no request left" if self.requests_left() < 1 else
                                                      f"{max(self.left(), 0):.0f} s left"))
                return None, script
            ask = CONTINUE_ASK.format(where=f" inside `{opened}`" if opened else "",
                                      start=f"`{opened}` written again whole, then every line after it" if opened
                                      else "every line after the last one above")
            self.client.kind = "continuation"
            try:
                reply = self.client.chat(N0_SYSTEM, self.redact(
                    f"{user}\n# Your script so far, cut off at the output limit\n\n```python\n{kept}```\n\n{ask}\n"))
            except model.ModelError as exc:
                self.log("continue", f"no reply ({type(exc).__name__}: {logged(str(exc))[:200]})")
                return None, script
            self.counts["turns"] += 1
            self.replied("continuation", reply)
            self.counts["continuations"] += 1
            rest = continuation_code(reply.text)
            joined, how = runner.stitch(kept, f"```python\n{rest}\n```\n") if rest.strip() else (kept, "nothing added")
            whole = bool(rest.strip()) and reply.finish_reason != "length" and runner.compiles(joined)
            self.log("continue", f"{reply.completion_tokens:,} completion tokens, finish {reply.finish_reason}; kept "
                                 f"{len(kept):,} B, the join {len(joined):,} B ({how}), "
                                 + ("whole" if whole else "not whole"))
            trace.event("continue", kept_bytes=len(kept), joined_bytes=len(joined), read=how, whole=whole,
                        finish=reply.finish_reason)
            if whole:
                self.counts["continuation_whole"] = 1
                if not runner.has_driver(joined):
                    name = runner.entry(joined) or "main"
                    self.entry = ENTRY.format(name=name)
                    self.log("entry", f"the stitched script calls none of what it defines (`{name}`)")
                return joined, joined
            if not rest.strip():
                return None, script
            if reply.finish_reason != "length" or len(joined) <= len(kept):
                return None, joined
            script = joined
        return None, script

    def drafted(self, text: str) -> bool:
        """A cut script that no continuation made whole: its complete top-level definitions kept as work/draft.py and
        the rest asked for in as few files as fit (n11c's; lane files1), or the task asked again where none is
        complete; under lane firstout1's `FIRSTOUT=on` the rest asked one output file at a time (`CUT_NOTE_FIRSTOUT`,
        `CUT_EMPTY_FIRSTOUT`)."""
        kept, names, where = draft(text)
        self.log("draft", f"{len(kept.encode()):,} B kept, {len(names)} definition(s); the cut {where}")
        if not kept:
            self.messages.append({"role": "user", "content": CUT_EMPTY_FIRSTOUT if self.firstout_armed else CUT_EMPTY})
            return False
        said = self.ws.write(f"{commands.WORK}/draft.py", kept)
        self.versions[f"{commands.WORK}/draft.py"] = [(self.counts["turns"], sha256(kept))]
        note = (CUT_NOTE_FIRSTOUT if self.firstout_armed else CUT_NOTE).format(where=where, defines=f" (it defines {', '.join(names[:20])})" if names else "")
        self.synthetic([Call("write", {"path": f"{commands.WORK}/draft.py", "text": kept})], [said], note)
        return False

    def ran_first(self, code: runner.Code) -> bool:
        """Request 1's whole script laid in as work/main.py and run (N0's run, its share of the time): True where the
        unit ends there (the run clean, after the library pass where the pointer named functions, and the review not
        opened)."""
        written = self.ws.write(f"{commands.WORK}/main.py", code.text)
        self.versions[f"{commands.WORK}/main.py"] = [(self.counts["turns"], sha256(commands.unfenced(code.text)))]
        if written.startswith("wrote "):
            self.saved(f"{commands.WORK}/main.py")
        seconds = max(MIN_RUN_S, self.left() * RUN_SHARE) if self.requests_left() >= 1 else self.left()
        if seconds < MIN_RUN_S or self.left() < MIN_RUN_S:
            self.log("run", f"not started: {max(self.left(), 0):.0f} s left")
            self.stop("no time for request 1's run")
            return True
        said, ran = self.ws.run(f"{commands.WORK}/main.py", seconds, self.seen, lambda done: self.tail(done, True))
        self.ran(ran, seconds)
        self.saw(said, ran)
        said = self.carded_after(said, ran)
        if ran is not None and ran.clean and ran.findings:
            self.counts["sanity_opened"] = 1     # a clean run the sanity checks fault opens the loop (lane n11d)
            self.log("sanity", f"request 1's clean run: {len(ran.findings)} finding(s); the loop opens")
        elif ran is not None and ran.clean and not self.gate_said:
            self.counts["finish_clean"] = 1
            self.standing = ran
            if self.fired:
                self.library_pass(code.text, ran)
            if self.open_review():
                return False
            self.stop("request 1's script ran with the checks passing")
            return True
        self.synthetic([Call("write", {"path": f"{commands.WORK}/main.py", "text": code.text}),
                        Call("run", {"path": f"{commands.WORK}/main.py"})], [written, said])
        return False

    def synthetic(self, calls: list[Call], results: list[str], note: str = "") -> None:
        """Request 1's script as if the House had written (and run) it through the commands."""
        for number, call in enumerate(calls, 1):
            call.id = f"call_0_{number}"
            call.raw = json.dumps(call.arguments)
        if self.mode == "tools":
            self.record("", calls, results, True)
            if note:
                self.messages.append({"role": "user", "content": note})
            return
        blocks = "\n\n".join(f"```python file={call.arguments['path']}\n{call.arguments['text'].rstrip()}\n```"
                             for call in calls if call.name == "write")
        self.messages.append({"role": "assistant", "content": blocks})
        self.messages.append({"role": "user", "content": "\n\n".join(
            [f"[{call.describe()}]\n{result}" for call, result in zip(calls, results)] + ([note] if note else []))})

    # ------------------------------------------------------------------------------------------ the library pass

    def library_pass(self, script: str, first_run: commands.Ran) -> None:
        """One request after a clean first run (above): the pointed functions' cards and the first script, the same
        script asked back with the pointed quantities computed by qf. Its script runs as work/libpass.py, the first
        run's output set aside, and is adopted only where `refused` finds nothing; else the first run's output is put
        back. Not asked without a request or `ASK_S` left; never asked again."""
        if not self.cards:
            return
        if self.requests_left() < 1 or self.left() < ASK_S:
            self.log("libpass", "not asked: " + ("no request left" if self.requests_left() < 1 else
                                                 f"{max(self.left(), 0):.0f} s left"))
            return
        user = self.redact(LIBPASS_ASK.format(cards=self.cards, script=script.rstrip("\n")))
        self.counts["libpass_units"] = 1
        self.counts["cards_cut"] = len(self.cuts)
        self.counts["libpass_request_bytes"] = len(LIBPASS_SYSTEM.encode()) + len(user.encode())
        self.log("libpass", f"asked: {self.counts['libpass_request_bytes']:,} B, the cards of "
                            + ", ".join(function.call for function, _ in self.fired))
        self.client.kind = "libpass"
        try:
            self.show(user)
            reply = self.client.chat(LIBPASS_SYSTEM, user)
        except model.ModelError as exc:
            self.log("libpass", f"no reply ({type(exc).__name__}: {logged(str(exc))[:200]})")
            return self.rejected("no reply")
        self.counts["turns"] += 1
        self.replied("libpass", reply)
        code = runner.extract(reply.text)
        self.log("libpass", f"{reply.prompt_tokens:,} prompt and {reply.completion_tokens:,} completion tokens, finish "
                            f"{reply.finish_reason}; " + (f"a script of {len(code.text):,} bytes" if code else
                                                          "no script"))
        if code is None:
            return self.rejected("its reply held no script")
        if reply.finish_reason == "length" or not code.closed:
            return self.rejected("its reply was cut at the output limit")
        seconds = min(self.left(), max(RUN_S, 2 * first_run.seconds))
        if seconds < MIN_RUN_S:
            return self.rejected(f"{max(self.left(), 0):.0f} s left for its run")
        wanted = sorted(contract.snapshot(self.out))
        self.ws.write(f"{commands.WORK}/{LIBPASS_FILE}", code.text)
        with guard.WRITING:
            self.set_aside()
        _, ran = self.ws.run(f"{commands.WORK}/{LIBPASS_FILE}", seconds, self.seen)
        self.ran(ran, seconds)
        why = refused(ran, wanted, sorted(contract.snapshot(self.out)))
        with guard.WRITING:
            self.settle(adopt=not why)
        here = self.ws.dir / LIBPASS_FILE
        if why:
            here.replace(self.work / f"rejected-{LIBPASS_FILE}")     # out of work/: the files at the end are main.py
            return self.rejected(why)
        here.replace(self.ws.dir / "main.py")
        adopted, turn = sha256((self.ws.dir / "main.py").read_text(encoding="utf-8", errors="replace")), \
            self.counts["turns"]
        self.versions[f"{commands.WORK}/main.py"] = [(turn, adopted)]       # main.py is the pass's now, and so is
        self.last_ran[f"{commands.WORK}/main.py"] = (turn, adopted, True)    # its clean run (form `review` reads on)
        self.standing = ran
        self.counts["libpass_adopted"] = 1
        self.log("libpass", f"adopted: exit 0, the checks pass, the same {len(wanted)} output file(s)")
        trace.event("libpass", adopted=True)

    def rejected(self, why: str) -> None:
        self.counts["libpass_rejected"] = 1
        self.log("libpass", f"rejected: {why}; the first run's output stands")
        trace.event("libpass", adopted=False, why=why)

    def set_aside(self) -> None:
        """The first run's output moved beside work/ (`SET_ASIDE`), the output directory left empty for the pass; the
        caller holds guard.WRITING."""
        aside = self.work / SET_ASIDE
        shutil.rmtree(aside, ignore_errors=True)
        aside.mkdir(parents=True)
        for entry in sorted(self.out.iterdir()):
            shutil.move(str(entry), str(aside / entry.name))
        self.aside = aside

    def settle(self, adopt: bool = False, step: str = "libpass") -> None:
        """The library pass decided, the caller holding guard.WRITING: adopted, the first run's output set aside is
        dropped; else, and wherever the pass did not end (the watchdog's close, an error), the pass's output is
        cleared and the first run's put back. Nothing where no output is set aside; a failure is logged (under
        `step`: the review's guard settles here too), never raised, so the close that follows always runs."""
        aside, self.aside = self.aside, None
        if aside is None:
            return
        try:
            if not adopt:
                clear(self.out)
                for entry in sorted(aside.iterdir()):
                    shutil.move(str(entry), str(self.out / entry.name))
            shutil.rmtree(aside, ignore_errors=True)
        except Exception as exc:  # noqa: BLE001 - the close follows whatever failed here
            self.log(step, f"the first run's output not put back whole ({type(exc).__name__})")

    # ------------------------------------------------------------------------------------------------ the replies

    def replied(self, kind: str, reply) -> None:
        """A reply counted for the summary's distribution: its kind, completion tokens, and whether the cap cut it."""
        self.replies.append((kind, int(reply.completion_tokens or 0), reply.finish_reason == "length"))

    def reply_keys(self) -> dict:
        """The replies' completion tokens (lane cards1's second addendum): the median, the 90th percentile (nearest
        rank) and the replies cut at the cap, over every reply and over each kind's (`REPLY_KINDS`; `-` where none)."""
        keys = {}
        for kind in ("", *REPLY_KINDS):
            tokens = sorted(n for k, n, _ in self.replies if not kind or k == kind)
            suffix = f"_{kind}" if kind else ""
            keys[f"reply_tokens_median{suffix}"] = round(statistics.median(tokens)) if tokens else "-"
            keys[f"reply_tokens_p90{suffix}"] = tokens[max(0, -(-9 * len(tokens) // 10) - 1)] if tokens else "-"
            keys[f"replies_at_cap{suffix}"] = sum(cut for k, _, cut in self.replies if not kind or k == kind)
        return keys

    # ------------------------------------------------------------------------------------------------ the review

    def open_review(self) -> bool:
        """Form `review` (lane loop1): where request 1's run, after the library pass (adopted or not), would end the
        unit clean, the loop opens instead on the review observation (`review_text`), as if the House had written and
        run work/main.py there (`review_room`, `review_begin`). True where it opened."""
        if not self.review_room():
            return False
        main = f"{commands.WORK}/main.py"
        script = (self.ws.dir / "main.py").read_text(encoding="utf-8", errors="replace")
        said = self.review_begin(self.standing)
        self.synthetic([Call("write", {"path": main, "text": script}), Call("run", {"path": main})],
                       [self.ws.write(main, script), said])
        self.review_said(said)
        return True

    def review_at_finish(self) -> str:
        """Lane review2, form `review` over the loop (`review_loop`): the loop's first finish that would end the unit
        clean (the checks passing, no finding) opens the review instead, its observation the finish's answer and
        work/main.py's last run the one it reads; the finish refusals start again for the review's own finish. The
        observation, "" where the review is not opened (not this form, opened already, or no room: the unit then ends
        on the finish)."""
        if not self.review_loop or self.review_from is not None or not self.review_room():
            return ""
        said = self.review_begin(self.last_main)
        self.finishes, self.sanity_refused = 0, False
        self.review_said(said)
        return said

    def review_room(self) -> bool:
        """Whether the review opens: not with fewer than `REVIEW_FLOOR` requests or `REVIEW_FLOOR` x `ASK_S` left (two
        requests at the platform's pace, 44 s, fit in it), counted `review_skipped`; nor where the output cannot be
        copied beside work/ (`SET_ASIDE`) for the guard at the unit's end (`review_end`), the output left as it
        stands. Where it does not, the unit ends on the clean run or finish that was due to end it."""
        requests = self.client.cap - self.client.usage.requests
        if requests < REVIEW_FLOOR or self.left() < REVIEW_FLOOR * ASK_S:
            self.counts["review_skipped"] = 1
            self.log("review", "not opened: " + (f"{requests} request(s) left" if requests < REVIEW_FLOOR else
                                                 f"{max(self.left(), 0):.0f} s left"))
            return False
        try:
            with guard.WRITING:
                self.keep_aside()
        except Exception as exc:  # noqa: BLE001 - the unit ends clean as it stands
            shutil.rmtree(self.work / SET_ASIDE, ignore_errors=True)
            self.log("review", f"not opened: the first output could not be set aside ({type(exc).__name__})")
            return False
        return True

    def review_begin(self, ran: commands.Ran | None) -> str:
        """The review opened: the first output's snapshot and hashes, the requests counted from here, and its
        observation over the run `ran` (`review_text`, the budget the review's)."""
        self.review_first, self.review_hashes = contract.snapshot(self.out), output_hashes(self.out)
        self.review_from = self.client.usage.requests
        self.counts["review_opened"] = 1
        asks = REVIEW_LOOP_ASK_DEFS if self.write_defs and self.review_loop else REVIEW_ASK_DEFS if self.write_defs \
            else REVIEW_LOOP_ASK if self.review_loop else REVIEW_ASK
        return review_text(self.ws, self.stated, ran, self.mode, self.budget(),
                           asks[self.mode])

    def review_said(self, said: str) -> None:
        """The review's opening logged and traced: what its observation read of the properties, and its bytes."""
        read = {state: said.count(f"\n- {state}: ") for state in ("held", "unchecked", "broken")}
        self.log("review", f"opened: {len(self.review_first)} output file(s) set aside; "
                           + ", ".join(f"{n} {state}" for state, n in read.items()) + f" of the properties; "
                           f"{len(said.encode()):,} B")
        trace.event("review", opened=True, **read)

    def keep_aside(self) -> None:
        """The first output copied beside work/ (`SET_ASIDE`), the output left as it stands for the review; the
        caller holds guard.WRITING."""
        aside = self.work / SET_ASIDE
        shutil.rmtree(aside, ignore_errors=True)
        shutil.copytree(self.out, aside, symlinks=True)
        self.aside = aside

    def review_holds(self) -> tuple[bool, str]:
        """The guard: whether the output the review leaves stands, and why. It stands where the review's last run
        that wrote to the output exited 0, passed the checks, met no finding and wrote exactly the first run's set of
        files, and nothing changed the output after it; where no run of the review wrote to it, where the output is
        still the first. A run writing nothing there (a check script's) is not read: the output is the one before it."""
        now = contract.snapshot(self.out)
        first = frozenset(self.review_first)
        if self.review_last is None:
            return (True, "no run of the review wrote the output") if now == self.review_first else \
                (False, "the output changed and no writing run of the review ended")
        clean, wrote, after = self.review_last
        if not clean:
            return False, "its last writing run was not clean"
        if wrote != first:
            return False, (f"its last writing run wrote {len(wrote & first)} of the first run's {len(first)} "
                           f"file(s) and {len(wrote - first)} other(s)")
        if now != after:
            return False, "the output changed after its last writing run"
        return True, f"its last writing run is clean and wrote the same {len(first)} file(s)"

    def review_end(self) -> None:
        """The review's guard at the unit's end (the loop's, the watchdog's or an error's), the caller holding
        guard.WRITING: the output the review leaves stands where `review_holds`, else the first output is put back
        whole (`review_kept`); counted, `review_requests` and `review_changed` (the output at the end differs from
        the first's). Never raises, so the close that follows always runs."""
        try:
            holds, why = self.review_holds()
        except Exception as exc:  # noqa: BLE001 - the first output is then put back
            holds, why = False, f"the guard failed ({type(exc).__name__})"
        self.settle(adopt=holds, step="review")
        self.counts["review_requests"] = self.client.usage.requests - (self.review_from or 0)
        self.counts["review_kept"] = int(not holds)
        self.counts["review_changed"] = int(output_hashes(self.out) != self.review_hashes)
        self.log("review", ("the review's output stands: " if holds else "the first output put back: ") + why
                 + f"; {self.counts['review_requests']} request(s)"
                 + ("; the output differs from the first" if self.counts["review_changed"] else ""))
        trace.event("review", kept=not holds, changed=bool(self.counts["review_changed"]))

    def reviewed(self) -> None:
        """The review's guard where it has not run yet, taking guard.WRITING: at the unit's end (the loop's or an
        error's), before the summary that counts it. Nothing outside the review."""
        if self.review_from is None:
            return
        with guard.WRITING:
            if self.aside is not None:
                self.review_end()

    def release(self) -> None:
        """At the watchdog's close, the caller holding guard.WRITING: lane drafts1's round (open, the floor so far laid;
        done, its held floor as at the close); the review's guard where it is open, else what the library pass set aside put back (`settle`);
        then lane rules1's put-back, the clock having ended the unit."""
        self.repair.discard(self.client)
        if self.drafting is not None:
            self.drafting.finish()
        if self.review_from is not None and self.aside is not None:
            self.review_end()
        else:
            self.settle()
        self.put_back("the watchdog's close")
        self.repeat_end("the watchdog's close")      # lane repeat1: the one-shot's output, where the loop's fails

    # ------------------------------------------------------------------------------------------------ form state

    def state_loop(self, task, pathmap) -> str:
        """Form `state` (lane state1; agent/n11/state.py): every request one system prompt and one user turn rebuilt
        from the harness's state (`state_text`), never a transcript, thinking off; each reply read in the state form
        and acted on (`state_turn`). Only `finish` ends the unit by the House's choice; the stops are the loop's (no
        request or `ASK_S` left) and the form's (`NOTHING_CAP` replies in a row reading nothing; an identical file
        resent a `STUCK_ENDS`-th time)."""
        self.mode = "fenced"            # the reply form is the fenced one's: blocks and directives, read as text
        self.state_system = self.state_prompt(self.selected(task, self.library, self.index, self.state_prompt))
        self.state_inputs = state.inputs(task.data_dir, task.instruction, task.canaries) \
            if self.inputs_setting != "rich" else self.described()
        self.log("state", f"the input files described in {len(self.state_inputs.encode()):,} B; the system prompt "
                          f"{len(self.state_system.encode()):,} B; the LAST RUN section {self.result_setting} "
                          f"({self.result_source})")
        while True:
            if self.requests_left() < 1:
                return self.stop(self.spent())
            if self.left() < ASK_S:
                return self.stop(f"{max(self.left(), 0):.0f} s left, under a request's {ASK_S:.0f} s")
            last = self.last_call_due()
            if last:
                self.counts["last_call_asked"] = 1
                timed = self.last_asked()
                self.log("last call", f"asked: {self.requests_left()} request(s) and {max(self.left(), 0):.0f} s left; "
                                      "no output file the task names" + timed)
            ask = self.state_sketch_ask() if self.sketch_on and not last and not self.state_bytes else ""
            user = self.state_text(task, pathmap, state.LAST_CALL if last else ask)
            self.client.kind = "sketch" if self.sketch_due and self.sketch_mode == "turn" else "loop"
            try:
                self.show(user)
                reply = self.client.chat(self.state_system, user, **self.sketch_think())
            except model.ModelError as exc:
                self.log("request", f"no reply ({type(exc).__name__}: {logged(str(exc))[:200]})")
                if last:
                    self.last_sent(None)
                return self.stop(f"the House did not answer ({type(exc).__name__})")
            self.counts["turns"] += 1
            self.replied("loop", reply)
            if last:
                self.last_sent(reply)
                return self.stop(self.state_last_call(self.counts["turns"], reply))
            if self.sketch_due:
                self.sketch_due = False
                if self.state_sketched(self.counts["turns"], reply):
                    self.replies[-1] = ("sketch", *self.replies[-1][1:])
                    continue
            ended = self.state_turn(self.counts["turns"], reply)
            self.state_notes[1].extend(state.note(line, refusal=True) for line in self.sketch_lines)
            self.sketch_lines = []
            if ended:
                return self.stop(ended)

    def state_prompt(self, index: str) -> str:
        """Form `state`'s system prompt (agent/n11/state.py `system`) with `index` in the index's place: the library's
        index, or the selection's shelves' section (lane state4)."""
        return state.system(index, api_card(), "\n".join(CONVENTION_LINES), LIBRARIES, RUN_S)

    def state_sketch_ask(self) -> str:
        """Form `state`'s sketch (lane plan4's addendum 3): the first state's closing ask (`state.SKETCH_ASK`, in the
        state's reply form's words: prose, no block, no directive), naming the cards where the state carries any,
        else the index. Never where the last call is due (the caller's test), which at the first state, no output yet,
        is wherever `sketch_short`'s floor would bite: one request left, or under `LAST_CALL_S` (2 x `ASK_S`)."""
        source = SKETCH_SOURCES[0] if self.state_cards() else SKETCH_SOURCES[1]
        ask = state.SKETCH_ASK[self.sketch_mode].format(source=source)
        self.counts["sketch_asked"], self.sketch_due, self.sketch_thought = 1, True, self.sketch_thinking
        self.log("sketch", f"asked ({self.sketch_mode}, {self.sketch_source}): the first state closes on its ask, "
                           f"{len(ask.encode('utf-8')):,} B, naming {source}; thinking {self.sketch_thinking} on its "
                           "request")
        return ask

    def state_sketched(self, turn: int, reply) -> bool:
        """Form `state`'s first reply after its ask (addendum 3), read and never acted on but for its names: True
        where it is `turn`'s sketch, a reply `state.read` reads nothing in, whose turn ends there (its ledger line, and
        the notes `state.SKETCH_NEXT` and its picks' lines); any other reply goes on to `state_turn` (`sketch_skipped`
        under `turn`), its text before the first block the sketch, its picks' lines after that turn's notes. The
        sketch, cut at its end within `state.SKETCH_TOKENS` (said in the ledger), is every later state's SKETCH
        section; its picks the pointer's cards lack join CARDS after them (`state_cards`), and a pick left out of it
        and each `qf.` name the index lacks are said once, in LAST RUN's notes."""
        sketch = self.sketch_mode == "turn" and state.read(reply.text, reply.finish_reason).nothing
        text = ((reply.text or "") if sketch else (reply.text or "").split("```", 1)[0]).strip()
        picks, unknown = self.sketch_read(reply, text, sketch)
        cut = False
        if text:
            self.state_sketch, cut = state.sketch(text)
        held = {function.name for function in self.state_base()}
        self.state_picks = [function for function in picks if function.name not in held]
        self.state_cards()
        left_out = {cut_.split(":", 1)[0] for cut_ in self.state_cuts if ": left out" in cut_}
        shown = [function for function in self.state_picks
                 if function.name in self.state_carded[0] and function.call not in left_out]
        left = [function for function in self.state_picks if function not in shown]
        self.counts["sketch_cards_added"] = len(shown)
        lines = ([SKETCH_LEFT.format(named=", ".join(f"`{function.call}`" for function in left))] if left else []) \
            + self.unknown_lines(unknown)
        kept = (f"the sketch kept as {state.SKETCH}" + (", cut at its end" if cut else "")) if text else ""
        self.sketch_said(reply, text, sketch, picks, unknown, "the sketch, kept as the SKETCH section: " if sketch
                         else "a block or a directive came first: the reply runs as the state's; "
                         if self.sketch_mode == "turn" else "the text before the block: ")
        if cut:
            self.log("sketch", "the SKETCH section cut at its end, within "
                               f"{measure.bound(state.SKETCH_TOKENS, state.SKETCH_BYTES):,} "
                               + ("tokens" if measure.UNIT.tokenizer is not None else "B"))
        if not sketch:
            self.sketch_did, self.sketch_lines = ([kept] if kept else []), lines
            return False
        self.state_notes = (turn, [state.note(line, refusal=True) for line in (state.SKETCH_NEXT, *lines)])
        self.gave[turn] = "no run"
        self.ledger.append(state.ledger_line(turn, [kept or "the sketch: no text"]))
        trace.event("turn", turn=turn, calls=[], ended=False)
        return True

    def state_text(self, task, pathmap, ask: str = "") -> str:
        """The user turn (agent/n11/state.py `render`), rebuilt from the harness's state, the canaries marked: the
        task, the paths, the input files described, the sketch where one was kept, the cards, the House's files under
        work/ as they stand, the last run's result with the harness's answers to the last reply, the ledger and the
        budget, `ask` after it (lane clock1's last call). Its bytes, its files section's, its LAST RUN section's and
        the ledger's lines are counted."""
        sections = [("TASK", task.instruction.strip()), ("PATHS", pathmap.text()), ("INPUT FILES", self.state_inputs)]
        if self.state_sketch:           # lane plan4's addendum 3: the House's sketch, after the input files
            sections.append((state.SKETCH, self.state_sketch))
        cards = self.state_cards()
        if cards:
            sections.append(("CARDS", cards))
        sections += [(state.FILES, state.files(self.ws.root, self.state_order())),
                     ("LAST RUN", self.state_result()), ("LEDGER", "\n".join(self.ledger) or "none yet"),
                     ("BUDGET", f"{self.budget()}\n\n{ask}" if ask else self.budget())]
        text, sizes = self.redact(state.render(sections)), state.sized(sections)
        if not self.state_bytes:                # the loop's first request: the cards it carries
            self.counts["pointer_request1"] = int(bool(cards))
            self.counts["cards_request1_bytes"] = len(cards.encode("utf-8"))
        self.state_bytes.append(len(text.encode("utf-8")))
        self.files_bytes.append(sizes[state.FILES])
        self.last_run_bytes.append(sizes["LAST RUN"])
        self.ledger_sent = len(self.ledger)
        self.log("state", f"{len(text.encode('utf-8')):,} B: " + ", ".join(f"{head.lower()} {n:,}"
                                                                           for head, n in sizes.items()))
        trace.event("state", bytes=len(text.encode("utf-8")), sections=sizes)
        return text

    def state_order(self) -> list[str]:
        """The House's files the state shows, in the order first saved, work/main.py last."""
        main = f"{commands.WORK}/main.py"
        return [named for named in self.state_shown if named != main] + ([main] if main in self.state_shown else [])

    def state_cards(self) -> str:
        """The cards the state carries: those of the functions `lib` opened (the latest first), then the base's
        (`state_base`), then the sketch's picks, at most `library.POINTERS` within `library.card_bound()`
        (`Library.carded`, the loop's trim); what was cut logged once a selection (`cards_cut`). Where the selection
        picked (lane state4) the loop's own composition under the call: the base whole and `library.POINTERS` more,
        under `selection.CARDS_HEAD` within `selection.card_bound()`, so before any `lib` or sketch the section is the
        loop's cards byte for byte."""
        shaped = self.selection.result
        chosen = list({function.name: function for function in [*self.state_opened, *self.state_base(),
                                                                 *self.state_picks]}.values())
        chosen = chosen[:library.POINTERS + (len(shaped.order) if shaped is not None else 0)]
        names = tuple(function.name for function in chosen)
        if names != self.state_carded[0]:
            ranked = [(function, function.name) for function in chosen]
            text, cuts = ("", []) if not chosen else library.Library.carded(ranked) if shaped is None else \
                library.Library.carded(ranked, selection.card_bound(), head=selection.CARDS_HEAD)
            self.state_carded, self.state_cuts = (names, text), cuts
            self.counts["cards_cut"] = len(cuts)
            for cut in cuts:
                self.log("cards", cut)
        return self.state_carded[1]

    def state_base(self) -> list[library.Function]:
        """The functions whose cards the state carries before any `lib` or sketch: the pointer's; where the selection
        picked (lane state4), the pointer's and then the picks' but the I/O helpers, as the loop's cards order them
        (`selection.Shaped.order`)."""
        shaped = self.selection.result
        return list(shaped.order) if shaped is not None else [function for function, _ in self.fired]

    def state_result(self) -> str:
        """The LAST RUN section (agent/n11/state.py `last_run`) in this process's setting (`STATE_RESULT`, lane
        state2): the last turn's runs, the turn they ran at, then the harness's answers to the last reply. `full`, the
        default, is lane state1's: each run as `Workspace.run` said it (the command and its exit, the streams' ends,
        the traceback's frames in work/, the output's files and digest, the findings, the checks, the gate's line)
        and each answer whole (its saves and refusals, why nothing ran, `lib`'s and `finish`'s answers),
        within `NOTES_TOKENS`; `lean` and `minimal` as state.py says them, beside the cards this request carries
        (`state_carded`, built just before)."""
        return state.last_run(self.state_last, self.state_notes, self.result_setting, self.state_carded[1])

    def state_turn(self, turn: int, reply) -> str:
        """One reply in the state form (`state.read`): its blocks saved in order (`state_write`; a cut one's complete
        definitions kept as a draft, `state_salvage`), the runs they make (`state_run`), its `lib` lines answered
        (`LIB_SHOWN` at most) and its `finish` decided last, over the output as it then stands (the loop's rules:
        `finish`); a reply reading nothing told so once, the `NOTHING_CAP`-th in a row ending the unit. The ledger's
        line and the harness's answers recorded. The reason the unit ends, "" where it goes on."""
        read = state.read(reply.text, reply.finish_reason)
        self.wrote_first(reply, bool(read.blocks))
        self.log("request", f"turn {turn}: {reply.prompt_tokens:,} prompt and {reply.completion_tokens:,} completion "
                            f"tokens, finish {reply.finish_reason}; " + (read.describe() or "nothing read"))
        notes: list[state.Note] = []
        did, self.sketch_did = list(self.sketch_did), []     # `inline`'s sketch kept (addendum 3), else none
        self.state_notes = (turn, notes)
        self.gave[turn] = "no run"
        if read.nothing:
            self.counts["no_call"] += 1
            self.no_call_streak += 1
            capped = self.no_call_streak >= state.NOTHING_CAP
            forced = capped and self.can_force()
            onward = forced or (capped and self.no_call_streak == state.NOTHING_CAP and self.last_call_due())
            trace.event("turn", turn=turn, calls=[], ended=capped and not onward)
            self.ledger.append(state.ledger_line(turn, [*did, *(state.UNSAVED_PART[why].format(name=state.bare(path))
                                                               for path, why, _ in read.unsaved),
                                                        "no file and no directive"
                                                        + (", cut at the cap" if read.cut else "")
                                                        + ("; asked once more for a file" if forced else "")]))
            if capped and not onward:
                self.counts["no_call_capped"] = 1
                return (f"{self.no_call_streak} replies in a row with no file and no directive"
                        + (", the forced ask among them" if self.no_call_streak > state.NOTHING_CAP else "")
                        + ": the unit ends with the output as it stands")
            if forced:
                self.counts["prose_forced"] = 1
                self.log("request", "the forced ask: a file block asked while two replies remain")
            self.state_unsaved(read, notes, [])
            notes.append(state.note((state.FORCED if forced else state.NOTHING) + (state.UNREAD if read.unread else ""),
                                    refusal=True))
            return ""
        self.no_call_streak = 0
        self.counts["cut_replies"] += read.cut
        self.cut_streak = self.cut_streak + 1 if read.cut else 0
        self.counts["calls"] += len(read.blocks) + len(read.libs) + (read.finish is not None)
        saved: list[tuple[str, bool]] = []
        ended = ""
        for block in read.blocks:
            if block.cut:
                said, part = self.state_salvage(block)
            else:
                said, part, kept, ended = self.state_write(block, turn)
                saved += [kept] if kept else []
            notes += [said] if said else []
            did.append(part)
            if ended:
                break
        if not ended:
            self.state_unsaved(read, notes, did)
            did.append(self.state_run(saved, turn, notes))
            ended = self.ending
        if not ended:
            for name in read.libs[:state.LIB_SHOWN]:
                notes.append(self.state_lib(name))
                did.append(f"lib {name[:40]}")
            if read.finish is not None:
                said, done = self.finish(False, read.finish)
                head = first_line(said)
                notes.append(state.note(said, head[:-1] + "; " + FINISH_AGAIN if head.endswith("check(s) fail:")
                                        else head, refusal=True))
                did.append(head.split(";")[0].rstrip(":")[:70])
                ended = self.finish_said if done else ""
        if not ended and read.finish is None and self.state_last is not None and self.state_last[0] == turn \
                and self.state_done():
            self.counts["finish_asked"] += 1
            notes.append(state.note(state.CLEAN.format(named=" and ".join(report.ran.path for report in
                                                                          self.state_last[1])), refusal=True))
        self.ledger.append(state.ledger_line(turn, did))
        trace.event("turn", turn=turn, calls=[part for part in read.describe().split(", ") if part],
                    ended=bool(ended))
        return ended

    def state_unsaved(self, read: state.Reply, notes: list[state.Note], did: list[str]) -> None:
        """Lane fence1: the reply's blocks never saved (`state.read`'s `unsaved`: no code, a path line alone, directive
        lines alone), each said as a refusal (`state.UNSAVED`), and in the ledger (`did`, where given) with the files
        a later block of the same reply replaced (`twice`); all counted."""
        self.counts["blocks_path_line"] += sum(block.line for block in read.blocks)
        self.counts["blocks_unsaved"] += len(read.unsaved)
        self.counts["blocks_twice"] += len(read.twice)
        self.named_counted([block.path for block in read.blocks if block.named and not block.cut])
        for path, why, what in read.unsaved:
            notes.append(state.note(state.UNSAVED[why].format(named=path[:120], what=what[:80]), refusal=True))
            did.append(state.UNSAVED_PART[why].format(name=state.bare(path)[:40]))
        for path, same in read.twice:
            did.append(f"{state.bare(path)[:40]} twice" + (", the same text: saved once" if same else
                                                          ": the later block saved"))

    def state_done(self) -> bool:
        """Lane fence1: whether the state's last runs were clean (exit 0, the checks passing, no finding; the checks
        hold every output file the task names present), so that the next state asks for `finish` (`state.CLEAN`, and
        the resends' answers `IDENTICAL_CLEAN` and `STUCK_CLEAN`)."""
        return self.state_last is not None and all(report.ran.clean and not report.ran.findings
                                                   for report in self.state_last[1])

    def state_write(self, block: state.Block, turn: int) -> tuple[state.Note | None, str, tuple[str, bool] | None, str]:
        """One block saved as `write` saves a file (agent/n11/commands.py: under work/ alone, no standard-library
        module's name, a .py file compiled) and hashed as the loop hashes a write: a text the file already holds
        refused (`IDENTICAL`), the same text resent a second time refused naming both turns and asking another
        method, not the file (`STUCK`; `identical_refused` counts both), a third time ending the unit on the last
        clean run's output (`state_stuck`); a return to an earlier version counted and on the file's
        `CYCLE_RECURRENCES`-th, among at most three versions, the cycle named. (what the harness says, the ledger's
        part, the file saved and whether it compiles or None, the reason the unit ends or "")."""
        here, named = self.ws.local(block.path)
        if here is None:
            return (state.note(state.OUTSIDE.format(why=named), refusal=True),
                    f"{block.path[:40]} refused: not under {commands.WORK}/", None, "")
        name = state.bare(named)
        clash = commands.shadows(named)
        if clash:
            return (state.note(self.ws.write(named, block.text), refusal=True), f"{name} refused: it shadows `{clash}`",
                    None, "")
        text = commands.unfenced(block.text)
        sha = sha256(text)
        history = self.versions.setdefault(named, [])
        if here.is_file() and sha256(here.read_text(encoding="utf-8", errors="replace")) == sha:
            self.counts["write_repeats"] += 1
            turns = self.state_resent.setdefault((named, sha), [])
            turns.append(turn)
            if len(turns) >= state.STUCK_ENDS:
                return None, f"{name} resent identical a third time: the unit ends", None, self.state_stuck(named, turns)
            self.counts["identical_refused"] += 1
            self.log("write", f"{named}: resent identical" + (" a second time" if len(turns) > 1 else ""))
            if len(turns) == 2:
                clean = self.state_done()       # the ledger line agrees with the answer (lane fence1's review)
                said = state.STUCK_CLEAN if clean else state.STUCK
                return (state.note(said.format(named=named, first=turns[0], second=turns[1]), refusal=True),
                        f"{name} resent identical again (turns {turns[0]}, {turns[1]}): refused; "
                        + ("the last run was clean" if clean else "change the method, not the file"), None, "")
            since = history[-1][0] if history and history[-1][1] == sha else None
            said = state.IDENTICAL_CLEAN if self.state_done() else state.IDENTICAL
            return (state.note(said.format(named=named, since=f" (saved at turn {since}, which gave: "
                                                                          f"{self.gave.get(since, 'no run')})"
                                                      if since else ""), refusal=True),
                    f"{name} resent identical: refused", None, "")
        earlier = next((seen_at for seen_at, seen in history if seen == sha), None)
        note = ""
        if earlier is not None:
            self.recurrences[named] += 1
            versions = {seen for _, seen in history[-(CYCLE_WINDOW - 1):]} | {sha}
            if self.recurrences[named] == CYCLE_RECURRENCES and len(versions) <= 3:
                note = CYCLE.format(named=named, k=len(versions), turn=earlier, gave=self.gave.get(earlier, "no run"))
                self.log("write", f"{named}: a cycle among {len(versions)} versions")
        said = self.ws.write(named, text)
        self.log("write", first_line(said)[:200])
        if not said.startswith("wrote "):
            return state.note(said, refusal=True), f"{name} refused", None, ""
        history.append((turn, sha))
        self.state_shown[named] = None
        self.saved(named)
        compiles = "does not compile" not in first_line(said)
        return (state.saved(said, note),
                f"wrote {name}" + ("" if compiles else " (it does not compile)") + ("; a cycle" if note else ""),
                (named, compiles), "")

    def state_salvage(self, block: state.Block) -> tuple[state.Note, str]:
        """A block the output cap cut: its complete definitions kept as `work/<name>.draft` (`draft`, as the loop's
        `salvage` keeps them), which the files section then shows; the `SHORT_STREAK`-th cut reply in a row asks for
        one small file. (what the harness says, the ledger's
        part)."""
        here, named = self.ws.local(block.path)
        streak = self.cut_streak if self.cut_streak >= SHORT_STREAK else 0
        short = state.CUT_SHORT.format(n=streak) if streak else ""
        if here is None:
            said = state.OUTSIDE.format(why=named)
            return (state.note(said + short, said + (state.CUT_STREAK.format(n=streak) if streak else ""),
                               refusal=True), "cut at the cap, outside work/")
        if named.endswith(".py"):
            kept, names, where = draft(block.text)
        else:
            kept, names, where = "".join(block.text.splitlines(keepends=True)[:-1]), [], "at its last line"
        path = f"{commands.WORK}/{Path(named).stem}.draft"
        if kept:
            self.ws.write(path, kept)
            self.state_shown[path] = None
            self.counts["drafts"] += 1
            self.log("draft", f"{named}: {len(kept.encode()):,} B kept as {path}, {len(names)} definition(s); the "
                              f"cut {where}")
        clause = CUT_KEPT.format(draft=path, defines=f" (it defines {', '.join(names[:12])})" if names else "",
                                 where=where) if kept else CUT_NONE
        return (state.cut(CUT_DRAFT.format(named=named, kept=clause) + short, named, clause, streak),
                f"cut at the cap in {state.bare(named)}: " + (f"kept as {state.bare(path)}" if kept else "nothing kept"))

    def state_run(self, saved: list[tuple[str, bool]], turn: int, notes: list[state.Note]) -> str:
        """The runs a reply's saves make, in this order: work/main.py where the reply saved it or a module work/main.py
        imports (directly or through a module under work/; `module_runs`), then each other saved file with an
        `if __name__ == "__main__":` block, itself, in the order saved, each file read as the reply last saved it.
        Nothing runs where the reply saved no .py file or a saved one does not compile, work/main.py not while it
        imports a module that is nowhere, none under a run's `MIN_RUN_S`; each said. Their results are the state's last
        runs (`state_last`: each as `run` said it, the run, and the lines its observation closed with), a failing one's
        frame and error remembered (the same traceback as an earlier turn's, said in the ledger), a clean one's output
        kept (`state_keep`). The ledger's part, "" where the reply saved nothing (its refusals say so)."""
        mainpath, main = f"{commands.WORK}/main.py", self.ws.dir / "main.py"
        latest = dict(saved)            # a file saved twice in the reply: as it last saved it, in the order first saved
        python = [named for named in latest if named.endswith(".py")]
        if not python:
            if saved:
                notes.append(state.note("(no run: no .py file was saved)"))
            return "nothing ran" if saved else ""
        broken = [named for named, compiles in latest.items() if named.endswith(".py") and not compiles]
        if broken:
            notes.append(state.note(f"(no run: {', '.join(broken)} does not compile; write it again, whole)"))
            return f"nothing ran: {state.bare(broken[0])} does not compile"
        local = self.ws.reached()[0] if main.is_file() else set()
        imported = [named for named in python if named != mainpath and named.count("/") == 1
                    and Path(named).stem in local]
        targets: list[str] = []
        parts: list[str] = []
        if mainpath in python or imported:
            missing, packaged = self.ws.unwritten(self.ws.reached()[1], mainpath)
            if missing or packaged:
                notes.append(state.note("(no run: " + "; ".join(part for part in (
                    f"{mainpath} imports these modules not yet under {commands.WORK}/: "
                    + ", ".join(f"{name}.py" for name in missing[:12]) + f"; write them, and {mainpath} runs"
                    if missing else "", packaged) if part) + ")"))
                parts.append(f"main.py not run: it imports {missing[0]}, not yet written" if missing else
                             f"main.py not run: it imports {commands.WORK}/'s modules as a package")
            else:
                targets.append(mainpath)
                if mainpath not in python:
                    notes.append(state.note(f"({mainpath} runs: it imports {imported[0]})"))
        targets += [named for named in python if named != mainpath and named not in imported and commands.has_main_block(
            self.ws.root.joinpath(*named.split("/")).read_text(encoding="utf-8", errors="replace"))]
        if not targets and not parts:
            notes.append(state.note("(no run: " + (f"{mainpath} does not exist yet; write it last, and it runs)" if not
                                                   main.is_file() else f"{mainpath} does not import "
                                                   f"{', '.join(python)}, and none has an `if __name__ == "
                                                   "\"__main__\":` block)")))
            return "nothing ran"
        runs: list[state.Report] = []
        closing: list[str] = []         # the lines each run's observation closed with (`tail`), the latest last

        def tail(done: commands.Ran | None) -> str:
            closing.append(self.tail(done, False))
            return closing[-1]

        for target in targets:
            limit, line = self.run_limit()
            if limit < MIN_RUN_S:
                notes.append(state.note(f"(no run of {target}: {max(self.left(), 0):.0f} s left)"))
                parts.append(f"{state.bare(target)} not run: too little time")
                break
            self.counts["implicit_runs"] += 1
            self.counts["module_runs"] += target == mainpath and mainpath not in python
            said, ran = self.ws.run(target, limit, None, tail)
            self.ran(ran, limit)
            if target != mainpath and mainpath not in targets and main.is_file() and target.count("/") == 1:
                notes.append(state.note(state.SELF_RUN.format(named=target, main=mainpath)))
            if self.past_reserve(ran, limit, line):
                if ran is not None:
                    runs.append(state.Report(said, ran, closing[-1] if closing else ""))
                parts.append(state.outcome(ran) if ran is not None else f"{state.bare(target)}'s run refused")
                break
            if ran is None:
                notes.append(state.note(said, refusal=True))
                parts.append(f"{state.bare(target)}'s run refused")
                continue
            runs.append(state.Report(said, ran, closing[-1] if closing else ""))
            same = None
            if ran.exit != 0 or ran.error:
                key = (state.where(ran), ran.error.split(":", 1)[0])
                same = self.state_failed.setdefault(key, turn)
                same = same if same != turn else None
            elif ran.clean and not ran.findings:
                self.state_keep(turn)
            parts.append(state.outcome(ran, same))
        if runs:
            self.state_last = (turn, runs)
        return "; ".join(parts)

    def state_last_call(self, turn: int, reply) -> str:
        """Lane clock1: the last call's reply in the state form: its whole blocks saved as `state_write` saves them (a
        cut one's definitions kept as its draft), its directives not read; then work/main.py run with all the time left
        (`run_last`). The stop's reason."""
        read = state.read(reply.text, reply.finish_reason)
        self.wrote_first(reply, bool(read.blocks))
        self.log("request", f"turn {turn}: {reply.prompt_tokens:,} prompt and {reply.completion_tokens:,} completion "
                            f"tokens, finish {reply.finish_reason}; the last call; " + (read.describe() or "nothing read"))
        self.state_notes = (turn, [])
        for block in read.blocks:
            if block.cut:
                self.state_salvage(block)
                continue
            written = self.state_write(block, turn)
            self.lastcall["saved"] = self.lastcall.get("saved", 0) + (written[2] is not None)
            if written[3]:
                break
        trace.event("turn", turn=turn, calls=[part for part in read.describe().split(", ") if part], ended=True)
        return self.run_last()

    def state_keep(self, turn: int) -> None:
        """A clean run's output (exit 0, the checks passing, no finding) copied beside work/ (`CLEAN_OUTPUT`): what the
        stuck end puts back. A copy that fails keeps none, logged."""
        aside = self.work / state.CLEAN_OUTPUT
        try:
            with guard.WRITING:
                shutil.rmtree(aside, ignore_errors=True)
                shutil.copytree(self.out, aside, symlinks=True)
            self.state_clean, self.state_clean_turn = output_hashes(self.out), turn
        except Exception as exc:  # noqa: BLE001 - the loop goes on with no clean output kept
            self.state_clean = None
            self.log("state", f"the clean output not kept ({type(exc).__name__})")

    def state_stuck(self, named: str, turns: list[int]) -> str:
        """The stuck detector's end (`stuck_ended`): an identical file resent a `STUCK_ENDS`-th time ends the unit on the
        best clean output so far, the last clean run's (`state_keep`), put back where a later run changed it; with
        none kept, the output stands as it is. The stop's reason."""
        self.counts["stuck_ended"] = 1
        aside = self.work / state.CLEAN_OUTPUT
        if self.state_clean is None or not aside.is_dir():
            how = "no run was clean, so the output stands as it is"
        elif output_hashes(self.out) == self.state_clean:
            how = f"the output is turn {self.state_clean_turn}'s clean run's"
        else:
            try:
                with guard.WRITING:
                    clear(self.out)
                    shutil.copytree(aside, self.out, symlinks=True, dirs_exist_ok=True)
                how = f"turn {self.state_clean_turn}'s clean output put back"
            except Exception as exc:  # noqa: BLE001 - the unit ends whatever the copy met
                how = f"turn {self.state_clean_turn}'s clean output not put back whole ({type(exc).__name__})"
        self.log("stuck", f"{named} resent identical {len(turns)} times (turns {', '.join(map(str, turns))}); {how}")
        trace.event("stuck", turns=turns)
        return f"stuck: {named} resent identical {len(turns)} times; the unit ends, {how}"

    def state_lib(self, name: str) -> state.Note:
        """A `lib` line answered as the loop's `lib` is (the function's whole docstring, or the shelf's cards, within
        `LIB_BYTES`), as each LAST RUN setting says it (`state.card`); a function it names joins the state's cards,
        the latest first (`state_cards`)."""
        self.counts["lib_calls"] += 1
        found = self.library.find(name) if self.library is not None else None
        if found is not None:
            self.state_opened = [found, *(function for function in self.state_opened if function.name != found.name)]
            self.opened(found.name)
            said = self.ws.lib(name=name)
        elif self.library is not None and self.library.shelf(name) is not None:
            said = self.ws.lib(shelf=name)
        else:
            said = self.ws.lib(name=name)
        self.log("lib", first_line(said)[:120])
        return state.card(name, said, found)

    # ------------------------------------------------------------------------------------------------- solve

    def solve(self) -> None:
        task = self.unit = unit.read(self.task_dir)
        try:
            size = (self.task_dir / "instruction.md").stat().st_size
        except OSError:
            size = len(task.instruction.encode())
        self.band = "loop" if size >= (LOOP_BYTES if self.form == "review" and not self.review_loop else 0) else "n0"
        self.log("unit", f"{size:,} bytes of instruction ({task.dropped} canary line(s) removed), band {self.band}, "
                         f"mode {self.mode}, form {self.form} ({self.form_source}), "
                         f"{self.deadline - time.time():.0f} s to the deadline; roster cap "
                         f"{(roster := model.roster_cap())[0]:g} s ({roster[1]})")
        self.log("settings", f"form {self.form} ({self.form_source}); input files {self.inputs_setting} "
                             f"({self.inputs_source}); review thinking "
                             f"{self.review_thinking} ({self.review_thinking_source}); requests {self.client.cap} "
                             f"({CAP_SOURCE if self.client.cap == CAP else 'given'}); no call {self.nocall_setting} "
                             f"({self.nocall_source}); sketch picks {self.picks_setting} ({self.picks_source}); "
                             f"pace {self.pace_setting} ({self.pace_source}); bounds in {measure.UNIT.said()} "
                             f"({measure.UNIT_SOURCE}); fallback {self.resort.setting} ({self.resort.source}"
                             + ("" if self.resort.armed or self.resort.setting == "off" else
                                f", not armed in form {self.form}") + f"); conventions {self.convcard_setting} "
                             f"({self.convcard_source}" + ("" if self.form in CONVCARD_FORMS or
                                                          self.convcard_setting == "off" else
                                                          f", not armed in form {self.form}") + ")"
                             + f"; cut {self.cut_setting} ({self.cut_source}"
                             + ("" if self.form in CUT_FORMS or self.cut_setting == "split" else
                                f", not armed in form {self.form}") + ")"
                             + f"; cards {self.cards_setting} ({self.cards_source}"     # lane first1's three
                             + ("" if self.form in CARDS_FORMS or self.cards_setting == "cut" else
                                f", not armed in form {self.form}")
                             + (", whole under lib never" if self.lib_never and self.cards_setting != "whole" else "")
                             + ")"
                             + f"; inspect {self.inspect_setting} ({self.inspect_source}"
                             + ("" if self.form in INSPECT_FORMS or self.inspect_setting == "open" else
                                f", not armed in form {self.form}") + ")"
                             + f"; select {self.selection.setting} ({self.selection.source}"     # lane select2's
                             + ("" if self.selection.armed or self.selection.setting == "off" else
                                f", not armed in form {self.form}")
                             + f"), thinking {self.selection.thinking} ({self.selection.thinking_source})"
                             + f"; select bound {self.selection.bound} ({self.selection.bound_source})"
                             + f"; last call {self.lastcall_setting} ({self.lastcall_source}); shim "     # lane clock1
                             f"{self.shim_setting} ({self.shim_source}); lint {self.lint_setting} ({self.lint_source})"
                             + f"; write {self.write_setting} ({self.write_source}"     # lane fixdef1's
                             + ("" if self.form in WRITE_FORMS or self.write_setting == "whole" else
                                f", not armed in form {self.form}") + ")"
                             + f"; redirect {self.redirect_setting} ({self.redirect_source}"   # lane redirect1's
                             + ("" if self.form in REDIRECT_FORMS or self.redirect_setting == "off" else
                                f", not armed in form {self.form}") + ")"
                             + f"; lib {self.lib_setting} ({self.lib_source}"     # lane libnever1's
                             + ("" if self.form in LIB_FORMS or self.lib_setting == "open" else
                                f", not armed in form {self.form}") + ")"
                             + f"; mode {self.mode_setting} ({self.mode_source}"     # lane fenced1's
                             + ("" if self.form in MODE_FORMS or self.mode_setting == MODES[0] else
                                f", not armed in form {self.form}") + ")"
                             + f"; prompt {self.prompt_setting} ({self.prompt_source}"     # lane prompt1's
                             + ("" if self.prompt_setting == "full" or self.prompt_word != "full" else
                                f", not armed in form {self.form}" if self.form not in PROMPT_FORMS else
                                f", not armed in the {self.mode_setting} form") + ")"
                             + f"; lean cards {self.lean_cards_setting} ({self.lean_cards_source}"   # lane lean2's
                             + ("" if self.lean_cards_setting == "cut" or self.prompt_word in prompts.LEAN_WORDS else
                                f", not armed under prompt {self.prompt_word}") + ")"
                             + f"; called {self.called_setting} ({self.called_source}"     # lane called1's
                             + ("" if self.form in CALLED_FORMS or self.called_setting == "off" else
                                f", not armed in form {self.form}") + ")"
                             + f"; rules {self.rules_setting} ({self.rules_source}"     # lane rules1's
                             + ("" if self.form in RULES_FORMS or self.rules_setting == "off" else
                                f", not armed in form {self.form}") + ")"
                             + f"; resend {self.resend_setting} ({self.resend_source}"     # lane resend1's
                             + ("" if self.form in RESEND_FORMS or self.resend_setting == "refuse" else
                                f", not armed in form {self.form}") + ")"
                             + f"; finishrun {self.finishrun_setting} ({self.finishrun_source}"    # lane finishrun1's
                             + ("" if self.form in FINISHRUN_FORMS or self.finishrun_setting == "off" else
                                f", not armed in form {self.form}") + ")"
                             + f"; joinask {self.joinask_setting} ({self.joinask_source}"      # lane joinask1's
                             + ("" if self.joinask_setting == "same" or self.form in JOINASK_FORMS and self.continuing
                                else f", not armed in form {self.form}" if self.form not in JOINASK_FORMS else
                                ", not armed without CUT=continue") + ")"
                             + f"; finishfail {self.finishfail_setting} ({self.finishfail_source}"  # lane finishfail1's
                             + ("" if self.form in FINISHFAIL_FORMS or self.finishfail_setting == "off" else
                                f", not armed in form {self.form}") + ")"
                             + f"; finishcheck {self.finishcheck_setting} ({self.finishcheck_source}"  # lane
                             + ("" if self.form in FINISHCHECK_FORMS or self.finishcheck_setting == "off" else
                                f", not armed in form {self.form}") + ")"   # finishcheck1's
                             + f"; firstout {self.firstout_setting} ({self.firstout_source}"  # lane firstout1's
                             + ("" if self.form in FIRSTOUT_FORMS or self.firstout_setting == "off" else
                                f", not armed in form {self.form}") + ")"
                             + f"; clause {self.clause_setting} ({self.clause_source}"      # lane clauses1's
                             + ("" if self.form in CLAUSE_FORMS or self.clause_setting == "off" else
                                f", not armed in form {self.form}") + ")"
                             + f"; cardfail {self.cardfail_setting} ({self.cardfail_source}"  # lane cardfail1's
                             + ("" if self.form in CARDFAIL_FORMS or self.cardfail_setting == "off" else
                                f", not armed in form {self.form}") + ")"
                             + f"; repeat {self.repeat_setting} ({self.repeat_source}"  # lane repeat1's
                             + ("" if self.form in REPEAT_FORMS or self.repeat_setting == "off" else
                                f", not armed in form {self.form}") + ")"
                             + f"; lint2 {self.lint2_setting} ({self.lint2_source})"     # lane lint2's
                             + f"; example {self.example_setting} ({self.example_source}"    # lane example1's
                             + ("" if self.form in EXAMPLE_FORMS or self.example_setting == "off" else
                                f", not armed in form {self.form}") + ")"
                             + f"; snapshot {self.snapshot_setting} ({self.snapshot_source}"    # lane snapshot1's
                             + ("" if self.form in SNAPSHOT_FORMS or self.snapshot_setting == "off" else
                                f", not armed in form {self.form}") + ")"
                             + f"; join {self.join_setting} ({self.join_source}"    # lane join2 of 10-16's
                             + ("" if self.form in JOIN_FORMS or self.join_setting == "today" else
                                f", not armed in form {self.form}") + ")"
                             + f"; write thinking {self.write_thinking_setting} ({self.write_thinking_source}"  # lane
                             + ("" if self.form in WRITE_THINKING_FORMS or self.write_thinking_setting == "off" else
                                f", not armed in form {self.form}")     # thinkwrite1's, and its budget
                             + f"), budget {self.write_thinking_budget} ({self.write_thinking_budget_source})"
                             + f"; derive {self.derive_setting} ({self.derive_source}"    # lane derive1's
                             + ("" if self.form in DERIVE_FORMS or self.derive_setting == "off" else
                                f", not armed in form {self.form}") + ")"
                             + f"; contract {self.contract_setting} ({self.contract_source}"    # lane contract1's
                             + ("" if self.form in CONTRACT_FORMS or self.contract_setting == "off" else
                                f", not armed in form {self.form}") + ")"
                             + f"; stated {self.stated_setting} ({self.stated_source}"    # lane conventions1's
                             + ("" if self.form in STATED_FORMS or self.stated_setting == "off" else
                                f", not armed in form {self.form}") + ")"
                             + f"; recon {self.recon_setting} ({self.recon_source}"    # lane recon1's
                             + ("" if self.form in RECON_FORMS or self.recon_setting == "off" else
                                f", not armed in form {self.form}") + ")"
                             + f"; audit {self.audit_setting} ({self.audit_source}"    # lane audit1's
                             + ("" if self.form in AUDIT_FORMS or self.audit_setting == "off" else
                                f", not armed in form {self.form}") + ")"
                             + f"; drafts {self.drafts_setting} ({self.drafts_source}"    # lane drafts1's
                             + ("" if self.form in DRAFTS_FORMS or self.drafts_setting == "off" else
                                f", not armed in form {self.form}") + ")"
                             + f"; repair {self.repair.setting} ({self.repair.source}"    # lane repairpair1's
                             + ("" if self.form in REPAIR_FORMS or self.repair.setting == "off" else
                                f", not armed in form {self.form}") + ")")
        if self.mode_setting == "fenced" and self.form in MODE_FORMS:
            self.log("mode", f"fenced ({self.mode_source}): the fenced form from the loop's first request")
        pathmap = paths.build(task.task_dir, task.instruction, task.dockerfile, self.out)
        data = digest.digest(task.data_dir, canaries=task.canaries)
        files = [path.name for path in task.data_dir.rglob("*") if path.is_file()] if task.data_dir.is_dir() else []
        named = contract.named_all(pathmap, task.instruction, files)
        self.log("digest", f"{len(data.encode()):,} bytes; {len(pathmap.entries)} path(s) named; {len(named)} output "
                           f"file(s) named")
        lib, index = library.load(HERE)
        if self.index_mode == "scored":
            index = lib.index(task=task.instruction)
        self.index = index
        self.counts["index_bytes"], self.counts["index_functions"] = len(index.encode("utf-8")), lib.whole(index)
        self.counts["index_scored_hits"] = sum(each.hit for each in lib.scored(task.instruction))
        self.client.marks = {**(getattr(self.client, "marks", None) or {}),
                             **{key: self.counts[key] for key in INDEX_KEYS}, MEASURE_KEY: measure.UNIT.name}
        self.fired = lib.pointers(task.instruction)
        if self.prompt_word == "minimal" and self.fired:   # lane prompt2: no pointer line, no card, no gate
            self.log("prompt", "minimal: the pointer's " + ", ".join(function.call for function, _ in self.fired)
                     + " left unsaid")
            self.fired = []
        self.cards, self.cuts = lib.carded(self.fired)
        if self.lib_never:          # lane libnever1: no lookup named, the library pass's cards too
            self.cards = cards_unsaid(self.cards)
        pointer = pointer_line(self.fired, lookup=not self.lib_never)
        self.counts["pointer_units"] = int(bool(self.fired))
        self.log("library", f"{len(lib.functions)} function(s) on {len(lib.shelves)} shelves, an index of "
                            f"{len(index.encode()):,} B" + (f" ({self.index_mode} from {self.index_source}, "
                                                            f"{self.counts['index_functions']} shown whole)"
                                                            if self.index_mode != "full" else "") + "; pointer: "
                            + (", ".join(function.call for function, _ in self.fired) or "none")
                            + f"; cards {len(self.cards.encode()):,} B")
        for cut in self.cuts if not self.whole_armed else ():     # under `whole` the cards are fitted again
            self.log("cards", cut)
        self.stated = props.stated(task.instruction, named, rules=self.rules_armed, wide=self.rules_wide)
        self.counts["instruction_props"] = len(self.stated.props)
        self.log("stated", f"{len(self.stated.props)} propert(ies) of the outputs; {len(self.stated.keys)} JSON "
                           f"file(s) with {sum(map(len, self.stated.keys.values()))} key(s), "
                           f"{len(self.stated.columns)} table(s) with {sum(map(len, self.stated.columns.values()))} "
                           f"column(s) named")
        self.named_files = tuple(named)
        self.contract_read(task.instruction, named)
        self.ws = commands.Workspace(self.work, self.out, task, pathmap, lib, named, stated=self.stated,
                                     shimmed=self.shim_setting == "on", linted=self.lint_setting == "on",
                                     rules=self.rules_armed, wide=self.rules_wide, cardfail=self.cardfail_armed,
                                     lint2=self.lint2_setting == "on", snapshotted=self.snapshot_armed,
                                     witnessed=self.derive_armed,
                                     contracted=self.contract_terms if self.contract_armed else None,
                                     reconciled=self.recon_armed)
        self.library = lib
        if self.called_armed:           # lane called1: the cards that reach the House, gathered from here on
            self.ws.carded = {}
        if self.form == "state":
            self.state_loop(task, pathmap)
            return
        self.repeat_with = (task, pathmap, data)    # lane repeat1: what a hand-off sends the one-shot with
        if self.band == "n0" and self.first(task, pathmap, data):
            return
        index = self.selected(task, lib, index)         # lane select2: the selection call, where armed
        index = self.made_whole(lib, index)             # lane first1: the cards whole, where armed
        block = self.drafts_first(task, pathmap, data)  # lane drafts1: the first drafts, where armed
        # the loop's opening, built as the loop begins: under form `review` after request 1, its clock and budget
        # those the loop starts with, before what request 1 left in the conversation
        opening = self.first_turn(task, pathmap, data, named, pointer)
        self.framing = self.redact(opening)    # lane repairpair1: the task as the rewrite frames it
        first = opening + self.short_start()     # lane pace1's close
        if self.form == "sketch":           # lane plan4: the loop form's first message and its closing ask
            first += self.sketch_ask()
        if self.clause_armed:
            first += f"\n{CLAUSE_SAID[self.clause_setting]}\n"
        if self.derive_armed:           # lane derive1: the dated derivation's ask, a paragraph of its own
            first += f"\n{derive.ASK}\n"
        if block:                       # lane drafts1: the drafts' scripts and table, the first message's last block
            first += f"\n{block}"
        if self.prompt_word == "taskfirst":     # lane prompt1: design B's last line, after any drafts' block
            first += f"\n{prompts.TASKFIRST_CLOSE}\n"
        self.messages = [{"role": "system", "content": self.worded(index)},
                         {"role": "user", "content": self.redact(first)}, *self.messages]
        self.prompt_layout = prompts.layout(self.messages[1]["content"])
        self.loop()
        if self.rules_armed:            # lane rules1: an earlier clean output put back at an end with no request left
            with guard.WRITING:
                self.put_back()
        if self.repeat_floor is not None:   # lane repeat1: the one-shot's output, where the loop's fails the checks
            with guard.WRITING:
                self.repeat_end()
        self.last_resort(task, pathmap, data)

    def last_resort(self, task, pathmap, data: str) -> None:
        """Lane fallback1: where the loop stopped for one of `fallback.REASONS` with nothing the harness could grade
        (`fallback.graded`), N0's one-shot request (`fallback.Resort.fire`), sent with a request and `ASK_S` left;
        nothing where the switch is off or not armed, the output holds a file to grade, or the House did not answer."""
        if not self.resort.armed or self.resort.used or fallback.graded(self.out, self.ws.named):
            return              # once a unit: lane repeat1's hand-off may have sent it
        if self.end_kind not in fallback.REASONS:
            self.log("fallback", "not sent: the House did not answer the loop")
            return
        self.resort.fire(self, task, pathmap, data, self.end_kind, ASK_S)

    # ------------------------------------------------------------------------------------ the first write

    def wrote_first(self, reply, writes: bool) -> None:
        """Lane scaffold1's addendum: the first reply that wrote a file (a write or a block, whole or cut) or was cut
        at the cap, its completion tokens and whether the cap cut it (`first_write_tokens`, `first_write_cut`)."""
        tokens = int(reply.completion_tokens or 0)
        cut = reply.finish_reason == "length" or tokens >= getattr(self.client, "max_output", model.MAX_OUTPUT_TOKENS)
        if self.first_write is None and (writes or cut):
            self.first_write = (tokens, cut)
            self.log("first write", f"{tokens:,} completion tokens" + ("; cut at the cap" if cut else ""))

    def first_write_keys(self) -> dict:
        """The first file-writing reply's tokens and cut (`FIRST_WRITE_KEYS`, `-` where none)."""
        return {"first_write_tokens": self.first_write[0] if self.first_write else "-",
                "first_write_cut": int(self.first_write[1]) if self.first_write else "-"}

    def summary(self) -> str:
        """The counts the registered read takes, one line (scripts/run_sweep_read.py N11_KEYS)."""
        imported, called = self.used() if self.ws else (False, set())
        keys = {"mode": self.mode, "band": self.band or "-", "turns": self.counts["turns"],
                "tool_malformed": self.counts["tool_malformed"], "cut_replies": self.counts["cut_replies"],
                "pointer_units": self.counts["pointer_units"],
                "pointer_called": int(bool(called & {function.name for function, _ in self.fired})),
                "pointer_shown": self.counts["pointer_shown"], "lib_rewritten": self.rewritten(),
                "pointer_request1": self.counts["pointer_request1"],
                "cards_request1_bytes": self.counts["cards_request1_bytes"],
                "libpass_units": self.counts["libpass_units"], "libpass_adopted": self.counts["libpass_adopted"],
                "libpass_rejected": self.counts["libpass_rejected"],
                "libpass_request_bytes": self.counts["libpass_request_bytes"],
                "lib_calls": self.counts["lib_calls"],
                "qf_imported": int(imported), "qf_calls": ",".join(sorted(called)) or "-",
                "gate_fired": self.counts["gate_fired"], "gate_adopted": self.counts["gate_adopted"],
                "finish_clean": self.counts["finish_clean"], "runs": self.counts["runs"],
                "implicit_runs": self.counts["implicit_runs"],
                "sanity_units": self.counts["sanity_units"], "sanity_opened": self.counts["sanity_opened"],
                "finish_refused": self.counts["finish_refused"],
                "digest_bytes_median": round(statistics.median(self.digests)) if self.digests else 0,
                "cards_cut": self.counts["cards_cut"], "no_call": self.counts["no_call"],
                "write_repeats": self.counts["write_repeats"], "drafts": self.counts["drafts"],
                "unchanged_main": self.counts["unchanged_main"],
                "instruction_props": self.counts["instruction_props"],
                "instruction_findings": self.counts["instruction_findings"],
                "contract_findings": self.counts["contract_findings"], "param_unread": self.counts["param_unread"],
                "same_frame": self.counts["same_frame"],
                "continuations": self.counts["continuations"],
                "continuation_whole": self.counts["continuation_whole"],
                "no_call_capped": self.counts["no_call_capped"], "module_runs": self.counts["module_runs"],
                "form": self.form, "review_opened": self.counts["review_opened"],
                "review_requests": self.counts["review_requests"],
                "review_changed": self.counts["review_changed"], "review_kept": self.counts["review_kept"],
                **self.reply_keys(),
                **self.prompt_keys(called),     # lane prompt1's
                "state_bytes": round(statistics.median(self.state_bytes)) if self.state_bytes else 0,
                "last_run_bytes": round(statistics.median(self.last_run_bytes)) if self.last_run_bytes else 0,
                "files_bytes": round(statistics.median(self.files_bytes)) if self.files_bytes else 0,
                "ledger_lines": self.ledger_sent, "identical_refused": self.counts["identical_refused"],
                "stuck_ended": self.counts["stuck_ended"],
                "state_result": self.result_setting if self.form == "state" else "-",
                **{key: self.counts[key] for key in (*CLOCK_COUNTS, *INDEX_KEYS)},
                "files_written": len(self.wrote),
                "single_file": int(solution(self.ws.dir if self.ws else None) == [MAIN_FILE]),
                "sketch_mode": self.sketch_mode if self.sketch_on else "-", "sketch_thinking": self.sketch_thought,
                **{key: self.counts[key] for key in SKETCH_COUNTS}, "index_mode": self.index_mode,
                "index_scored_hits": self.counts["index_scored_hits"], MEASURE_KEY: measure.UNIT.name,
                **self.selection.keys(),        # lane select2's, after the index's
                **self.convcard_keys(),         # lane convcard1's, after them
                "inputs_mode": self.inputs_setting, "inputs_bytes": len(self.carried_inputs().encode()),
                "review_thinking": self.review_thinking if self.form == "review" else "-",
                "review_reasoning_tokens": self.counts["review_reasoning_tokens"],
                "review_cut": self.counts["review_cut"], "requests_cap": self.client.cap - self.repair.raised(),
                **{key: self.counts[key] for key in FENCE_COUNTS},
                "nocall_mode": self.nocall_setting if self.form != "state" else "-",
                "picks_mode": self.picks_setting if self.sketch_on and self.form != "state" else "-",
                **{key: self.counts[key] for key in CAP1_COUNTS}, **self.pace_keys(), **self.first_write_keys(),
                **self.resort.keys(), **self.cut_keys(), **{key: self.counts[key] for key in REVIEW_KEYS},
                **self.first1_keys(), **self.lastcall_keys(), **self.clock1_keys(), **self.redirect_keys(),
                **self.system_keys(), **self.lib_keys(called), **self.mode_keys(), **self.called_keys(),
                **self.write_keys(), **self.rules_keys(), **self.resend_keys(), **self.finishrun_keys(),
                **self.joinask_keys(), **self.finishfail_keys(), **self.firstout_keys(), **self.clause_keys(),
                **self.cardfail_keys(), **self.repeat_keys(), **self.lint2_keys(), **self.example_keys(),
                **self.finishcheck_keys(), **self.snapshot_keys(), **self.join_keys(), **self.write_thinking_keys(),
                **self.derive_keys(), **self.contract_keys(), **self.stated_keys(), **self.recon_keys(),
                **self.audit_keys(), **self.drafts_keys(), **self.output_keys(), **self.repair_keys()}
        return " ".join(f"{key}={value}" for key, value in keys.items())

    def write_thinking_keys(self) -> dict:
        """Lane thinkwrite1's keys (`WRITE_THINKING_KEYS`): the setting and the budget (`-` outside
        `WRITE_THINKING_FORMS`, the budget `-` under `off`), the requests that asked, their replies' reasoning tokens,
        those over the budget, and whether the switch stood down."""
        return self.thought.keys(self.form in WRITE_THINKING_FORMS)

    def example_keys(self) -> dict:
        """Lane example1's keys (`EXAMPLE_KEYS`): the setting (`-` outside `EXAMPLE_FORMS`), the example's unit, its
        overlap to three decimals, its measure in `measure.UNIT` and whether a part of it was cut (`-` and 0 where none
        was shown)."""
        return (self.example_shown or example.Shown()).keys(self.example_setting if self.form in EXAMPLE_FORMS
                                                            else "-")

    def lint2_keys(self) -> dict:
        """Lane lint2's keys (`LINT2_KEYS`): the setting (armed in every form, as `LINT` is), and the findings the
        unit's writes said of each kind (commands.Workspace.lint2_found)."""
        found = getattr(self.ws, "lint2_found", None) or Counter()
        return {"lint2_mode": self.lint2_setting, "lint2_shadowed": found["shadowed"], "lint2_unused": found["unused"]}

    def write_keys(self) -> dict:
        """Lane fixdef1's keys (`WRITE_KEYS`): the setting (`-` outside `WRITE_FORMS`), the writes spliced, those that
        added a definition, the splices refused for a result that does not compile, and the texts that looked partial
        saved whole (commands.Workspace.spliced)."""
        spliced = getattr(self.ws, "spliced", None) or Counter()
        return {"write_mode": self.write_setting if self.form in WRITE_FORMS else "-",
                **{key: spliced[key] for key in commands.SPLICE_KEYS}}

    def mode_keys(self) -> dict:
        """Lane fenced1's key (`MODE_KEYS`) beside today's `mode`: where the reply form the unit ended in came from,
        the switch's variable or file where it named the form, `refused` where the route refused the tools and the
        loop turned fenced (`MODE_REFUSED`), `form` under form `state` (fenced whatever the switch says), `given` a
        test's, and `-` where nothing named it (today's tools form, or a word refused)."""
        source = "form" if self.form == "state" else MODE_REFUSED if self.mode_refused else \
            self.mode_source if self.mode_source in (MODE_ENV, MODE_FILE, "given") else "-"
        return {"mode_source": source}

    def called_keys(self) -> dict:
        """Lane called1's keys (`CALLED_KEYS`): the setting (`-` outside `CALLED_FORMS`); a called finding said after
        any run or at any finish; a finish refused on its lines alone; after a finding, a later run's files no longer
        computing that function by hand and calling it (`called_ran`); and a function a finding named still computed
        by hand by the files at the end."""
        kept = bool(self.called_named) and any(rewrite.function.name in self.called_named
                                               for rewrite in self.ws.called())
        return {"called_mode": self.called_setting if self.form in CALLED_FORMS else "-",
                **{key: self.counts[key] for key in CALLED_KEYS[1:4]}, "called_kept": int(kept)}

    def clock1_keys(self) -> dict:
        """Lane clock1's run and write keys of 10-13 (`SHIM_KEYS`, `LINT_KEYS`): each setting, the shim's kinds that
        fired by the runs they fired in (`paths:2,json:1`, `-` where none) and the runs where one fired, the write's
        findings by kind and the writes it flagged."""
        return {"shim_mode": self.shim_setting, "shim_fired": by_kind(self.shim_fired),
                "shim_runs": self.counts["shim_runs"], "lint_mode": self.lint_setting,
                "lint_findings": by_kind(getattr(self.ws, "lint_found", None) or Counter()),
                "lint_writes": getattr(self.ws, "lint_writes", 0)}

    def lib_keys(self, called: set[str]) -> dict:
        """Lane libnever1's keys (`LIB_KEYS`): the setting (`-` outside `LIB_FORMS`), the lookups answered in one line
        under `never`, and under either setting the functions the House's files call (`called`, `qf_calls`' reading)
        whose card the first message did not carry (`carried`; `-` outside the forms)."""
        armed = self.form in LIB_FORMS
        return {"lib_mode": self.lib_setting if armed else "-", "lib_asked": self.counts["lib_asked"],
                "lib_blind": len(called - self.carried_cards()) if armed else "-"}

    def carried_cards(self) -> set[str]:
        """The functions whose card the first message's cards carry, whole or cut: each whose typed signature opens a
        line of them (`library.Card`, as `show` reads them); under lane prompt1's `lean` and lane leanconst1's
        `leanconst` (`prompts.LEAN_WORDS`), the lean cards'."""
        if self.prompt_carded is not None and self.prompt_word in prompts.LEAN_WORDS:
            return {function.name for function in self.prompt_carded}
        heads = set(self.cards.split("\n")) if self.cards else set()
        return {function.name for function in self.library.functions if function.typed in heads} \
            if self.library is not None else set()

    def system_keys(self) -> dict:
        """Lane putaway1's keys (`SYSTEM_KEYS`): the tools the last loop request's system prompt described, by name
        (`-` where the loop sent none), and the loop's requests whose system prompt differed from its first's."""
        said = self.system_said
        return {"system_tools": ",".join(name for name in commands.NAMES if name not in said[0] + said[1])
                if said is not None else "-", "system_changed": self.counts["system_changed"]}

    def prompt_keys(self, called: set[str]) -> dict:
        """Lane prompt1's keys (`PROMPT_KEYS`), every one `-` outside `PROMPT_FORMS`: the word run (`prompt_word`), the
        system prompt's bytes before its cards, the functions the House's files call (`called`) whose card it did not
        carry, whether the first message holds design B's order (`prompts.layout`), and the unit's first attempted
        work/main.py read for design C's blocks (`prompts.constants_read`): three constants or more, the percent whose
        comment quotes the task or says it silent, every OUTPUTS name in the task. The first attempt is read only
        where it was saved as sent (`constants_first`, `prompts.FIRST`); an attempt refused or another text saved
        first, and no attempt, are unreached (0, the percent `-`). Both reads are logged. Lane lean2's two: the lean
        cards' form where `lean` or `leanconst` runs (`prompts.LEAN_WORDS`; `-` elsewhere) and their bytes in the
        system prompt (`-` where neither sent any). Lane clean1's: the first loop request's messages, their bytes as
        sent (`said`; `-` where the loop sent none)."""
        if self.form not in PROMPT_FORMS:
            return {key: "-" for key in PROMPT_KEYS}
        first = prompts.FIRST[2] if self.prompt_attempt is None else \
            prompts.FIRST[0] if self.prompt_attempt_saved and self.prompt_first == self.prompt_attempt \
            else prompts.FIRST[1]
        read = prompts.constants_read(self.prompt_attempt) if first == prompts.FIRST[0] else None
        task = self.unit.instruction if self.unit is not None else ""
        quoted = read.quoted(task) if read is not None else None
        if first != prompts.FIRST[2]:
            said = [(name, prompts.constants_read(text)) for name, text in (("attempted", self.prompt_attempt),
                                                                             ("saved", self.prompt_first)) if text]
            self.log("prompt", f"the first work/main.py {first}; " + "; ".join(
                f"the first {name}: {len(each.conventions)} constant(s), outputs "
                f"{'none' if each.outputs is None else len(each.outputs)}" for name, each in said))
        carded = {function.name for function in self.prompt_carded or ()}
        return {"prompt_mode": self.prompt_word,
                "prompt_bytes_system": self.prompt_head if self.prompt_head is not None else "-",
                "lib_uncarded_calls": len(called - carded) if self.prompt_carded is not None else "-",
                "prompt_layout": self.prompt_layout if self.prompt_layout is not None else "-",
                "constants_block": read.block() if read is not None else 0,
                "constants_quoted": quoted if quoted is not None else "-",
                "outputs_block": read.outputs_named(task) if read is not None else 0, "constants_first": first,
                "lean_cards_mode": self.lean_cards_setting if self.prompt_word in prompts.LEAN_WORDS else "-",
                "lean_cards_bytes": self.lean_cards_bytes if self.lean_cards_bytes is not None else "-",
                "prompt_bytes_first": self.prompt_first_bytes if self.prompt_first_bytes is not None else "-"}

    def first1_keys(self) -> dict:
        """Lane first1's keys (`CARDS_KEYS`, `INSPECT_KEYS`; the selection call's bound is among its own): each
        setting (`-` outside its forms), the cards the first message carries whole and the index's bytes given up for
        them, the inspects answered from the first message's description and those answered as before, and the
        requests sent without the inspect tool."""
        return {"cards_mode": self.cards_setting if self.form in CARDS_FORMS else "-",
                "cards_whole": self.cards_whole() if self.form in CARDS_FORMS else "-",
                "index_trimmed_for_cards": self.whole.given_up if self.whole is not None else 0,
                "inspect_mode": self.inspect_setting if self.form in INSPECT_FORMS else "-",
                **{key: self.counts[key] for key in INSPECT_KEYS[1:]}}

    def cut_keys(self) -> dict:
        """Lane review2's cut keys (`CUT_KEYS`): the setting (`-` outside `CUT_FORMS`), the continuation asks made, the
        files the stitch completed (compiling), and the most replies cut in a row (under either setting)."""
        return {"cut_mode": self.cut_setting if self.form in CUT_FORMS else "-",
                "cut_continued": self.counts["cut_continued"], "cut_stitched": self.counts["cut_stitched"],
                "cut_streak_max": self.streak_top, **{key: self.counts[key] for key in JOIN_KEYS}}

    def convcard_keys(self) -> dict:
        """Lane convcard1's keys (`CONVCARD_KEYS`): the card's setting (`-` outside `CONVCARD_FORMS`), its lines and
        its size in the process's unit (`measure`; 0 where none was built), and the unit-run's span and scale
        findings (1 where a run said one)."""
        return {"conventions_mode": self.convcard_setting if self.form in CONVCARD_FORMS else "-",
                "conventions_terms": len(self.convcard.terms) if self.convcard else 0,
                "conventions_tokens": self.convcard_size, "span_finding": self.counts["span_finding"],
                "scale_finding": self.counts["scale_finding"]}

    def pace_keys(self) -> dict:
        """Lane pace1's keys: the setting (`-` but under the loop's forms `loop` and `sketch`), the measured pace
        (`pace_s`, every loop request's median, `-` before any reply), the requests the unit's whole clock bought at
        it (`requests_bought`: the time left at the loop's first message over the pace and the runs' median
        seconds), and the mode's counts (`PACE_COUNTS`)."""
        pace = self.pace()
        bought = self.bought(pace, self.clock0) if pace is not None and self.clock0 is not None else None
        return {"pace_mode": self.pace_setting if self.form in ("loop", "sketch") else "-",
                "pace_s": f"{pace:.1f}" if pace is not None else "-",
                "requests_bought": f"{bought:.1f}" if bought is not None else "-",
                **{key: self.counts[key] for key in PACE_COUNTS}}

    def summarise(self) -> None:
        if not self.summarised:
            self.summarised = True
            try:
                self.log("summary", self.summary())
            except Exception as exc:  # noqa: BLE001 - the summary never costs the close
                self.log("summary", f"unavailable ({type(exc).__name__})")


def solve(task_dir: Path, out: Path, *, client=None, work: Path | None = None, requests: int | None = None,
          exit=os._exit, watchdog_s: float = guard.WATCHDOG_S, stream=None, mode: str | None = None,
          form: str | None = None) -> int:
    """One unit, start to close; 0 whatever happens. The cap is `CAP` (`REQUESTS` unless `AGENTHON_N11_REQUESTS` or
    agent/n11/REQUESTS names another within `REQUESTS_BOUNDS`, lane inputs1); `requests` lowers it (a desk probe's
    budget), never raises it; `client`, `exit`, `watchdog_s`, `mode` and `form` (else the process's, `FORM`) are a
    test's. The client's usage line names the form (common.house.Client.form)."""
    out = Path(out)
    steps = guard.Steps(AGENT, lambda: client.usage.requests if client is not None else 0, stream)
    keep = work is not None
    canaries: frozenset[str] = frozenset()
    watchdog = agent = None
    try:
        out.mkdir(parents=True, exist_ok=True)
        trace.begin(AGENT, out)
        if client is None:
            client = house.client(Path(task_dir) / "card.toml", max(0, min(CAP if requests is None else requests,
                                                                            CAP)), out)
        deadline = client.deadline if client.deadline is not None else time.time() + model.UNIT_SECONDS
        work = Path(work) if keep else Path(tempfile.mkdtemp(prefix=f"{AGENT}-"))
        work.mkdir(parents=True, exist_ok=True)
        agent = N11(task_dir, out, client, steps, work, deadline, mode or MODE, form or FORM,
                    "given" if form else FORM_SOURCE)
        client.form = agent.form
        if agent.form == "state":           # its LAST RUN setting (lane state2) on the usage line too
            client.marks = {"state_result": agent.result_setting}

        def finish() -> None:
            agent.release()
            agent.summarise()
            closed = guard.close(out, agent.unit.canaries if agent.unit else ())
            steps("close", closed.summary())

        watchdog = guard.Watchdog(deadline, agent.kill, finish, steps, reserve=watchdog_s, exit=exit,
                                  moved=agent.moved).arm()
        agent.solve()
    except BaseException as exc:  # noqa: BLE001 - the exit is 0 whatever failed
        steps("error", failure(exc))
    finally:
        try:
            if watchdog is not None:
                watchdog.cancel()
            if agent is not None:
                with guard.WRITING:
                    agent.repair.discard(client)
                agent.reviewed()            # the review's guard before the summary counts it; nothing outside it
                with guard.WRITING:         # lane drafts1: the held floor, where the output holds nothing to grade
                    agent.drafts_close()
                agent.summarise()
                if agent.unit is not None:
                    canaries = agent.unit.canaries
            with guard.WRITING:
                if agent is not None:
                    agent.settle()
                closed = guard.close(out, canaries)
            steps("close", closed.summary())
            if work is not None and not keep:
                shutil.rmtree(work, ignore_errors=True)
        except BaseException as exc:  # noqa: BLE001
            steps("error", failure(exc))
    return 0


def refused(ran: commands.Ran | None, wanted: list[str], wrote: list[str]) -> str:
    """Why the library pass's run is not adopted, "" where it is: it exits 0, the checks pass, it meets no sanity
    finding (the first run met none, or the pass is not asked) and it writes the same output files (`wanted`, the
    first run's) and no other."""
    if ran is None:
        return "its run was refused"
    if ran.exit is None:
        return "its run was stopped at its limit"
    if ran.exit != 0 or ran.error:
        return f"its run exited {ran.exit}" + (f" ({ran.error.split(':', 1)[0][:60]})" if ran.error else "")
    if ran.failures:
        return f"{len(ran.failures)} check(s) fail"
    if ran.findings:
        return f"{len(ran.findings)} sanity finding(s), where the first run had none"
    if wrote != wanted:
        return (f"it wrote {len(set(wrote) & set(wanted))} of the first run's {len(wanted)} output file(s) and "
                f"{len(set(wrote) - set(wanted))} other(s)")
    return ""


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def output_hashes(out: Path) -> dict[str, str]:
    """Each regular file under the output directory and its bytes' sha256 (the review's and form `state`'s guards)."""
    found = {}
    try:
        for path in sorted(Path(out).rglob("*")):
            if path.is_file() and not path.is_symlink():
                found[path.relative_to(out).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        pass
    return found


PARAMETER = re.compile(r"<parameter=(\w+)>\n?(.*?)(?:\n?</parameter>|\Z)", re.S)


def cut_from_text(text: str) -> dict[str, str]:
    """A cut call's arguments from the reply's text where the route parsed no call: a `<tool_call>`'s JSON
    `arguments` as far as they got, or its `<parameter=...>` blocks."""
    text = text or ""
    at = text.rfind('"arguments"')
    if at >= 0:
        found = commands.cut_arguments(text[at + len('"arguments"'):])
        if found:
            return found
    return {name: value for name, value in PARAMETER.findall(text) if name in ("path", "text")}


def clear(out: Path) -> None:
    """Every entry of the output directory removed, a link as itself, never followed."""
    for entry in sorted(Path(out).iterdir()):
        if entry.is_dir() and not entry.is_symlink():
            shutil.rmtree(entry, ignore_errors=True)
        else:
            entry.unlink(missing_ok=True)


def refilled(source: Path, out: Path) -> None:
    """The output directory cleared (`clear`) and `source`'s entries copied into it, links as themselves (lane
    repeat1: the loop's output put back after the one-shot, or the one-shot's at the end)."""
    clear(out)
    for entry in sorted(Path(source).iterdir()):
        (shutil.copytree if entry.is_dir() and not entry.is_symlink() else shutil.copy2)(
            str(entry), str(Path(out) / entry.name), **({"symlinks": True} if entry.is_dir() and not entry.is_symlink()
                                                       else {"follow_symlinks": False}))


def failure(exc: BaseException) -> str:
    """The exception's type and where it was raised; a model error's own message (model.py's, which never carries a
    credential), with any canary line dropped. No other message is logged: it could quote the instruction."""
    frame = traceback.extract_tb(exc.__traceback__)[-1] if exc.__traceback__ else None
    where = f" at {Path(frame.filename).name}:{frame.lineno}" if frame else ""
    said = f": {logged(str(exc))[:200]}" if isinstance(exc, model.ModelError) else ""
    return f"{type(exc).__name__}{where}{said}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="solve", description="N11: a short tool-call loop over small files and "
                                                               "the method library.")
    parser.add_argument("verb", choices=["solve"])
    parser.add_argument("--task-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--work", default=None, help="keep the House's files and the runs' streams here (the desk's; "
                                                      "the image's go to a temporary directory, removed at the end)")
    parser.add_argument("--requests", type=int, default=None, help="at most this many requests, below the cap "
                                                                     f"({CAP}: a desk probe's budget)")
    try:
        args = parser.parse_args(argv)
    except SystemExit:
        return 0
    return solve(Path(args.task_dir), Path(args.out), work=Path(args.work) if args.work else None,
                 requests=args.requests)


if __name__ == "__main__":
    try:
        main()
    finally:
        # The status is 0 whatever the interpreter's own shutdown meets: a stderr nobody reads fails its last flush,
        # which Python reports as status 120.
        for stream in (sys.stdout, sys.stderr):
            try:
                stream.flush()
            except (AttributeError, OSError, ValueError):
                pass
        os._exit(0)
