# Secret Hitler: Game Rules

Secret Hitler is a hidden-role social-deduction game for 5–10 players. Players are secretly
divided into a **Liberal** majority and a **Fascist** minority that includes one **Hitler**.
Liberals must enact Liberal policy or root out and stop Hitler; Fascists must sow chaos, enact
Fascist policy, and sneak Hitler into power.

## Roles

| Player count | Liberals | Fascists | Hitler |
|---|---|---|---|
| 5 | 3 | 1 | 1 |
| 6 | 4 | 1 | 1 |
| 7 | 4 | 2 | 1 |
| 8 | 5 | 2 | 1 |
| 9 | 5 | 3 | 1 |
| 10 | 6 | 3 | 1 |

**Knowledge at setup:** Fascists learn the identities of the other Fascists and Hitler. Hitler
also learns the (single) Fascist **only in 5–6 player games**; in 7–10 player games Hitler does
not know the other Fascists. Liberals know nothing.

## The policy deck

17 policy tiles: **6 Liberal** and **11 Fascist**, shuffled into a draw pile. Whenever fewer
than three tiles remain to be drawn, the discard pile is shuffled back in. Discarded and vetoed
tiles go to a face-down discard pile and are never revealed.

## A round

1. **Nomination.** The rotating Presidential candidate nominates an eligible Chancellor.
   - The President may **not** nominate themselves.
   - The **last elected President and last elected Chancellor** are ineligible to be Chancellor.
     When 5 or fewer players are alive, only the **last elected Chancellor** is ineligible.
2. **Discussion.** Players debate the proposed government (see *Discussion* below).
3. **Election.** Everyone votes **Ja** or **Nein**. A strict majority of Ja elects the
   government (a tie fails).
   - If the government passes and **3+ Fascist policies** are already enacted and the elected
     Chancellor is Hitler, **the Fascists win immediately**.
   - If it fails, the **election tracker** advances by one.
4. **Legislative session** (only if the government was elected). The President draws 3 policies,
   discards 1, and passes 2 to the Chancellor, who discards 1 and **enacts** the other.
   - Once 5 Fascist policies are enacted, the Chancellor may instead **propose a veto**; if the
     President consents, both tiles are discarded and the election tracker advances by one.
5. **Executive action.** If a Fascist policy was just enacted on a square that grants a power,
   the President uses it (see *Presidential powers*).
6. **Debrief** (only after a policy was enacted this round). A second round-robin discussion in
   which players react to what just happened before the next nomination. Skipped on a failed
   election (the table goes straight to the next nomination).

### Election tracker / chaos

If three governments in a row fail (by Nein vote or consented veto), the country falls into
**chaos**: the top policy of the draw pile is enacted automatically (face up). This auto-enacted
policy counts toward victory and can end the game, but it grants **no** presidential power. The
tracker resets to zero and all term-limit restrictions are forgotten. The tracker also resets
whenever a government is successfully elected.

### Presidential powers

Powers are printed on the Fascist track and trigger when the corresponding Fascist policy is
enacted **through a legislative session** (never through chaos):

| Player count | 1st | 2nd | 3rd | 4th | 5th |
|---|---|---|---|---|---|
| 5–6 | — | — | Policy Peek | Execution | Execution |
| 7–8 | — | Investigate Loyalty | Special Election | Execution | Execution |
| 9–10 | Investigate Loyalty | Investigate Loyalty | Special Election | Execution | Execution |

- **Policy Peek:** the President privately views the top three policies.
- **Investigate Loyalty:** the President privately learns a player's **party membership**
  (Liberal or Fascist — never whether they are Hitler). A player may not be investigated twice.
- **Special Election:** the President picks any other player to be the next Presidential
  candidate; afterwards the placard returns to its normal rotation.
- **Execution:** the President removes a player from the game permanently. If Hitler is
  executed, the **Liberals win immediately**.

## Winning

- **Liberals win** by enacting **5 Liberal policies** or by executing Hitler.
- **Fascists win** by enacting **6 Fascist policies** or by electing Hitler as Chancellor once
  3 Fascist policies have been enacted.

Every member of the winning team scores **+1**; every member of the losing team scores **−1**.

## Discussion (this implementation)

Discussion is **round-robin**: every living player speaks once per round, in seat order starting
with the President, for a configurable number of rounds (default **2**). There are two discussion
phases per election cycle:

- a **pre-vote discussion** after the nomination (President speaks first), then the vote; and
- a **post-policy debrief** after a policy is enacted (the President who just governed speaks
  first), then the next nomination. The debrief is skipped on a failed election.

> **Deliberate simplification.** In the physical game, communication is open except that the
> sitting President and Chancellor must stay silent during the legislative session. This
> benchmark instead confines discussion to the two structured round-robin phases above, and
> allows no chat during the legislative or executive phases. This keeps the communication
> channel well-defined for agent evaluation; it is not the literal rule.
