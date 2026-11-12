# Project Harbor - Onboarding Handbook

This handbook explains how a new volunteer joins Project Harbor and how the
programme runs week to week. It describes roles, never individuals, and holds
no contact detail, address or credential of any kind.

## Joining

A new volunteer starts as a *host* or as a *steward*. A host offers a site and
attends one commissioning visit. A steward takes on-call duty for one region
and shadows an experienced steward for two weeks before taking a week alone.

The induction is three short sessions:

1. **What the programme measures.** The four measurements, the subsystems and
   the published dataset. Reading `harbor_overview.md` first makes this session
   mostly a question-and-answer.
2. **What a station is.** The parts list, the power budget and the three known
   hardware weaknesses, all from `station_hardware.md`.
3. **What to do when an alert fires.** The fault codes and the escalation rule
   from `maintenance_runbook.md`.

A steward is signed off after completing one supervised on-call week with no
unescalated fault left open.

## Weekly rhythm

| Day | Activity |
|---|---|
| Monday | on-call handover between stewards |
| Wednesday | open call, 30 minutes, optional |
| Friday | calibration lead reviews any drift flagged that week |
| Sunday | archive build runs, data editor approves or rejects |

The on-call handover is the only meeting a steward must attend. Everything else
is optional, and nothing is decided in a meeting that is not written down
afterwards.

## Decisions

Decisions are made by the three-person steering group and recorded as a single
numbered line each. Any volunteer may propose a decision; the proposer writes
the line, and the steering group accepts, rejects or defers it at the next
Wednesday call. A purchase above the approval threshold in `budget_2026.md`
cannot be made on a steward's own authority.

## Data handling rules for volunteers

1. Never record a host's name, address or contact detail in a visit log, an
   alert note or the dataset.
2. Refer to a station by its identifier and its region only.
3. Never publish a photograph that shows a house number, a vehicle plate or a
   person.
4. Share the dataset by linking to the published archive rather than by sending
   extracts, so that a correction reaches everyone.
5. If a host withdraws, the station is retired and the identifier is never
   reused.

## Common questions

**Does hosting a station cost anything?** No. The programme pays for parts,
and the host provides a site, a little shade-free sky and, for a gateway host,
a household internet connection.

**How much work is hosting?** One commissioning visit, then a quarterly sweep
visit that a volunteer carries out. A host is not expected to maintain
anything.

**Can a host see their own station's data?** Yes, in the published weekly
archive, like everyone else. The programme runs no private view, because a
private view would be one more thing to secure.

**What happens to a station in a storm?** Nothing special. The safety rule
forbids mast work above 12 m/s wind, so a fault found during a storm waits
until conditions allow a visit.

## Leaving

A volunteer leaving gives two weeks' notice if they hold on-call duty, hands
back any spares from the replacement pool, and writes a handover line for each
open fault. No access to any system needs revoking, because stewards use no
shared account: every tool the programme runs is either public or operated by
the data editor alone.
