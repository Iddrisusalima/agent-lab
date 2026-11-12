# Project Harbor - Maintenance Runbook

The runbook is what a steward follows on call and what a volunteer follows on
site. All intervals below are invented for this sample corpus and are stated
only here.

## Scheduled intervals

| Activity | Interval |
|---|---|
| Humidity sensor calibration check | every 180 days |
| Pressure sensor calibration check | every 365 days |
| Anemometer spin test | every 90 days |
| Radiation shield cleaning | every 90 days |
| Battery capacity test | every 365 days |
| Mast and mounting inspection | every 365 days |

The humidity calibration interval is **180 days**. It is the shortest
calibration interval in the programme because the humidity element drifts
fastest, and because autumn condensation makes a stale humidity calibration
the most common source of a disputed reading.

## The quarterly sweep

One sweep covers every station in a region. A sweep combines the 90-day items:
the anemometer spin test and the shield cleaning. A single station visit takes
about 25 minutes when nothing is wrong. Cascade Foothills needs two days for a
full sweep because of driving distance; the other three regions fit into one
day each.

Fitting a Veridian TH-5 upgrade kit is not part of a sweep. It is scheduled
separately, because the post-fit calibration check has to be signed off by the
calibration lead.

## On-site checklist

1. Photograph the station before touching anything.
2. Read the logger status light: steady means buffering is healthy, slow blink
   means the buffer is more than half full, fast blink means a sensor fault.
3. Spin the anemometer by hand and confirm a non-zero wind speed appears in the
   next reading.
4. Open the shield, remove insects and debris, and close it without
   repositioning the sensor board.
5. Check the battery box seal and the cable gland for water.
6. Log the visit with the region, the date and the fault code, never with the
   host's name or address.

## Fault codes

| Code | Meaning | Resolution |
|---|---|---|
| `E-11` | sensor board not responding | reseat the ribbon cable, then replace the board |
| `E-12` | humidity and temperature both implausible | combined part fault, replace the Veridian TH-4 |
| `E-21` | wind speed zero for 12 consecutive readings | anemometer spin test on site |
| `E-31` | battery below the autonomy threshold | panel shading check, then battery replacement |
| `E-32` | panel voltage zero in daylight | cable or panel fault, replace the cable first |
| `E-41` | radio frames missing but logger healthy | gateway fault, not a station fault |

`E-12` always implicates a single part rather than two, because temperature and
humidity come from one combined sensor.

## Escalation

A steward handles one region for one week. Anything not resolved within five
working days goes to the steering group with the fault code, the region and the
number of affected stations. A fault affecting more than a quarter of a
region's stations is escalated immediately rather than after five days.

## Spares

The replacement pool is held centrally and topped up quarterly. A volunteer
collects a spare before a visit rather than ordering one after a fault is
confirmed, because a second visit costs more volunteer time than a spare that
sits on a shelf for a month. Unit prices for every spare are in
`budget_2026.md`.

## Safety

No work on a mast in winds above 12 m/s, no work alone above ground level, and
no mains work of any kind: every station is battery and solar only.
