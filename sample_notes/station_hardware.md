# Station Hardware Specification

Every Project Harbor station is assembled from the same parts list. All part
names below are invented for this sample corpus.

## Sensor board

| Measurement | Sensor model | Range | Stated accuracy |
|---|---|---|---|
| Barometric pressure | Veridian BP-7 | 300 to 1100 hPa | +/- 0.6 hPa |
| Air temperature | Veridian TH-4 | -40 to 85 C | +/- 0.3 C |
| Relative humidity | Veridian TH-4 | 0 to 100 % RH | +/- 2 % RH |
| Wind speed | Caldwell AN-2 cup anemometer | 0 to 55 m/s | +/- 0.5 m/s |

The barometric sensor is the **Veridian BP-7**. The temperature and humidity
readings come from a single combined part, the Veridian TH-4, which is why a
humidity fault and a temperature fault are always reported together.

## Logger and radio

The logger is a Thistle M0 micro-controller board with 256 KB of flash. It
samples the sensor board, buffers readings locally and transmits them to the
gateway over a 868 MHz sub-gigahertz radio link. Local buffering holds up to
**48 hours** of readings, so a gateway outage shorter than two days loses
nothing.

## Power budget

| Item | Figure |
|---|---|
| Solar panel rating | 12 W |
| Battery capacity | 84 Wh |
| Average station draw | 1.4 Wh per hour |
| Winter autonomy target | 3 days without sun |

Average draw across a full day is therefore well inside the battery capacity,
and the winter autonomy target is the figure the steering group uses when it
decides whether a shaded site is viable.

## Enclosure and mounting

The sensor board sits inside a stacked-plate radiation shield on a 2 m mast.
The anemometer mounts on a 0.4 m cross-arm above the shield. The battery and
logger live in a sealed box at the base of the mast, rated IP65.

## Upgrade path

The current field units all carry the first-generation sensor board. A
second-generation board, the **Veridian TH-5 upgrade kit**, adds a heated
humidity element that reduces condensation error in autumn. The kit is a
drop-in replacement for the sensor board and needs no change to the mast, the
logger or the radio. Fitting one kit takes a trained volunteer about 40
minutes, including the post-fit calibration check.

The per-kit price and the budget line that would pay for a regional rollout are
not recorded here; they are in `budget_2026.md`. The number of stations in each
region is not recorded here either; it is in `deployment_log.txt`.

## Known hardware weaknesses

1. The cup anemometer under-reads below 0.6 m/s and reports zero below
   0.3 m/s, so calm periods are indistinguishable from a stalled cup.
2. The radiation shield traps insects in late summer, which biases the
   temperature reading upward by a few tenths of a degree until it is cleaned.
3. The battery box has no heater, so a prolonged frost reduces usable capacity
   and shortens the autonomy margin.

Each weakness has a matching check in the maintenance runbook.
