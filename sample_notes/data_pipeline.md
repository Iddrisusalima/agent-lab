# Project Harbor - Data Pipeline

This file describes how a reading travels from a station to the published
weekly archive. Every figure is invented for this sample corpus, and this file
is the only place where the pipeline timings, record sizes and retention
windows are stated.

## Stages

The pipeline has five stages, in order: collection, ingestion, validation,
storage and publication. Each stage has one owner role and one failure mode
that the steward is expected to recognise from the alert text alone.

## 1. Collection

Each station samples its sensor board once per minute and averages the samples
into a single reading. The logger transmits the averaged reading to its
gateway, and the gateway holds frames until the ingestion service asks for
them.

The ingestion service **polls each gateway every 15 minutes**. Fifteen minutes
is a compromise: short enough that a gateway outage is noticed within one
publication cycle, long enough that a household internet connection is not
saturated by the polling traffic. At one stored reading per poll, each station
contributes **96 readings per day**.

A reading that cannot be delivered stays in the logger's local buffer, which
holds 48 hours of data. When the gateway comes back, the buffer is drained
oldest first, and late readings are stored against the timestamp they were
measured at rather than the timestamp they arrived at.

## 2. Ingestion

The ingestion service is a single scheduled worker. It reads at most 500
readings per gateway per poll, writes them to the raw table and acknowledges
them to the gateway only after the write has committed. Acknowledging after
the commit is the reason a replayed frame can appear twice in the raw table:
the pipeline prefers a duplicate over a loss, and the duplicate is removed
during validation.

Each stored reading carries **12 measurement fields** and occupies **384 bytes**
in the raw table, including the station identifier, the measurement timestamp
and the ingestion timestamp. The 12 fields are: temperature, humidity,
pressure, wind speed, wind gust, battery voltage, panel voltage, logger
temperature, radio signal strength, sample count, firmware revision and a
quality flag.

With 96 readings per station per day and 384 bytes per reading, the daily raw
volume of a single station is small; the network total depends on how many
stations are active, which is recorded in `deployment_log.txt` and summarised
in `harbor_overview.md` rather than here.

## 3. Validation

Validation runs once per hour over everything ingested since the last run. It
applies six rules:

1. **Duplicate removal.** Two readings with the same station identifier and the
   same measurement timestamp collapse to one; the earlier ingestion wins.
2. **Range check.** A reading outside the sensor range stated in
   `station_hardware.md` is flagged rather than dropped.
3. **Rate-of-change check.** A temperature change of more than 4 C between
   consecutive readings is flagged as suspect.
4. **Stall check.** Twelve consecutive identical wind-speed readings of zero
   are flagged as a possible stalled anemometer cup.
5. **Battery check.** A battery voltage below the autonomy threshold raises a
   low-power alert for the steward on call.
6. **Clock check.** A measurement timestamp more than 120 seconds ahead of the
   ingestion timestamp is rejected outright, because a station clock running
   fast would otherwise place readings in the future.

Flagged readings stay in the dataset with the quality flag set. Rejected
readings are written to a reject log with the rule that rejected them. The
programme publishes the reject count with the weekly archive, because hiding it
would make the dataset look cleaner than it is.

## 4. Storage

Validated readings live in the readings table and are **retained for 540
days**. Retention is deliberately longer than a full year so that any month can
be compared with the same month of the previous year before the older data
expires. Expiry runs weekly and deletes whole days at a time, never partial
days, so a query never sees a half-trimmed day.

The raw table is pruned far more aggressively: raw readings are deleted 14 days
after they have been validated. The raw table exists to make a validation bug
recoverable, not to serve queries.

Three indexes cover the readings table: station and timestamp, region and
timestamp, and quality flag. Any query pattern outside those three is expected
to be slow, and the data editor is asked to raise a request rather than add an
index directly.

## 5. Publication

The archive build runs **every Sunday at 02:00 UTC**. It selects the trailing
seven days of validated readings, writes one compressed CSV per region plus one
network-wide file, computes a checksum for each file and waits for the data
editor's approval. Approval is a manual step by design: a bad archive is public
the moment it is published, and the programme would rather publish late than
republish.

If the data editor does not approve an archive within 48 hours, the build is
discarded and the next Sunday's build covers 14 days instead of 7. This has
happened twice since the programme started.

## Failure modes the steward should recognise

| Alert text | Meaning | First action |
|---|---|---|
| `GATEWAY_SILENT` | no poll succeeded for 90 minutes | check the host's internet connection |
| `BUFFER_NEAR_FULL` | a logger is within 6 hours of its 48-hour buffer limit | expect data loss; escalate the gateway fault |
| `CLOCK_REJECT` | the clock check rejected readings | reflash the logger during the next sweep |
| `STALL_SUSPECT` | the stall check fired | inspect the anemometer on site |
| `VALIDATION_LATE` | the hourly validation run has not completed | check the worker, do not run a second one |

Running a second validation worker is the one action the runbook forbids
outright, because two workers deduplicating the same window can each decide
that the other's row is the duplicate.

## What the pipeline does not store

The pipeline stores measurements and station telemetry. It does not store host
names, host addresses, exact station coordinates or any contact detail. A
station identifier maps to a region and nothing finer, and the mapping from
identifier to host is kept outside the dataset entirely. No rainfall, snow
depth, solar irradiance or soil measurement is collected, because no station
carries a sensor for any of them.
