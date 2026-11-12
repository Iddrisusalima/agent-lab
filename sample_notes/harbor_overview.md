# Project Harbor - Programme Overview

Project Harbor is an invented community weather-station network used as the
sample document corpus for Agent Lab. Nothing in this file describes a real
organisation, a real person or a real deployment.

## Purpose

Project Harbor builds and maintains low-cost ground weather stations hosted by
volunteers. Each station reports temperature, humidity, barometric pressure and
wind speed to a shared ingestion service, and the collected readings are
published as an open dataset once per week. The programme exists so that small
inland towns, which commercial forecasting networks cover poorly, have local
measurements of their own.

## Key facts

- Programme name: Project Harbor
- Founded: March 2023
- Network size: **62 active stations** as of the most recent review
- Regions covered: four, listed station by station in `deployment_log.txt`
- Governance: a three-person steering group, rotating every 12 months
- Dataset licence: open, published weekly as a single compressed archive

## The four subsystems

1. **Station** - the field unit: sensor board, solar panel, battery, radio.
   The parts list and the power budget are in `station_hardware.md`.
2. **Gateway** - a shed-mounted receiver that collects radio frames from every
   station within range and forwards them over a household internet connection.
3. **Pipeline** - the ingestion, validation and storage path described in
   `data_pipeline.md`, including the polling interval and the retention window.
4. **Publication** - the weekly archive build and the public dataset page.

## Roles

Project Harbor describes roles, never individuals. The roles are: the *host*,
a volunteer whose property carries a station; the *steward*, the on-call
volunteer who triages alerts for one region for a week at a time; the
*calibration lead*, who signs off sensor drift checks; and the *data editor*,
who approves the weekly archive before publication. The onboarding path for
each role is in `onboarding_handbook.md`.

## What the programme does not do

Project Harbor measures ground-level conditions only. It operates no marine
buoys, no balloon soundings and no radar. It does not forecast; it publishes
measurements. It does not sell data, and it does not accept sponsorship that
would restrict the open licence.

## Scope of this corpus

These sample files cover the programme overview, the station hardware, the
deployment history, the data pipeline, the 2026 budget, the maintenance
runbook and the onboarding handbook. They are deliberately self-consistent:
figures that appear in more than one file agree, and figures that appear in
only one file are stated only there, so that a question can be built that
requires reading two files to answer.
