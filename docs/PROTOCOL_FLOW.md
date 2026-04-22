# Protocol Flow

This note documents the intended measurement flow for the stable ViPSA path.

## DCIV step flow

1. Optional `ALIGN`
2. Optional `APPROACH`
3. Apply the commanded DCIV sweep segments
4. Optional state read-probe
5. Save sweep CSV, plot image, and metadata
6. Save resistance/read-probe CSV, plot image, and metadata when probe data exists

## LRS / HRS read-probe placement

When a DCIV step uses `read_probe_mode="between_segments"` and `include_read_probe=True`, the read-probes are taken at a fixed and explicit place in the sweep pipeline:

- `HRS` probe is taken after the positive return segment completes and the device has returned from the positive excursion toward zero.
- `LRS` probe is taken after the negative return segment completes and the device has returned from the negative excursion toward zero.

For the 4-way split this corresponds to:

- `pb` -> HRS probe
- `nb` -> LRS probe

This placement was chosen so the probe happens after the switching portion of the sweep has settled, not before the stress leg and not interleaved randomly inside a segment.

## Protocol metadata

Each saved dataset now carries protocol context in the JSON sidecar:

- current step index
- total number of steps
- current step parameters
- the full list of protocol steps

This makes it easier to reconstruct how a saved DCIV or pulse file was produced without relying on the UI state alone.
