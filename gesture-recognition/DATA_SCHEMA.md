# Data Schema (MLOps Lesson 2)

**Why we document this:** So anyone (or any script) can check that data is valid before training. If the app or the data collection changes the JSON format, validation will catch it.

## Expected folder layout

- **Training/** – JSON files used for training. Filename format: `LABEL.xxx.json` (e.g. `TAP.abc123.json`).
- **Testing/** – JSON files used for testing. Same filename format. No file should appear in both folders (leakage check).

## Allowed labels

Exactly these (case-insensitive): **TAP**, **WAVE**, **SHAKE**, **IDEAL**.

## JSON structure

Each file must be valid JSON with at least:

```json
{
  "payload": {
    "values": [ [accX, accY, accZ], [accX, accY, accZ], ... ],
    "interval_ms": 16
  }
}
```

- **payload.values**: Array of arrays. Each inner array has exactly 3 numbers (accX, accY, accZ).
- **payload.interval_ms**: Positive number (milliseconds between samples). Typically 16 for 62.5 Hz.

## Minimum length

For a 2 s window at 62.5 Hz we need at least **125 samples** per recording. Shorter recordings are still loaded but may be zero-padded; validation can warn.

## Validation checks (what `validate_data.py` does)

1. **Schema**: Each JSON has `payload`, `payload.values`, `payload.interval_ms`; values are numeric.
2. **No leakage**: No filename appears in both Training/ and Testing/.
3. **Labels**: Every training file’s label is one of TAP, WAVE, SHAKE, IDEAL; training has at least one file per label.
4. **Minimum samples**: Optional warning for recordings shorter than 125 samples.
