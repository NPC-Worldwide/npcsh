---
name: lookback
description: Reprint recent conversation messages to the console. Useful when terminal
  history is lost (e.g. switching panes/tabs). Does not re-inject messages into the
  conversation.
---

# lookback

Reprint recent conversation messages to the console. Useful when terminal history is lost (e.g. switching panes/tabs). Does not re-inject messages into the conversation.

## Inputs

- `n` (default: `'10'`)

## Steps

- `lookback` → [`lookback.py`](./lookback.py)

## Usage

```
/run_jinx jinx_ref=lookback input_values={"n": "10"}
```
