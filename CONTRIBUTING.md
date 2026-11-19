# Contributing

Thanks for taking a look. This project intentionally stays small.

## Ground rules

- Keep runtime dependencies at zero. The shipped CLI must work with the Python
  standard library only.
- Keep the implementation in `npu_top.py`; this is a single-file tool by design.
- Do not add telemetry, analytics, or writes to vLLM/NPU devices.
- Never commit API keys, hostnames, serial numbers, or real customer data.

## Tests

```bash
python3 -m py_compile npu_top.py
python3 -m unittest discover -s tests -v
```

When changing a parser, add a small fixture and a test for the exact format you
are supporting. For `npu-smi`, sanitize serial numbers and hostnames before
submitting a fixture.

## Pull requests

- One focused change per pull request.
- Explain which metric or hardware format is affected.
- Include the command you used to verify the change.
