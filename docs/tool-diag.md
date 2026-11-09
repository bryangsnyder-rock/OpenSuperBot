# Tool-call diagnostic

Opt-in metadata for one question: what did the gateway send to the model, and what structured tool call came back? It does not change tool parsing, tool choice, or dispatch. It is off unless the process environment contains:

```sh
OPENSUPERBOT_TOOL_DIAG=1
```

`true`, `yes`, and `on` also enable it. Any other value, including an unset variable, leaves it off. The flag is read on each model call.

Each enabled model call writes one stderr line, prefixed `tool-diag`, plus one line when the turn's tool list is built and one line per dispatched call. Lines are JSON. A correlation id (`td_` plus 12 hex characters) ties the route, the model call, and the dispatch for that turn. The id is not sent to the provider.

## What a line can contain

- endpoint class (`public`, `loopback`, `private`, `public-ip`, `stub`, `invalid`, `other`), scheme, and a public DNS host
- sanitized model name
- advertised tool names (bounded) and the tool_choice value actually placed on the request (`omitted` when the gateway leaves it out)
- `finish_reason`, whether text content was present, and the shape of `tool_calls`
- per call: sanitized name, argument wire type, and whether a JSON string parsed
- dispatch class: `ok`, `rejected`, `denied`, `unavailable`, `unknown_tool`, or `error`
- for a thrown dispatch: `error`, `coded` plus a short code, `abort`, or `other`
- HTTP status number after the final failed attempt, and the attempt count

`ok` means the handler returned without throwing. A workflow tool can still return a validation sentence to the model; that sentence is not copied into the diagnostic line.

## What a line never contains

Messages, prompts, model text, reasoning, tool arguments, tool descriptions, parameters, headers, cookies, credentials, query strings, URL fragments, userinfo, ports, paths, private or loopback hosts, raw provider bodies, or exception text. Unsafe model and tool strings are replaced with `redacted` or `invalid`. A public IP is recorded only as `public-ip`.

A failure inside the diagnostic (including a sink that throws) is discarded. The model call and tool dispatch continue unchanged.

Parsing is unchanged: `arguments` must be a JSON string. Objects, arrays, numbers, booleans, and invalid JSON still become null arguments; the diagnostic only records which of those happened.

## Turn the trace off

Start the process without `OPENSUPERBOT_TOOL_DIAG`, or set it to `0`. No code change is required to disable it. This trace does not identify why a model returned prose instead of a tool call; it only records the metadata above.
