# AGENTS.md

This repository exchanges task material with the paper repository through an
ignored `workspace/exchange/` buffer.

## Repository Exchange

- Read incoming task batches from `workspace/exchange/inbox/`.
- Copy completed results and a concise handoff note into a clearly named batch
  under `workspace/exchange/outbox/`.
- Do not move or modify source files merely to prepare a handoff.
- Do not automatically clear either inbox or outbox.
- Do not commit exchange-buffer contents.
- Treat the batch instructions and the user's prompt as the authority for what
  to process; the transport layer must not filter, rename, promote, or delete
  files.

The paper-side workstation initiates both network operations with
`repo-exchange push` and `repo-exchange pull`. This server repository does not
need a reverse SSH connection or a `.exchange.env` file.
