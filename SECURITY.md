# Security

This is an evaluation harness and a local demonstrator, not a hosted service.
The notes below state what the code does at runtime so a report can be judged
against it.

## What the runtime does

- The demo service binds to `127.0.0.1` only. There is no remote-facing mode;
  the `--port` flag changes nothing about that.
- It writes no per-request details to disk. Counters, held sentences and the
  profile of the visitor in the browser are process state, and they end when
  the process ends.
- It makes no outbound network calls. The one exception is the optional
  encoded route: when the language extras are loaded, the sentence-embedding
  model is fetched from the Hugging Face Hub on first use unless it is already
  cached on the machine.
- There is no login and no account, because nothing the service holds belongs
  to a visitor. The release check refuses any file that carries a user
  identifier, an item's parent identifier, or a ranked list, so the public
  repository holds aggregate numbers and nothing that names a person's
  requests.

## Reporting a vulnerability

Send it to Ahnaf An Nafee at <ahnafnafee@gmail.com>, or open a private
advisory on the GitHub repository. There is no hosted deployment to patch, so
a report is acknowledged and answered rather than triaged against a release
cycle.
