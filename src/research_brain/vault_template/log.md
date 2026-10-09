# Log

Append-only timeline of operations on this vault. Each entry starts with
`## [YYYY-MM-DD HH:MM] operation | title`, so `grep "^## \[" log.md | tail -5` shows the
last five operations.
