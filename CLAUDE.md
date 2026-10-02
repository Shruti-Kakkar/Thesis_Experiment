# Thesis Memory Instructions

This project uses a shared Mem0 MCP server for memory that persists across
Claude sessions on claude.ai, this machine, and other Claude Code instances.

## At the start of every session

Before starting work, call search_memories to check for relevant prior
context about this thesis project — decisions made, results obtained,
open questions, or conventions established in past sessions.

## During the session

Whenever any of the following comes up, call add_memory right away rather
than waiting until the end of the session:

- A decision about methodology, scope, or approach
- A result from an experiment, including numbers, plots, or file locations
- A problem encountered and how it was resolved
- A convention adopted for naming, structure, or workflow
- Anything explicitly flagged by the user as important to remember

If the fact updates or corrects something already stored, call
update_memory on the existing entry instead of creating a duplicate.

## Do not

Do not wait for the user to say "remember this." Save proactively.
Do not save transient details that will not matter in a future session,
such as which file is open right now or a one-off typo fix.