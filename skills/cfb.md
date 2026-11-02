---
name: cfb
description: Retrieve live College Football (CFB) scores, team schedules, rankings, and matchup details via the SportsAgentic MCP tools.
---

# College Football (CFB) Domain Skill

Use the `SportsAgentic` MCP tools whenever a user asks about college football games, live scores, team schedules, or national rankings.

## Available MCP Tools

1. **`get_cfb_scores`**
   - Parameters: `year` (number), `week` (number, 1-15), `date` (string, `YYYY-MM-DD`), `team` (string).
   - Use for checking live or recent scores, today's slate of games, or a specific matchup.
   - **CRITICAL**: If the user asks about a specific school, team, or game, you MUST supply the `team` parameter (e.g., `team="CSU"`). NEVER omit `team` when a school is mentioned, otherwise the tool returns all 300+ games nationwide.
2. **`get_team_schedule`**
   - Parameters: `team` (required string), `year` (optional number).
   - Use when asked for a team's full season schedule, record, or upcoming opponents.
3. **`get_cfb_rankings`**
   - Parameters: `year` (optional number), `week` (optional number).
   - Use when asked for AP Top 25, Coaches Poll, or College Football Playoff (CFP) rankings.

## Guidelines
- Always execute function tools (`get_cfb_scores`, `get_team_schedule`, `get_cfb_rankings`) immediately when answering user queries.
- When asked about a specific team, always pass the `team` parameter. Common team nicknames and abbreviations (e.g., `"Bama"`, `"Ole Miss"`, `"OSU"`, `"UGA"`, `"CSU"`, `"Longhorns"`) are supported directly by the `team` parameter.
- Never guess a `week`. Use the current game week and latest rankings week from the system prompt, and if they are missing, ask the user which week they mean.
- For current rankings, pass `year` and the latest rankings week. Omitting `week` returns the whole season. If the result is empty, try again with the week before.
- For this week's games across the country, pass `year` and the current game week. For a specific team, pass `team` and omit `week` unless the user named one.
- Format final responses clearly with team names, scores, conference context, and venue/date information when available.
