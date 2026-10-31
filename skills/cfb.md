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
2. **`get_team_schedule`**
   - Parameters: `team` (required string), `year` (optional number).
   - Use when asked for a team's full season schedule, record, or upcoming opponents.
3. **`get_cfb_rankings`**
   - Parameters: `year` (optional number), `week` (optional number).
   - Use when asked for AP Top 25, Coaches Poll, or College Football Playoff (CFP) rankings.

## Guidelines
- Common team nicknames and abbreviations (e.g., `"Bama"`, `"Ole Miss"`, `"OSU"`, `"UGA"`, `"CSU"`, `"Longhorns"`) are supported directly by the `team` parameter.
- Format final responses clearly with team names, scores, conference context, and venue/date information when available.
