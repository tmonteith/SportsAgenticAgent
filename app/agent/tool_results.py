"""Compact MCP tool results so a small llama-server context can answer from them."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple
from zoneinfo import ZoneInfo

RANKINGS_TOOL = "get_cfb_rankings"
SCORES_TOOL = "get_cfb_scores"
SCHEDULE_TOOL = "get_team_schedule"

# Most authoritative first: game ranks come from the first of these a week has.
MAJOR_POLLS = ("Playoff Committee Rankings", "AP Top 25", "Coaches Poll")

POWER_CONFERENCES = frozenset({"ACC", "Big 12", "Big Ten", "SEC"})
GROUP_OF_FIVE_CONFERENCES = frozenset(
    {
        "American Athletic",
        "Conference USA",
        "FBS Independents",
        "Mid-American",
        "Mountain West",
        "Pac-12",
        "Sun Belt",
    }
)

EASTERN = ZoneInfo("America/New_York")

# Room kept for the envelope around a nationwide game list.
ENVELOPE_CHARS = 400

WeekKey = Tuple[int, int]


def to_json(value: Any) -> str:
    """Serialize without whitespace, which costs tokens but tells the model nothing."""
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False)


def _is_nationwide_scores(tool_name: str, arguments: Mapping[str, Any]) -> bool:
    return tool_name == SCORES_TOOL and not arguments.get("team")


def needs_rankings(tool_name: str, arguments: Mapping[str, Any], result: str) -> bool:
    """Return whether a tool result is a nationwide game list, which needs ranks to pick from."""
    if not _is_nationwide_scores(tool_name, arguments):
        return False
    try:
        return _is_games(json.loads(result))
    except json.JSONDecodeError:
        return False


def compact_tool_result(
    tool_name: str,
    arguments: Mapping[str, Any],
    result: str,
    max_chars: int,
    rankings: Optional[str] = None,
    now: Optional[datetime] = None,
) -> str:
    """Return the JSON to show the model for a tool result.

    Only nationwide scores are held to max_chars. Rankings and single-team results are
    small once compacted. Anything that isn't a known CFBD shape is returned unchanged.
    """
    try:
        data = json.loads(result)
    except json.JSONDecodeError:
        return result

    if tool_name == RANKINGS_TOOL and _is_rankings(data):
        return _compact_rankings(data)
    if tool_name in (SCORES_TOOL, SCHEDULE_TOOL) and _is_games(data):
        if _is_nationwide_scores(tool_name, arguments):
            return _compact_nationwide_games(data, max_chars, rankings, now)
        return to_json(_games_envelope([_game_line(game) for game in data], len(data)))
    return result


def _is_rankings(data: Any) -> bool:
    return (
        isinstance(data, list)
        and bool(data)
        and all(isinstance(week, dict) and isinstance(week.get("polls"), list) for week in data)
    )


def _is_games(data: Any) -> bool:
    return (
        isinstance(data, list)
        and bool(data)
        and all(isinstance(game, dict) and "homeTeam" in game and "awayTeam" in game for game in data)
    )


def _week_key(item: Mapping[str, Any]) -> WeekKey:
    """Order weeks across a season: week 1 of the postseason follows the last regular week."""
    return (0 if item.get("seasonType") == "regular" else 1, int(item.get("week") or 0))


def _major_polls(week: Mapping[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    polls = {poll.get("poll"): poll.get("ranks") or [] for poll in week["polls"]}
    return {name: polls[name] for name in MAJOR_POLLS if name in polls}


def _rank_line(entry: Mapping[str, Any]) -> str:
    # CFBD rankings have no record today; keep it if the MCP server starts sending one.
    details = ", ".join(str(part) for part in (entry.get("conference"), entry.get("record")) if part)
    suffix = f" ({details})" if details else ""
    return f"{entry.get('rank')}. {entry.get('school')}{suffix}"


def _compact_rankings(weeks: List[Dict[str, Any]]) -> str:
    # Without a week argument CFBD returns the whole season, in no particular order.
    latest = max(
        (week for week in weeks if _major_polls(week)),
        key=_week_key,
        default=max(weeks, key=_week_key),
    )
    polls = _major_polls(latest)
    compacted: Dict[str, Any] = {
        "season": latest.get("season"),
        "seasonType": latest.get("seasonType"),
        "week": latest.get("week"),
        "polls": {name: [_rank_line(entry) for entry in ranks] for name, ranks in polls.items()},
    }

    notes: List[str] = []
    if len(weeks) > 1:
        notes.append(f"Only the latest of the {len(weeks)} weeks returned is shown.")
    left_out = [poll.get("poll") for poll in latest["polls"] if poll.get("poll") not in polls]
    if left_out:
        notes.append(f"Polls left out: {', '.join(str(name) for name in left_out)}.")
    if notes:
        compacted["note"] = " ".join(notes)
    return to_json(compacted)


def _kickoff(game: Mapping[str, Any]) -> Optional[datetime]:
    try:
        parsed = datetime.strptime(str(game.get("startDate"))[:19], "%Y-%m-%dT%H:%M:%S")
    except ValueError:
        return None
    return parsed.replace(tzinfo=timezone.utc)


def _nearest_week(games: Iterable[Mapping[str, Any]], now: datetime) -> WeekKey:
    """Pick the week by kickoff date, since unplayed games linger in past weeks."""

    def distance(game: Mapping[str, Any]) -> timedelta:
        kickoff = _kickoff(game)
        return abs(kickoff - now) if kickoff else timedelta.max

    return _week_key(min(games, key=distance))


def _ranks_for_week(rankings: Optional[str], week: WeekKey) -> Tuple[Optional[str], Dict[str, int]]:
    """Return (poll label, school to rank) from the latest poll going into a game week."""
    try:
        data = json.loads(rankings) if rankings else None
    except json.JSONDecodeError:
        return None, {}
    if not _is_rankings(data):
        return None, {}

    eligible = [w for w in data if _major_polls(w) and _week_key(w) <= week]
    if not eligible:
        return None, {}
    latest = max(eligible, key=_week_key)
    name, ranks = next(iter(_major_polls(latest).items()))
    return (
        f"{name}, week {latest.get('week')}",
        {entry["school"]: entry["rank"] for entry in ranks if entry.get("school") and entry.get("rank")},
    )


def _conference_tier(game: Mapping[str, Any]) -> int:
    conferences = {game.get("homeConference"), game.get("awayConference")}
    if conferences <= POWER_CONFERENCES:
        return 2
    if conferences & POWER_CONFERENCES:
        return 3
    if conferences <= POWER_CONFERENCES | GROUP_OF_FIVE_CONFERENCES:
        return 4
    return 5


def _notability(game: Mapping[str, Any], ranks: Mapping[str, int]) -> Tuple[int, int]:
    """Sort key: ranked matchups first, best ranks first, then by conference."""
    team_ranks = [ranks[team] for team in (game["homeTeam"], game["awayTeam"]) if team in ranks]
    if len(team_ranks) == 2:
        return (0, sum(team_ranks))
    if team_ranks:
        return (1, team_ranks[0])
    return (_conference_tier(game), 0)


def _when(game: Mapping[str, Any]) -> Optional[str]:
    """Say when a game is in US Eastern time, which the model misreads less than UTC stamps."""
    kickoff = _kickoff(game)
    if kickoff is None:
        return None
    local = kickoff.astimezone(EASTERN)
    day = f"{local:%a %b} {local.day}"
    if game.get("completed"):
        return day
    # CFBD gives games without a kickoff time midnight Eastern.
    if (local.hour, local.minute) == (0, 0):
        return f"{day}, time TBD"
    return f"{day}, {local.strftime('%I:%M %p').lstrip('0')} ET"


def _game_line(game: Mapping[str, Any], ranks: Optional[Mapping[str, int]] = None) -> str:
    """Describe a game in one string, which takes about 40% fewer tokens than an object."""

    def team(side: str) -> str:
        name = game[f"{side}Team"]
        rank = (ranks or {}).get(name)
        return f"#{rank} {name}" if rank else str(name)

    away_points, home_points = game.get("awayPoints"), game.get("homePoints")
    if away_points is None or home_points is None:
        matchup = f"{team('away')} at {team('home')}"
    elif game.get("completed") and away_points != home_points:
        # Naming the winner stops the model from reading a road win as a home win.
        sides = sorted(((away_points, "away"), (home_points, "home")), reverse=True)
        matchup = f"{team(sides[0][1])} beat {team(sides[1][1])} {sides[0][0]}-{sides[1][0]}"
    else:
        status = "final" if game.get("completed") else "in progress"
        matchup = f"{team('away')} {away_points} at {team('home')} {home_points} ({status})"

    conferences = [game.get("awayConference"), game.get("homeConference")]
    if conferences[0] == conferences[1]:
        conferences.pop()
    parts = [
        matchup,
        _when(game),
        game.get("venue"),
        "/".join(str(conference) for conference in conferences if conference),
    ]
    return ", ".join(str(part) for part in parts if part)


def _games_envelope(lines: List[str], total: int, note: str = "") -> Dict[str, Any]:
    envelope: Dict[str, Any] = {
        "totalGames": total,
        "gamesShown": len(lines),
        "gamesOmitted": total - len(lines),
    }
    if note:
        envelope["note"] = note
    envelope["games"] = lines
    return envelope


def _take_within(lines: Iterable[str], max_chars: int) -> List[str]:
    kept: List[str] = []
    used = 0
    for line in lines:
        used += len(to_json(line)) + 1
        if used > max_chars:
            break
        kept.append(line)
    return kept


def _compact_nationwide_games(
    games: List[Dict[str, Any]],
    max_chars: int,
    rankings: Optional[str],
    now: Optional[datetime],
) -> str:
    total = len(games)
    week = _nearest_week(games, now or datetime.now(timezone.utc))
    in_week = [game for game in games if _week_key(game) == week]
    poll, ranks = _ranks_for_week(rankings, week)

    notes: List[str] = []
    if len(in_week) < total:
        notes.append(f"Only week {week[1]} is included, the week nearest today.")
    notes.append(f"Ranks: {poll}." if poll else "No poll ranks were available.")

    everything = to_json(
        _games_envelope([_game_line(game, ranks) for game in in_week], total, " ".join(notes))
    )
    if len(everything) <= max_chars:
        return everything

    ordered = sorted(in_week, key=lambda game: _notability(game, ranks))
    kept = _take_within((_game_line(game, ranks) for game in ordered), max_chars - ENVELOPE_CHARS)
    notes.append(
        f"These are the {len(kept)} most notable games, best first. The other "
        f"{total - len(kept)} were left out, so this is not the full slate. "
        "Mention only the top few unless asked for more."
    )
    return to_json(_games_envelope(kept, total, " ".join(notes)))
