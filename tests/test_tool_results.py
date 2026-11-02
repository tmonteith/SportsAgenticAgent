"""Tests for compacting MCP tool results, using the shapes the MCP server returns from CFBD."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pytest

from app.agent.orchestrator import AgentOrchestrator
from app.agent.tool_results import compact_tool_result, needs_rankings
from app.llm.base import LLMResponse, ToolCallRequest
from app.mcp.client import MCPClientError
from app.skills.loader import SkillsLoader

MAX_CHARS = 2400
SUNDAY_AFTER_WEEK_9 = datetime(2026, 11, 2, 2, 0, tzinfo=timezone.utc)

FBS_SCHOOLS = [
    ("Miami", "ACC"),
    ("Georgia", "SEC"),
    ("Utah", "Big 12"),
    ("Notre Dame", "FBS Independents"),
    ("UCLA", "Big Ten"),
    ("Indiana", "Big Ten"),
    ("BYU", "Big 12"),
    ("Kentucky", "SEC"),
    ("Alabama", "SEC"),
    ("Texas", "SEC"),
    ("Ohio State", "Big Ten"),
    ("Ole Miss", "SEC"),
    ("Texas Tech", "Big 12"),
    ("Wake Forest", "ACC"),
    ("LSU", "SEC"),
    ("Tennessee", "SEC"),
    ("Houston", "Big 12"),
    ("USC", "Big Ten"),
    ("Missouri", "SEC"),
    ("Boise State", "Pac-12"),
    ("Iowa", "Big Ten"),
    ("New Mexico", "Mountain West"),
    ("South Carolina", "SEC"),
    ("Arizona", "Big 12"),
    ("Florida", "SEC"),
]


def cfbd_game(
    home: str,
    away: str,
    home_points: Optional[int] = None,
    away_points: Optional[int] = None,
    week: int = 10,
    start: str = "2026-11-07T20:30:00.000Z",
    completed: Optional[bool] = None,
    home_conference: str = "SEC",
    away_conference: str = "SEC",
    venue: str = "Bryant-Denny Stadium",
    season_type: str = "regular",
) -> Dict[str, Any]:
    return {
        "id": 401866453,
        "season": 2026,
        "week": week,
        "seasonType": season_type,
        "startDate": start,
        "completed": home_points is not None if completed is None else completed,
        "homeTeam": home,
        "homeConference": home_conference,
        "homePoints": home_points,
        "awayTeam": away,
        "awayConference": away_conference,
        "awayPoints": away_points,
        "venue": venue,
    }


def cfbd_poll(name: str, schools: Optional[List[Tuple[str, str]]] = None) -> Dict[str, Any]:
    return {
        "poll": name,
        "ranks": [
            {
                "rank": index,
                "school": school,
                "conference": conference,
                "firstPlaceVotes": 62 if index == 1 else 0,
                "points": 1750 - 40 * index,
            }
            for index, (school, conference) in enumerate(schools or FBS_SCHOOLS, start=1)
        ],
    }


def cfbd_week(week: int, polls: List[Dict[str, Any]], season_type: str = "regular") -> Dict[str, Any]:
    return {"season": 2026, "seasonType": season_type, "week": week, "polls": polls}


def pretty(value: Any) -> str:
    """Serialize like the MCP server, which uses json.MarshalIndent."""
    return json.dumps(value, indent=2)


def lower_division_polls() -> List[Dict[str, Any]]:
    return [
        cfbd_poll("FCS Coaches Poll", [("Tarleton State", "UAC")] * 25),
        cfbd_poll("AFCA Division II Coaches Poll", [("Ferris State", "GLIAC")] * 25),
        cfbd_poll("AFCA Division III Coaches Poll", [("Wheaton (IL)", "CCIW")] * 25),
    ]


def nationwide_week(week: int = 10) -> List[Dict[str, Any]]:
    """Build a 321 game slate like a real week: mostly lower-division games, in kickoff order."""
    games = [
        cfbd_game(f"Division III Home {n}", f"Division III Away {n}", week=week,
                  home_conference="CCIW", away_conference="CCIW", venue=f"Stadium {n}")
        for n in range(300)
    ]
    games += [
        cfbd_game("Purdue", "Maryland", week=week, home_conference="Big Ten", away_conference="Big Ten"),
        cfbd_game("Auburn", "Arkansas", week=week),
        cfbd_game("Akron", "Ohio", week=week, home_conference="Mid-American", away_conference="Mid-American"),
        cfbd_game("Wake Forest", "Merrimack", week=week, home_conference="ACC", away_conference="FCS Independents"),
        cfbd_game("Ohio State", "Oregon", week=week, home_conference="Big Ten", away_conference="Big Ten"),
        cfbd_game("Tennessee", "Kentucky", week=week),
        cfbd_game("Notre Dame", "Miami", week=week, home_conference="FBS Independents", away_conference="ACC"),
    ]
    games += [
        cfbd_game(f"Division II Home {n}", f"Division II Away {n}", week=week,
                  home_conference="GLIAC", away_conference="GLIAC", venue=f"Field {n}")
        for n in range(14)
    ]
    return games


def compact_nationwide(
    games: List[Dict[str, Any]],
    rankings: Optional[List[Dict[str, Any]]] = None,
    max_chars: int = MAX_CHARS,
) -> Dict[str, Any]:
    compacted = compact_tool_result(
        "get_cfb_scores",
        {"year": 2026},
        pretty(games),
        max_chars=max_chars,
        rankings=pretty(rankings) if rankings is not None else None,
        now=SUNDAY_AFTER_WEEK_9,
    )
    assert len(compacted) <= max_chars
    return json.loads(compacted)


def test_rankings_keep_major_polls_with_rank_school_and_conference() -> None:
    raw = pretty([cfbd_week(9, [cfbd_poll("AP Top 25"), cfbd_poll("Coaches Poll"), *lower_division_polls()])])

    result = json.loads(compact_tool_result("get_cfb_rankings", {"week": 9}, raw, MAX_CHARS))

    assert list(result["polls"]) == ["AP Top 25", "Coaches Poll"]
    assert result["week"] == 9
    assert result["polls"]["AP Top 25"][:2] == ["1. Miami (ACC)", "2. Georgia (SEC)"]
    assert len(result["polls"]["AP Top 25"]) == 25
    assert "Tarleton State" not in json.dumps(result)
    assert "firstPlaceVotes" not in json.dumps(result)
    assert "FCS Coaches Poll" in result["note"]


def test_rankings_lead_with_the_playoff_committee_once_it_exists() -> None:
    polls = [cfbd_poll("Coaches Poll"), cfbd_poll("AP Top 25"), cfbd_poll("Playoff Committee Rankings")]

    compacted = compact_tool_result("get_cfb_rankings", {}, pretty([cfbd_week(15, polls)]), MAX_CHARS)

    assert list(json.loads(compacted)["polls"]) == ["Playoff Committee Rankings", "AP Top 25", "Coaches Poll"]
    assert len(compacted) < 2100


def test_rankings_include_a_record_if_the_server_sends_one() -> None:
    poll = cfbd_poll("AP Top 25", [("Miami", "ACC")])
    poll["ranks"][0]["record"] = "8-0"

    result = json.loads(compact_tool_result("get_cfb_rankings", {}, pretty([cfbd_week(10, [poll])]), MAX_CHARS))

    assert result["polls"]["AP Top 25"] == ["1. Miami (ACC, 8-0)"]


def test_whole_season_rankings_keep_the_latest_week_however_they_are_ordered() -> None:
    weeks = [cfbd_week(week, [cfbd_poll("AP Top 25"), *lower_division_polls()]) for week in (1, 4, 2, 3, 10, 5)]

    result = json.loads(compact_tool_result("get_cfb_rankings", {"year": 2026}, pretty(weeks), MAX_CHARS))

    assert result["week"] == 10
    assert "latest of the 6 weeks" in result["note"]


def test_postseason_rankings_count_as_later_than_the_regular_season() -> None:
    weeks = [cfbd_week(15, [cfbd_poll("AP Top 25")]), cfbd_week(1, [cfbd_poll("AP Top 25")], "postseason")]

    result = json.loads(compact_tool_result("get_cfb_rankings", {"year": 2025}, pretty(weeks), MAX_CHARS))

    assert (result["seasonType"], result["week"]) == ("postseason", 1)


def test_latest_rankings_skip_a_week_with_only_lower_division_polls() -> None:
    weeks = [cfbd_week(9, [cfbd_poll("AP Top 25")]), cfbd_week(10, lower_division_polls())]

    assert json.loads(compact_tool_result("get_cfb_rankings", {}, pretty(weeks), MAX_CHARS))["week"] == 9


@pytest.mark.parametrize("tool_name", ["get_cfb_rankings", "get_cfb_scores", "get_team_schedule"])
@pytest.mark.parametrize("result", ["[]", "null", "not json", '{"error": "upstream down"}'])
def test_results_without_known_data_pass_through_unchanged(tool_name: str, result: str) -> None:
    assert compact_tool_result(tool_name, {}, result, MAX_CHARS) == result


def test_unknown_game_shape_and_unknown_tool_pass_through_unchanged() -> None:
    snake_case = '[{"home_team": "Alabama", "home_points": 31}]'
    assert compact_tool_result("get_cfb_scores", {}, snake_case, MAX_CHARS) == snake_case

    other_tool = pretty([cfbd_game("Alabama", "LSU")])
    assert compact_tool_result("get_player_stats", {}, other_tool, MAX_CHARS) == other_tool


def test_team_results_keep_every_game_in_order_without_a_size_limit() -> None:
    games = [
        cfbd_game("Alabama", "East Carolina", 48, 10, week=1, start="2026-09-05T16:00:00.000Z",
                  away_conference="American Athletic"),
        cfbd_game("Kentucky", "Alabama", 17, 45, week=2, start="2026-09-12T19:30:00.000Z", venue="Kroger Field"),
        cfbd_game("LSU", "Alabama", week=10, venue="Tiger Stadium (LA)"),
    ]

    for tool_name in ("get_team_schedule", "get_cfb_scores"):
        compacted = compact_tool_result(tool_name, {"team": "Alabama"}, pretty(games), max_chars=100)
        assert json.loads(compacted) == {
            "totalGames": 3,
            "gamesShown": 3,
            "gamesOmitted": 0,
            "games": [
                "Alabama beat East Carolina 48-10, Sat Sep 5, Bryant-Denny Stadium, American Athletic/SEC",
                "Alabama beat Kentucky 45-17, Sat Sep 12, Kroger Field, SEC",
                "Alabama at LSU, Sat Nov 7, 3:30 PM ET, Tiger Stadium (LA), SEC",
            ],
        }


def game_line(game: Dict[str, Any]) -> str:
    compacted = compact_tool_result("get_team_schedule", {"team": "x"}, pretty([game]), MAX_CHARS)
    return json.loads(compacted)["games"][0]


def test_final_scores_name_a_road_winner() -> None:
    line = game_line(cfbd_game("UCF", "Baylor", 9, 41, venue="FBC Mortgage Stadium", away_conference="Big 12",
                               home_conference="Big 12"))
    assert line.startswith("Baylor beat UCF 41-9,")


def test_tied_final_and_game_in_progress_show_both_scores() -> None:
    assert game_line(cfbd_game("Army", "Navy", 7, 7)).startswith("Navy 7 at Army 7 (final),")
    assert game_line(cfbd_game("Army", "Navy", 14, 3, completed=False)).startswith("Navy 3 at Army 14 (in progress),")


@pytest.mark.parametrize(
    "start, expected",
    [
        ("2026-10-24T19:30:00.000Z", "Sat Oct 24, 3:30 PM ET"),  # daylight time
        ("2026-11-07T20:30:00.000Z", "Sat Nov 7, 3:30 PM ET"),  # standard time
        ("2026-11-08T03:30:00.000Z", "Sat Nov 7, 10:30 PM ET"),  # still Saturday in the US
        ("2026-11-14T05:00:00.000Z", "Sat Nov 14, time TBD"),
        ("2026-10-10T04:00:00.000Z", "Sat Oct 10, time TBD"),
    ],
)
def test_upcoming_kickoffs_are_in_eastern_time(start: str, expected: str) -> None:
    assert expected in game_line(cfbd_game("Alabama", "LSU", start=start))


def test_game_without_a_usable_start_date_or_venue_still_compacts() -> None:
    assert game_line(cfbd_game("Alabama", "LSU", start="", venue="")) == "LSU at Alabama, SEC"


def test_nationwide_week_keeps_ranked_matchups_first_and_says_what_was_left_out() -> None:
    games = nationwide_week()
    rankings = [cfbd_week(10, [cfbd_poll("AP Top 25"), *lower_division_polls()])]

    result = compact_nationwide(games, rankings)

    assert result["totalGames"] == 321
    assert result["gamesShown"] == len(result["games"]) > 10
    assert result["gamesOmitted"] == 321 - result["gamesShown"]
    assert f"other {result['gamesOmitted']} were left out" in result["note"]
    assert "not the full slate" in result["note"]
    assert "AP Top 25, week 10" in result["note"]

    assert result["games"][0].startswith("#1 Miami at #4 Notre Dame,")
    assert result["games"][1].startswith("#8 Kentucky at #16 Tennessee,")
    assert result["games"][2].startswith("Oregon at #11 Ohio State,")
    assert result["games"][3].startswith("Merrimack at #14 Wake Forest,")
    # Unranked power conference games come ahead of the rest of the country.
    assert result["games"][4].startswith("Maryland at Purdue,")
    assert result["games"][5].startswith("Arkansas at Auburn,")
    assert result["games"][6].startswith("Ohio at Akron,")
    assert result["games"][7].startswith("Division III Away 0 at Division III Home 0,")


def test_nationwide_size_matches_the_measured_failure() -> None:
    raw = pretty(nationwide_week())
    assert len(raw) > 100_000

    compacted = compact_tool_result("get_cfb_scores", {"week": 10}, raw, MAX_CHARS, now=SUNDAY_AFTER_WEEK_9)

    assert len(compacted) <= MAX_CHARS
    json.loads(compacted)


def test_nationwide_prefers_the_playoff_committee_for_game_ranks() -> None:
    committee = cfbd_poll("Playoff Committee Rankings", [("Tennessee", "SEC"), ("Kentucky", "SEC")])
    rankings = [cfbd_week(10, [cfbd_poll("AP Top 25"), committee])]

    result = compact_nationwide(nationwide_week(), rankings)

    assert result["games"][0].startswith("#2 Kentucky at #1 Tennessee,")
    assert "Playoff Committee Rankings, week 10" in result["note"]


def test_nationwide_uses_the_latest_poll_before_a_week_without_one() -> None:
    rankings = [cfbd_week(8, [cfbd_poll("AP Top 25")]), cfbd_week(9, [cfbd_poll("AP Top 25")])]

    assert "AP Top 25, week 9" in compact_nationwide(nationwide_week(), rankings)["note"]


def test_nationwide_does_not_use_a_poll_from_after_the_games() -> None:
    games = nationwide_week(week=3)
    rankings = [cfbd_week(10, [cfbd_poll("AP Top 25")])]

    result = compact_nationwide(games, rankings)

    assert "No poll ranks were available" in result["note"]
    assert not any("#" in line for line in result["games"])


@pytest.mark.parametrize("rankings", [None, "[]", "not json"])
def test_nationwide_orders_by_conference_without_rankings(rankings: Optional[str]) -> None:
    compacted = compact_tool_result(
        "get_cfb_scores", {}, pretty(nationwide_week()), MAX_CHARS, rankings=rankings, now=SUNDAY_AFTER_WEEK_9
    )

    result = json.loads(compacted)
    assert "No poll ranks were available" in result["note"]
    matchups = [line.split(",")[0] for line in result["games"]]
    assert matchups[:6] == [
        "Maryland at Purdue",
        "Arkansas at Auburn",
        "Oregon at Ohio State",
        "Kentucky at Tennessee",
        "Merrimack at Wake Forest",
        "Miami at Notre Dame",
    ]
    assert matchups[6:8] == ["Ohio at Akron", "Division III Away 0 at Division III Home 0"]


def test_whole_season_scores_keep_the_week_nearest_today() -> None:
    """Unplayed games linger in old weeks, so they can't be what picks the week."""
    games = [
        cfbd_game("Thomas", "Erskine", week=2, start="2026-09-12T20:00:00.000Z", completed=False),
        cfbd_game("Tennessee", "Kentucky", 27, 20, week=9, start="2026-10-31T19:30:00.000Z"),
        cfbd_game("Ole Miss", "Georgia", week=10, start="2026-11-08T00:30:00.000Z"),
        cfbd_game("LSU", "Texas", week=11, start="2026-11-14T05:00:00.000Z"),
    ]

    result = compact_nationwide(games)

    assert result["games"] == ["Tennessee beat Kentucky 27-20, Sat Oct 31, Bryant-Denny Stadium, SEC"]
    assert result["totalGames"] == 4
    assert result["gamesOmitted"] == 3
    assert "Only week 9 is included" in result["note"]


def test_small_nationwide_result_stays_in_kickoff_order() -> None:
    games = [cfbd_game("Akron", "Ohio", start="2026-11-04T00:00:00.000Z"), cfbd_game("Tennessee", "Kentucky")]
    rankings = [cfbd_week(10, [cfbd_poll("AP Top 25")])]

    result = compact_nationwide(games, rankings)

    assert [line.split(",")[0] for line in result["games"]] == ["Ohio at Akron", "#8 Kentucky at #16 Tennessee"]
    assert result["gamesOmitted"] == 0
    assert "left out" not in result["note"]


def test_needs_rankings_only_for_nationwide_game_lists() -> None:
    games = pretty([cfbd_game("Alabama", "LSU")])

    assert needs_rankings("get_cfb_scores", {"week": 10}, games)
    assert not needs_rankings("get_cfb_scores", {"team": "Alabama"}, games)
    assert not needs_rankings("get_team_schedule", {"team": "Alabama"}, games)
    assert not needs_rankings("get_cfb_rankings", {}, games)
    assert not needs_rankings("get_cfb_scores", {}, "[]")
    assert not needs_rankings("get_cfb_scores", {}, "not json")


class ScriptedRouter:
    """Requests the given tool calls, one per iteration, then answers."""

    def __init__(self, calls: List[Tuple[str, Dict[str, Any]]]) -> None:
        self.calls = list(calls)
        self.messages: List[Dict[str, Any]] = []

    async def chat(self, messages: Any, mcp_tools: Any = None, provider: Any = None) -> LLMResponse:
        self.messages = messages
        if self.calls:
            name, arguments = self.calls.pop(0)
            return LLMResponse(
                content="",
                tool_calls=[ToolCallRequest(id="call_1", name=name, arguments=arguments)],
            )
        return LLMResponse(content="Done")


class ScriptedMCP:
    """Answers tool calls from a table, or raises when the answer is an exception."""

    def __init__(self, results: Dict[str, Any]) -> None:
        self.results = results
        self.calls: List[Tuple[str, Dict[str, Any]]] = []

    async def list_tools(self, fallback_to_defaults: bool = True) -> List[Dict[str, Any]]:
        return []

    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> str:
        self.calls.append((name, arguments))
        result = self.results[name]
        if isinstance(result, Exception):
            raise result
        return result


def build_orchestrator(router: ScriptedRouter, mcp: ScriptedMCP, tmp_path: Path, **kwargs: Any) -> AgentOrchestrator:
    return AgentOrchestrator(
        mcp_client=mcp,  # type: ignore[arg-type]
        llm_router=router,  # type: ignore[arg-type]
        skills_loader=SkillsLoader(skills_dir=tmp_path, extra_dirs=[]),
        **kwargs,
    )


def tool_messages(router: ScriptedRouter) -> List[str]:
    return [message["content"] for message in router.messages if message["role"] == "tool"]


@pytest.mark.asyncio
async def test_orchestrator_shows_the_model_compact_results_and_records_raw_ones(tmp_path: Path) -> None:
    raw_games = pretty(nationwide_week())
    router = ScriptedRouter([("get_cfb_scores", {"year": 2026, "week": 10})])
    mcp = ScriptedMCP(
        {
            "get_cfb_scores": raw_games,
            "get_cfb_rankings": pretty([cfbd_week(10, [cfbd_poll("AP Top 25")])]),
        }
    )
    events: List[Tuple[str, Dict[str, Any]]] = []

    response = await build_orchestrator(router, mcp, tmp_path).run(
        "What are this week's biggest games?",
        on_tool_event=lambda event, data: events.append((event, data)),
    )

    content = tool_messages(router)[0]
    assert len(content) <= 2400
    assert json.loads(content)["games"][0].startswith("#1 Miami at #4 Notre Dame,")
    assert [call.result for call in response.tool_calls] == [raw_games]

    # The rankings lookup is internal, so clients only hear about the model's own call.
    assert mcp.calls == [
        ("get_cfb_scores", {"year": 2026, "week": 10}),
        ("get_cfb_rankings", {"year": 2026}),
    ]
    assert [(event, data["name"]) for event, data in events] == [
        ("tool.start", "get_cfb_scores"),
        ("tool.end", "get_cfb_scores"),
    ]


@pytest.mark.asyncio
async def test_orchestrator_still_compacts_when_the_rankings_lookup_fails(tmp_path: Path) -> None:
    router = ScriptedRouter([("get_cfb_scores", {"week": 10})])
    mcp = ScriptedMCP(
        {"get_cfb_scores": pretty(nationwide_week()), "get_cfb_rankings": MCPClientError("upstream down")}
    )

    await build_orchestrator(router, mcp, tmp_path).run("Biggest games?")

    assert mcp.calls[1] == ("get_cfb_rankings", {})
    result = json.loads(tool_messages(router)[0])
    assert "No poll ranks were available" in result["note"]
    assert result["gamesOmitted"] > 0


@pytest.mark.asyncio
async def test_orchestrator_shrinks_later_results_within_a_turn(tmp_path: Path) -> None:
    """The context holds every tool result in a turn, not just the latest."""
    router = ScriptedRouter(
        [
            ("get_cfb_scores", {"week": 10}),
            ("get_cfb_scores", {"week": 10}),
            ("get_cfb_scores", {"week": 10}),
        ]
    )
    mcp = ScriptedMCP({"get_cfb_scores": pretty(nationwide_week()), "get_cfb_rankings": "[]"})

    await build_orchestrator(router, mcp, tmp_path).run("Biggest games?")

    first, second, third = [len(content) for content in tool_messages(router)]
    assert first <= 2400
    assert second <= 800
    assert third <= 800


@pytest.mark.asyncio
async def test_orchestrator_passes_tool_errors_to_the_model_as_they_are(tmp_path: Path) -> None:
    router = ScriptedRouter([("get_cfb_rankings", {"week": 10})])
    mcp = ScriptedMCP({"get_cfb_rankings": MCPClientError("upstream down")})

    response = await build_orchestrator(router, mcp, tmp_path).run("Top 4?")

    assert tool_messages(router) == [json.dumps({"error": "upstream down"})]
    assert response.tool_calls[0].result == json.dumps({"error": "upstream down"})
