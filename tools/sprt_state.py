#!/usr/bin/env python3

import argparse
import json
import os
import re
import sys
from pathlib import Path


def load(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def matchup(data: dict) -> tuple[str, dict]:
    stats = data.get("stats", {})
    if len(stats) != 1:
        raise ValueError("expected exactly one matchup in fastchess stats")
    return next(iter(stats.items()))


def games_played(data: dict) -> int:
    if not data.get("stats"):
        return 0
    _, stats = matchup(data)
    return int(stats["wins"]) + int(stats["losses"]) + int(stats["draws"])


def engine(data: dict, index: int) -> dict:
    engines = data.get("engines", [])
    if len(engines) != 2:
        raise ValueError("expected exactly two engines")
    return engines[index]


def summary(data: dict) -> str:
    if not data.get("stats"):
        return "games=0"
    name, stats = matchup(data)
    penta = [
        stats.get("penta_LL", 0),
        stats.get("penta_LD", 0),
        stats.get("penta_WL", 0) + stats.get("penta_DD", 0),
        stats.get("penta_WD", 0),
        stats.get("penta_WW", 0),
    ]
    return (
        f"matchup={name} games={games_played(data)} "
        f"wins={stats['wins']} losses={stats['losses']} draws={stats['draws']} "
        f"ptnml={penta}"
    )


def matching_games(path: Path, white: str, black: str) -> list[str]:
    text = path.read_text(encoding="utf-8", errors="replace")
    chunks = re.split(r"(?=^\[Event )", text, flags=re.MULTILINE)
    games = []
    expected = {white, black}
    for chunk in chunks:
        if not chunk.startswith("[Event "):
            continue
        players = set(re.findall(r'^\[(?:White|Black) "([^"]+)"\]$', chunk, flags=re.MULTILINE))
        if players == expected:
            games.append(chunk)
    return games


def write_json(path: Path, data: dict) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=4)
        handle.write("\n")
    os.replace(temp, path)


def adopt(args: argparse.Namespace) -> None:
    source = load(args.config)
    cand = engine(source, 0)
    base = engine(source, 1)
    count = games_played(source)

    games = matching_games(args.legacy_pgn, cand["name"], base["name"])
    if len(games) < count:
        raise ValueError(f"legacy PGN has {len(games)} matching games, config expects {count}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    selected = games[-count:] if count else []
    args.pgn.write_text("".join(selected), encoding="utf-8")

    cand["cmd"] = str(args.candidate.resolve())
    base["cmd"] = str(args.baseline.resolve())
    source["pgn"]["file"] = str(args.pgn.resolve())
    source["pgn"]["append_file"] = True
    source["config_name"] = str(args.output.resolve())
    write_json(args.output, source)
    print(summary(source))


def validate(args: argparse.Namespace) -> None:
    data = load(args.config)
    for index in range(2):
        path = Path(engine(data, index)["cmd"])
        if not path.is_file() or not os.access(path, os.X_OK):
            raise ValueError(f"engine is missing or not executable: {path}")
    opening = Path(data["opening"]["file"])
    if not opening.is_file():
        raise ValueError(f"opening book is missing: {opening}")
    print(summary(data))


def get_value(args: argparse.Namespace) -> None:
    data = load(args.config)
    values = {
        "candidate": engine(data, 0)["cmd"],
        "baseline": engine(data, 1)["cmd"],
        "pgn": data["pgn"]["file"],
        "games": games_played(data),
    }
    print(values[args.field])


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser()
    commands = root.add_subparsers(dest="command", required=True)

    adopt_cmd = commands.add_parser("adopt")
    adopt_cmd.add_argument("config", type=Path)
    adopt_cmd.add_argument("legacy_pgn", type=Path)
    adopt_cmd.add_argument("output", type=Path)
    adopt_cmd.add_argument("pgn", type=Path)
    adopt_cmd.add_argument("candidate", type=Path)
    adopt_cmd.add_argument("baseline", type=Path)
    adopt_cmd.set_defaults(run=adopt)

    validate_cmd = commands.add_parser("validate")
    validate_cmd.add_argument("config", type=Path)
    validate_cmd.set_defaults(run=validate)

    get_cmd = commands.add_parser("get")
    get_cmd.add_argument("config", type=Path)
    get_cmd.add_argument("field", choices=("candidate", "baseline", "pgn", "games"))
    get_cmd.set_defaults(run=get_value)
    return root


def main() -> int:
    args = parser().parse_args()
    try:
        args.run(args)
    except (KeyError, OSError, ValueError, json.JSONDecodeError) as error:
        print(f"sprt state error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
