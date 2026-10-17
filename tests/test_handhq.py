"""HandHQ conversion on a tiny hand-written folder."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from pokerbot.data.handhq import convert_folder

SAMPLE = """[1]
variant = 'NT'
ante_trimming_status = false
antes = [0, 0, 0]
blinds_or_straddles = [1, 2, 0]
min_bet = 2
starting_stacks = [247.15, 237.05, 47.90]
actions = ['d dh p1 ????', 'd dh p2 ????', 'd dh p3 ????', 'p3 cc', 'p1 cc', 'p2 cc', 'd db 6hTdKd', 'p1 cc', 'p2 cbr 4', 'p3 f', 'p1 f']
venue = 'PokerStars'
time = 00:00:00
day = 1
month = 7
year = 2009
hand = 59937793530
seats = [3, 5, 6]
seat_count = 6
table = 'uJ9juaXXg9yIcSU8amxaXQ'
players = ['aaa', 'bbb', 'ccc']
winnings = [0, 6, 0]
currency_symbol = '$'
time_zone_abbreviation = 'ET'

[2]
variant = 'NT'
ante_trimming_status = false
antes = [0, 0]
blinds_or_straddles = [1, 2]
min_bet = 2
starting_stacks = [100, 100]
actions = ['d dh p1 ????', 'd dh p2 ????', 'p2 f']
venue = 'PokerStars'
time = 00:00:05
day = 2
month = 7
year = 2009
hand = 59937793531
seats = [1, 2]
seat_count = 2
table = 't2'
players = ['bbb', 'aaa']
winnings = [3, 0]
currency_symbol = '$'
time_zone_abbreviation = 'ET'
"""


def test_convert_folder(tmp_path: Path) -> None:
    src = tmp_path / "PS-2009-07-01_2009-07-23_200NLH_OBFU" / "1"
    src.mkdir(parents=True)
    (src / "a.phhs").write_text(SAMPLE, encoding="utf-8")
    (src / "broken.phhs").write_text("[1\nnot toml", encoding="utf-8")

    result = convert_folder(src.parent, tmp_path / "out")
    assert result.hands == 2 and result.bad_files == ["1/broken.phhs"]

    df = pl.read_parquet(tmp_path / "out" / "PS-200NL.parquet")
    first = df.row(0, named=True)
    assert first["site"] == "PS" and first["stake"] == 200
    assert first["players"] == ["aaa", "bbb", "ccc"]
    assert first["starting_stacks"] == [247.15, 237.05, 47.9]
    assert first["big_blind"] == 2.0
    assert first["actions"][-1] == "p1 f"
    assert df["date"].to_list() == ["2009-07-01", "2009-07-02"]
