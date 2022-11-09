import datetime
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
import pandas as pd
import numpy as np
from dataclasses import dataclass
from enum import Enum
import heapq


def read_excel(path) -> dict[str, pd.DataFrame]:
    def time_parser(time: datetime.time | datetime.datetime | str) -> pd.Timedelta:
        """Convert excel [h]h:mm:ss to pd.Timedelta.
        For data < 24 hours, it will be parsed as datetime.time, otherwise datetime.datetime.
        For NA, it will be parsed as NaN.
        """

        if isinstance(time, datetime.datetime):
            # workaround: pd parse will reduce the datetime by 1 day, so we add 1 day to it
            return pd.Timedelta(
                time + datetime.timedelta(days=1) - datetime.datetime(1900, 1, 1)
            )
        elif isinstance(time, datetime.time):
            return pd.Timedelta(
                datetime.datetime(1900, 1, 1, time.hour, time.minute, time.second)
                - datetime.datetime(1900, 1, 1)
            )
        elif isinstance(time, str) and time == "NA":  # NaT is stored as NA
            return np.nan
        else:
            raise TypeError(
                f'Expected datetime.time or datetime.datetime or string "NA", got {type(time)}'
            )

    return pd.read_excel(
        path, sheet_name=None, converters={"time": time_parser, "time/cp": time_parser}
    )


def flatten_all_buildings(dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Flatten the all buildings data, so that each row also contains the name of the building"""

    all_buildings_df_lists = []
    for name, df in dfs.items():
        df["name"] = name
        all_buildings_df_lists.append(df)
    flattened_df = pd.concat(all_buildings_df_lists).reset_index(drop=True)
    return flattened_df


def add_cp_delta_column(dfs: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Add a column to each dataframe that contains the difference in cp per level up.
    This needs to be applied before flattening.
    """

    for _name, df in dfs.items():
        df["cp_delta"] = df["cp"].diff()
        df.loc[0, ["cp_delta"]] = df.loc[0]["cp"]  # replace first row's NaN to its cp
    return dfs


def build_name_level_to_index_mapping(
    flattened_df: pd.DataFrame,
) -> dict[tuple[str, int], int]:
    """Build a hashmap of (name + level) to df's index"""

    return {
        (name, level): index
        for index, (name, level) in flattened_df[["name", "level"]].iterrows()
    }


def build_building_prerequisites() -> dict[str, list[tuple[str, int]]]:
    """Build a hashmap of building name to a list of building names that are required to build it.
    NOTE: Buildings without prerequsites are not included in the hashmap. So if you use the existence of a
    key in the hashmap to check whether a building has prerequsites, a typo will still return negative
    """

    prerequisites: dict[str, list[tuple[str, int]]] = {
        "Sawmill": [("Woodcutter", 10)],
        "Brickyard": [("Clay Pit", 10)],
        "Iron Foundry": [("Iron Mine", 10)],
        "Grain Mill": [("Cropland", 10)],
        "Bakery": [("Grain Mill", 5)],
        "Smithy": [("Main Building", 5), ("Academy", 1)],
        "Tournament Square": [("Rally Point", 15)],
        "Marketplace": [("Main Building", 3), ("Warehouse", 1), ("Granary", 1)],
        "Barracks": [("Main Building", 3)],
        "Stable": [("Smithy", 3), ("Academy", 5)],
        "Workshop": [("Main Building", 5), ("Academy", 10)],
        "Academy": [("Main Building", 3), ("Barracks", 3)],
        "Townhall": [("Main Building", 10), ("Academy", 10)],
        "Residence": [("Main Building", 5)],
        "Palace": [("Embassy", 1), ("Main Building", 5)],
        "Treasury": [("Main Building", 10)],
        "Trade Office": [("Marketplace", 20), ("Stable", 10)],
        "Great Barracks": [("Barracks", 20)],
        "Great Stable": [("Stable", 20)],
        "Stonemason's Lodge": [("Main Building", 5)],
        "Brewery": [("Granary", 20), ("Rally Point", 10)],
        "Trapper": [("Rally Point", 1)],
        "Hero's Mansion": [("Rally Point", 1), ("Main Building", 3)],
        "Great Warehouse": [("Main Building", 10)],
        "Great Granary": [("Main Building", 10)],
        "Horse Drinking Pool": [("Stable", 20), ("Rally Point", 10)],
        "Hospital": [("Academy", 15), ("Main Building", 10)],
    }
    return prerequisites


class ValueType(Enum):
    """Enum's value is the dataframe's column name"""

    RES = "total"
    TIME = "time"


@dataclass
class Group:
    """A group of buildings that needs to be built in sequence due to prerequisites.
    Each group contains an effective efficiency for ranking (lower value = more efficient),
    and an ordered list of buildings which are stored as df's index, and a set containing
    all the dependent buildings' names.
    """

    efficiency: float | pd.Timedelta
    order: list[int]
    dependency: set[str]

    def __init__(
        self,
        efficiency: float | pd.Timedelta,
        order: list[int],
        flattened_df: pd.DataFrame,
    ) -> None:
        self.efficiency = efficiency
        self.order = order
        self._update_dependency(flattened_df)

    def _update_dependency(self, flattened_df: pd.DataFrame):
        """Update the dependency set based on the order list"""

        self.dependency = set()
        for index in self.order:
            self.dependency.add(flattened_df.loc[index]["name"])

    def __gt__(self, other) -> bool:
        return self.efficiency > other.efficiency

    def __eq__(self, other) -> bool:
        return self.efficiency == other.efficiency


@dataclass
class GroupPriorityQueue:
    """A priority queue of groups, with the highest efficiency group at the top.
    It contains a dependency set that is the union of all the groups' dependency sets
    for quick look up. The priority queue is implemented as min heap using the group's
    efficiency so the most efficient group is placed at the top. value_type determine
    whether it is optimising time or resources
    """

    dependency: set[str]
    p_queue: list[Group]
    value_type: ValueType

    def peek(self):
        """Return the min efficiency in the priority queue"""

        return self.p_queue[0].efficiency

    def update_priorty_queue(
        self,
        flattened_df: pd.DataFrame,
        updated_buildings: set[str],
        current_level: dict[str, int],
    ) -> None:
        """Update the priority queue by rebuilding it one group at a time.
        If the group is dependent on any of the updated buildings, new build order and
        efficiency will be found using the updated current level and added to the new queue.
        Otherwise, it will be added directly to the new queue.
        """

        if self.dependency.isdisjoint(updated_buildings):
            return
        new_p_queue: list[Group] = []
        for group in self.p_queue:
            # no update is required for this group
            if group.dependency.isdisjoint(updated_buildings):
                heapq.heappush(new_p_queue, group)
                continue

            new_order = _find_build_order(
                flattened_df, current_level, index=group.order[-1]
            )
            if len(new_order) == 0:  # all buildings are already met
                continue

            eff_efficiency = _calculate_effective_efficiency(
                flattened_df, new_order, value_type=self.value_type
            )
            new_group = Group(eff_efficiency, new_order, flattened_df)
            heapq.heappush(new_p_queue, new_group)
        self.p_queue = new_p_queue
        self._update_dependency()

    def pop(self) -> Group:
        """Pop the most efficient group and update the dependency set"""

        popped = heapq.heappop(self.p_queue)
        self._update_dependency()
        return popped

    def push(
        self,
        group: Group,
    ) -> None:
        """Push a new group to the priority queue and update the dependency set"""

        heapq.heappush(self.p_queue, group)
        self._update_dependency()

    def _update_dependency(self) -> None:
        """Update the whole dependency set to match with the current priority queue"""

        self.dependency = set()
        for group in self.p_queue:
            self.dependency.update(group.dependency)

    def __len__(self) -> int:
        return len(self.p_queue)


def construct_current_level_only_core(flattened_df: pd.DataFrame) -> dict[str, int]:
    """Construct a dictionary of current level of all buildings.
    Non-core buildings are disabled by maxing their levels"""

    current_level: dict[str, int] = {name: 0 for name in flattened_df["name"].unique()}
    current_level["Main Building"] = 1
    current_level["Warehouse"] = 100
    current_level["Granary"] = 100
    current_level["Iron Foundry"] = 100
    current_level["Brickyard"] = 100
    current_level["Sawmill"] = 100
    current_level["Bakery"] = 100
    current_level["Grain Mill"] = 100
    current_level["Stonemason's Lodge"] = 100
    current_level["Brewery"] = 100
    current_level["Trapper"] = 100
    current_level["Great Warehouse"] = 100
    current_level["Great Granary"] = 100
    current_level["Horse Drinking Pool"] = 100

    return current_level


def construct_current_level_all_empty(flattened_df: pd.DataFrame) -> dict[str, int]:
    """Construct a dictionary of current level of all buildings"""

    current_level: dict[str, int] = {name: 0 for name in flattened_df["name"].unique()}
    current_level["Main Building"] = 1

    return current_level


def sort_by_efficiency(
    flattened_df: pd.DataFrame, current_level: dict[str, int], value_type: ValueType
) -> pd.DataFrame:
    """Sort the df by the efficiency (by time or by res depending on value_type).
    current_level determines the current level of all buildings. Buildings can be ignored
    by setting current_level to max level.

    It starts by iterating the df row by row. If the current row requires a group, it will
    be pushed to a priority queue and the row will be incremented. The current row will be ignored
    if it is already satisfied. If the current row does not require a group and its efficiency is
    better then the top of the priority queue (or empty), it will be added to the sorted df directly.
    Otherwise, the top of the priority queue will be popped and added to the sorted df and the current
    row will not be incremented to retest the current row.
    """

    global name_level_to_index_mapping
    if value_type is ValueType.RES:
        column_key = "res/cp"
    elif value_type is ValueType.TIME:
        column_key = "time/cp"
    else:
        raise ValueError("Invalid value type")

    sorted_df: pd.DataFrame = flattened_df.sort_values(
        by=[column_key], ascending=True
    ).copy()
    sort_by_res_df: pd.DataFrame = pd.DataFrame(columns=sorted_df.columns)
    sort_by_res_df["eff_efficiency"] = sort_by_res_df[column_key]
    group_priority_queue: GroupPriorityQueue = GroupPriorityQueue(set(), [], value_type)
    # HACK for some reasons, pd.concat of an empty df will not work so we still use append
    row_num = 0
    rows = sorted_df.iterrows()
    index, _row = next(rows)
    try:
        while True:
            order = _find_build_order(flattened_df, current_level, index)
            if len(order) == 0:  # already met
                row_num += 1
                index, _row = next(rows)
                continue
            eff_efficiency = _calculate_effective_efficiency(
                flattened_df, order, value_type
            )
            if len(order) > 1:  # this is a group
                group_priority_queue.push(Group(eff_efficiency, order, flattened_df))
                row_num += 1
                index, _row = next(rows)
                continue

            if (
                len(group_priority_queue) == 0
                or eff_efficiency < group_priority_queue.peek()
            ):
                new_rows, updated_buildings = _create_build_order_df(
                    flattened_df, current_level, order
                )
                row_num += 1
                index, _row = next(rows)
                new_rows["eff_efficiency"] = eff_efficiency
            else:
                popped_group = group_priority_queue.pop()

                new_rows, updated_buildings = _create_build_order_df(
                    flattened_df, current_level, popped_group.order
                )
                new_rows["eff_efficiency"] = popped_group.efficiency

            group_priority_queue.update_priorty_queue(
                flattened_df, updated_buildings, current_level
            )
            sort_by_res_df = sort_by_res_df.append(new_rows, ignore_index=True)
    except StopIteration:
        return sort_by_res_df


def _create_build_order_df(
    flattened_df: pd.DataFrame, current_level: dict[str, int], order: list[int]
) -> tuple[pd.DataFrame, set[str]]:
    """Create a df based on the order and update current level.
    Return the new rows and the updated buildings set.
    """

    updated_buildings: set[int] = set()
    for index in order:
        name: str = flattened_df.loc[index, "name"]
        level: int = flattened_df.loc[index, "level"]
        updated_buildings.add(name)
        current_level[name] = level
    return flattened_df.loc[order].copy(), updated_buildings


def _find_build_order(
    flattened_df: pd.DataFrame,
    current_level: dict[str, int],
    index: int,
) -> list[int]:
    global name_level_to_index_mapping, building_prereq
    name: str = flattened_df.loc[index]["name"]
    level: int = flattened_df.loc[index]["level"]
    # all buildings (including itself) has been added already
    if level <= current_level[name]:
        return []

    build_order: list[int] = []
    # check if there are missing levels
    if current_level[name] == 0 or level > current_level[name] + 1:
        # check if ther are potentially unmet prereq
        if name in building_prereq and current_level[name] == 0:
            added_buildings: set[int] = set()
            for prereq_name, prereq_level in building_prereq[name]:
                prereq = _find_build_order(
                    flattened_df,
                    current_level,
                    name_level_to_index_mapping[(prereq_name, prereq_level)],
                )
                for prereq_index in prereq:
                    if prereq_index not in added_buildings:
                        build_order.append(prereq_index)
                        added_buildings.add(prereq_index)
        # add the missing levels
        for missing_level in range(current_level[name] + 1, level):
            build_order.append(name_level_to_index_mapping[(name, missing_level)])

    build_order.append(index)  # add itself
    return build_order


def _calculate_effective_efficiency(
    flattened_df: pd.DataFrame, order: list[int], value_type: ValueType
) -> float | pd.Timedelta:
    """Calculate the effective efficiency by adding up the cp delta of each building and
    divide by the total res/time.
    Return np.nan or NaT depending on the value_type if the total cp delta is zero.
    """

    cp_sum: int = 0
    if value_type is value_type.RES:
        res_or_time_sum: int = 0
    elif value_type is value_type.TIME:
        res_or_time_sum: pd.Timedelta = pd.Timedelta(0)
    else:
        raise ValueError(f"Invalid value_type: {value_type}")
    for index in order:
        cp_sum += flattened_df.loc[index]["cp_delta"]
        res_or_time_sum += flattened_df.loc[index][value_type.value]
    if cp_sum == 0:
        return np.nan if value_type is ValueType.RES else pd.Timedelta(np.nan)
    # print(res_or_time_sum, cp_sum, res_or_time_sum / cp_sum)
    return res_or_time_sum / cp_sum


if __name__ == "__main__":
    all_buildings_df = read_excel("data/travian_data_raw.xlsx")
    all_buildings_df = add_cp_delta_column(all_buildings_df)
    flattened_df = flatten_all_buildings(all_buildings_df)

    name_level_to_index_mapping: dict[
        tuple[str, int], int
    ] = build_name_level_to_index_mapping(flattened_df)
    building_prereq: dict[str, list[tuple[str, int]]] = build_building_prerequisites()

    current_level = construct_current_level_only_core(flattened_df)
    # current_level = construct_current_level_all_empty(flattened_df)
    sort_by_res_df: pd.DataFrame = sort_by_efficiency(
        flattened_df, current_level, ValueType.RES
    )

    sort_by_res_df.to_csv("pqueue_sort_by_res.csv", index=False)
