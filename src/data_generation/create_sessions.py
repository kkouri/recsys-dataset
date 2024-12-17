import polars as pl
from beartype import beartype

@beartype
def create_sessions(df: pl.DataFrame):
    """
    Form sessions by splitting the events of a session into sub-sessions based on time between the events
    If the time between events is more than 30 minutes, it is considered a new session

    timestamps are expected to be in seconds
    """

    sessions_df = (
        df
        .sort(["session", "ts"], descending=[False, False])
        .with_columns(
            previous_session=pl.col("session").shift(1),
            previous_ts=pl.col("ts").shift(1),
        )
        .with_columns(
            time_between_sec=(
                pl.when(pl.col("previous_session") == pl.col("session"))
                .then((pl.col("ts") - pl.col("previous_ts")))
                .otherwise(None)
            )
        )
        .with_columns(
            is_session_boundary=(pl.col("time_between_sec").is_null() | (pl.col("time_between_sec") >= 30 * 60))
        )
        .with_columns(
            session=pl.col("is_session_boundary").cum_sum().cast(pl.UInt32),
        )
        .drop(["previous_session", "previous_ts", "time_between_sec", "is_session_boundary"])
        # Count the amount of events in a session
        .with_columns(events_count=(pl.col("ts").rle_id().max().over("session") + 1))
        # Filter out sessions with only 1 event
        .filter(pl.col("events_count") > 1)
        .drop("events_count")
        # Make session ids to be consistent
        .with_columns(session=pl.col("session").rle_id()+1)
    )

    return sessions_df
