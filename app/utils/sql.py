import os
import pandas as pd
from sqlalchemy import create_engine, MetaData
from sqlalchemy.engine.base import Engine

MYSQL_USER = os.environ.get("MYSQL_USER", "user")
MYSQL_PASSWORD = os.environ.get("MYSQL_PASSWORD", "password")
MYSQL_SERVICE_HOST = os.environ.get("MYSQL_SERVICE_HOST", "localhost")
MYSQL_DATABASE = os.environ.get("MYSQL_DATABASE", "db")

db_connection_str = f"mysql+pymysql://{MYSQL_USER}:{MYSQL_PASSWORD}@{MYSQL_SERVICE_HOST}/{MYSQL_DATABASE}"
db_connection = create_engine(db_connection_str)


def df_to_sql(table_name: str, df: pd.DataFrame) -> None:
    df.to_sql(
        name=table_name,
        con=db_connection,
        if_exists="replace",
        index=False,
        chunksize=2000,
    )


def sql_to_df(table_name: str) -> pd.DataFrame:
    return pd.read_sql(sql=table_name, con=db_connection)


def drop_all_tables(db_connection: Engine) -> None:
    db_metadata = MetaData(db_connection)
    db_metadata.reflect()
    for table_name, table in db_metadata.tables.items():
        table.drop(db_connection)
