"""Alembic environment configuration for async SQLAlchemy.

Connects Alembic to the application's async engine and loads all
ORM models via the Base.metadata for autogenerate support.
"""

import asyncio
from logging.config import fileConfig

from alembic import context
from sqlalchemy.ext.asyncio import create_async_engine

# Load Alembic config
config = context.config

# Configure Python logging from alembic.ini
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Import all models for autogenerate detection
from app.database.base import Base
from app.database.models import *  # noqa: F401, F403

target_metadata = Base.metadata


def get_database_url() -> str:
    """Get the database URL from app settings or alembic config.

    Returns:
        str: Async PostgreSQL connection URL.
    """
    try:
        from app.core.config import get_settings
        return get_settings().database_url
    except Exception:
        return config.get_main_option("sqlalchemy.url") or ""


def run_migrations_offline() -> None:
    """Run migrations without an active DB connection (SQL script mode)."""
    url = get_database_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
    )

    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection):
    """Execute migrations with an active connection.

    Args:
        connection: Active database connection.
    """
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """Create async engine and run migrations."""
    url = get_database_url()
    connectable = create_async_engine(url, echo=False)

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()


def run_migrations_online() -> None:
    """Run migrations with an active database connection."""
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
