# Postgres init scripts

Any `*.sql` or `*.sh` file placed in this directory is executed by the
`postgres:16` container **on first initialization only** (i.e. when the
`postgres_data` volume is empty).

The schema is owned by EF Core migrations (see `backend/src/TokenMiner.Infrastructure`),
so this directory should normally stay empty. Use it only for things migrations
cannot express, such as creating extra databases or roles.
