-- PaperWhisperer Hosted tier: storage extensions.
--
-- Runs once, from `/docker-entrypoint-initdb.d`, on an *empty* PGDATA. That
-- constraint is the postgres image's, not ours, and it is the reason the
-- store's own `initialize()` repeats both statements below: a database
-- restored from somebody else's dump, or a volume that already existed when
-- someone edited this file, will not come through here. Idempotent statements,
-- so whichever of the two paths runs second is a no-op.
--
-- Kept as a file rather than inline under Compose `configs:`. The inline form
-- needs Compose v2.23+ and is a hard error naming the field on anything older,
-- which turns a working deployment file into one that will not parse on a
-- machine with an older Compose -- a portability constraint with no benefit,
-- because the file is reviewable in a diff and a mount is understood by every
-- Compose version in use. See docker-compose.yml.
--
-- Ordering: plain `CREATE EXTENSION`, no ordering between them, and no
-- dependency on the app's schema. The store creates its own schema and tables
-- on first connect, so this file has exactly one job.

-- pgvector provides the `vector` type the VectorStore column is declared
-- with. Without it the table cannot be created at all, and the error names a
-- type the reader has never heard of rather than a missing feature -- which is
-- why the app image's healthcheck waits for *this* specifically and not merely
-- for `pg_isready`.
CREATE EXTENSION IF NOT EXISTS vector;

-- pg_trgm backs the trigram index on the lexical table. Provisioned now so a
-- later task can add fuzzy recall without a migration against a live database,
-- and because creating an extension is a superuser action an application role
-- will not have later.
CREATE EXTENSION IF NOT EXISTS pg_trgm;
