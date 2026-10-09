using Microsoft.EntityFrameworkCore;
using TokenMiner.Application.Authentication.Abstractions;
using TokenMiner.Domain.Users;

namespace TokenMiner.Infrastructure.Persistence.Repositories;

internal sealed class RefreshTokenStore(AppDbContext dbContext) : IRefreshTokenStore
{
    public void Add(RefreshToken token) => dbContext.RefreshTokens.Add(token);

    public Task<RefreshToken?> GetByHashAsync(string tokenHash, CancellationToken cancellationToken) =>
        dbContext.RefreshTokens.FirstOrDefaultAsync(token => token.TokenHash == tokenHash, cancellationToken);

    public async Task<bool> TryRotateAsync(
        Guid tokenId,
        string reason,
        DateTimeOffset now,
        Guid replacedByTokenId,
        CancellationToken cancellationToken)
    {
        // The rotation writes the claim with ExecuteUpdate (a single statement, invisible to the
        // change tracker) and the new token through SaveChanges. Those are two different paths to
        // the same database, so they only land together inside one explicit transaction.
        await using var transaction =
            await dbContext.Database.BeginTransactionAsync(cancellationToken);

        // A conditional UPDATE, not a read-then-write: the `RevokedAt is null` predicate is part of
        // the statement, so Postgres serializes two concurrent rotations of the same token and
        // exactly one of them sees a row to change. The other reports zero rows and loses.
        var affected = await dbContext.RefreshTokens
            .Where(token => token.Id == tokenId && token.RevokedAt == null)
            .ExecuteUpdateAsync(
                setters => setters
                    .SetProperty(token => token.RevokedAt, now)
                    .SetProperty(token => token.RevokeReason, reason)
                    .SetProperty(token => token.ReplacedByTokenId, replacedByTokenId),
                cancellationToken);

        if (affected == 0)
        {
            // The loser must still be able to stage its replay record, which the handler saves
            // after the transaction closes, so only this attempt is rolled back.
            await transaction.RollbackAsync(cancellationToken);
            return false;
        }

        await dbContext.SaveChangesAsync(cancellationToken);
        await transaction.CommitAsync(cancellationToken);
        return true;
    }

    public async Task<int> RevokeSessionChainAsync(
        Guid tokenId,
        string reason,
        DateTimeOffset now,
        CancellationToken cancellationToken)
    {
        // Walk the rotation chain in the database. `chain` collects the token and everything
        // descended from it through replaced_by_token_id; the UPDATE then touches only the
        // members that are still active. Intermediate nodes are normally already revoked (each
        // one was rotated on), so the walk must not stop at them — the live token sits at the end.
        // A depth guard bounds the walk in case of impossible cyclic data; parameters are passed
        // positionally so the same value is not bound under several names.
        const string sql = """
            WITH RECURSIVE chain AS (
                SELECT id, replaced_by_token_id, 0 AS depth
                FROM refresh_tokens
                WHERE id = {0}
                UNION ALL
                SELECT t.id, t.replaced_by_token_id, c.depth + 1
                FROM refresh_tokens t
                JOIN chain c ON t.id = c.replaced_by_token_id
                WHERE c.depth < 1000
            )
            UPDATE refresh_tokens
            SET revoked_at = {1}, revoke_reason = {2}
            WHERE revoked_at IS NULL
              AND id IN (SELECT id FROM chain)
              AND id <> {0}
            """;

        return await dbContext.Database.ExecuteSqlRawAsync(
            sql,
            [tokenId, now, reason],
            cancellationToken);
    }
}
