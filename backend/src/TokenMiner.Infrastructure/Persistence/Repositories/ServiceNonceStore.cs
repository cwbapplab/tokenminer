using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.Logging;
using Npgsql;
using TokenMiner.Application.Mining.Abstractions;

namespace TokenMiner.Infrastructure.Persistence.Repositories;

/// <summary>
/// Replay protection backed by a unique constraint: the first use of a nonce inserts a row,
/// and any later use violates the primary key.
/// </summary>
internal sealed class ServiceNonceStore(
    AppDbContext dbContext,
    TimeProvider timeProvider,
    ILogger<ServiceNonceStore> logger) : IServiceNonceStore
{
    public async Task<bool> TryRegisterAsync(
        string serviceId,
        string nonce,
        DateTimeOffset expiresAt,
        CancellationToken cancellationToken)
    {
        var now = timeProvider.GetUtcNow();

        // Opportunistic cleanup keeps the table bounded without a dedicated job.
        await dbContext.ServiceRequestNonces
            .Where(entry => entry.ExpiresAt < now)
            .ExecuteDeleteAsync(cancellationToken);

        dbContext.ServiceRequestNonces.Add(new ServiceRequestNonce(serviceId, nonce, expiresAt, now));

        try
        {
            await dbContext.SaveChangesAsync(cancellationToken);
            return true;
        }
        catch (DbUpdateException exception) when (IsUniqueViolation(exception))
        {
            // Detach the rejected insert so the request's context stays usable.
            var added = dbContext.ChangeTracker
                .Entries<ServiceRequestNonce>()
                .Where(entry => entry.State == EntityState.Added)
                .ToList();

            foreach (var entry in added)
            {
                entry.State = EntityState.Detached;
            }

            logger.LogWarning("Rejected a replayed nonce from service {ServiceId}.", serviceId);
            return false;
        }
    }

    private static bool IsUniqueViolation(DbUpdateException exception) =>
        exception.InnerException is PostgresException { SqlState: PostgresErrorCodes.UniqueViolation };
}
