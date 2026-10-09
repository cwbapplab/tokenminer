namespace TokenMiner.Application.Common.Abstractions;

/// <summary>
/// Commits all changes staged through the repositories in the current scope.
/// Shared by every feature area so handlers can commit their work atomically.
/// </summary>
public interface IUnitOfWork
{
    Task<int> SaveChangesAsync(CancellationToken cancellationToken);

    /// <summary>
    /// Wraps work that must be retried as a whole if the commit fails on a transient fault.
    /// EF Core forbids starting such a retry from inside a transaction it already opened, so a
    /// handler that opens its own transaction (or issues conditional writes it cannot replay)
    /// has to run through the strategy the store is configured with.
    /// </summary>
    IExecutionStrategy CreateExecutionStrategy();
}
