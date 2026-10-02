namespace TokenMiner.Application.Common.Abstractions;

/// <summary>
/// Commits all changes staged through the repositories in the current scope.
/// Shared by every feature area so handlers can commit their work atomically.
/// </summary>
public interface IUnitOfWork
{
    Task<int> SaveChangesAsync(CancellationToken cancellationToken);
}
