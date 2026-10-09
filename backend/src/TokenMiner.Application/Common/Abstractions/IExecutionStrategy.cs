namespace TokenMiner.Application.Common.Abstractions;

/// <summary>
/// Runs a unit of work with whatever retry policy the persistence layer is configured with.
/// Mirrors EF Core's <c>IExecutionStrategy</c> so the application layer can opt a handler into
/// it, or provide a no-op when the test double has no retry policy to honour.
/// </summary>
public interface IExecutionStrategy
{
    Task<TResult> ExecuteAsync<TResult>(Func<Task<TResult>> operation);

    Task ExecuteAsync(Func<Task> operation);
}
