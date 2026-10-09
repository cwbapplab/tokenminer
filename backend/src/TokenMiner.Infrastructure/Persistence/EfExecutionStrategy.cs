using Microsoft.EntityFrameworkCore;
using TokenMiner.Application.Common.Abstractions;
using EfStrategy = Microsoft.EntityFrameworkCore.Storage.IExecutionStrategy;

namespace TokenMiner.Infrastructure.Persistence;

/// <summary>Adapts EF Core's execution strategy to the application-layer port.</summary>
internal sealed class EfExecutionStrategy(EfStrategy strategy) : IExecutionStrategy
{
    public Task<TResult> ExecuteAsync<TResult>(Func<Task<TResult>> operation) =>
        strategy.ExecuteAsync(operation);

    public Task ExecuteAsync(Func<Task> operation) =>
        strategy.ExecuteAsync(operation);
}
