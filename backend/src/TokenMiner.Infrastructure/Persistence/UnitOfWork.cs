using TokenMiner.Application.Common.Abstractions;

namespace TokenMiner.Infrastructure.Persistence;

internal sealed class UnitOfWork(AppDbContext dbContext) : IUnitOfWork
{
    public Task<int> SaveChangesAsync(CancellationToken cancellationToken) =>
        dbContext.SaveChangesAsync(cancellationToken);

    public IExecutionStrategy CreateExecutionStrategy() =>
        new EfExecutionStrategy(dbContext.Database.CreateExecutionStrategy());
}
