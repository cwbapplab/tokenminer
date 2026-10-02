using Microsoft.EntityFrameworkCore;
using TokenMiner.Application.Mining.Abstractions;
using TokenMiner.Domain.Mining;

namespace TokenMiner.Infrastructure.Persistence.Repositories;

internal sealed class PoolRepository(AppDbContext dbContext) : IPoolRepository
{
    public Task<Pool?> GetByIdAsync(Guid id, CancellationToken cancellationToken) =>
        dbContext.Pools.FirstOrDefaultAsync(pool => pool.Id == id, cancellationToken);

    public Task<Pool?> GetBySystemPoolIdAsync(string systemPoolId, CancellationToken cancellationToken) =>
        dbContext.Pools.FirstOrDefaultAsync(pool => pool.SystemPoolId == systemPoolId, cancellationToken);

    public async Task<IReadOnlyList<Pool>> ListAsync(CancellationToken cancellationToken) =>
        await dbContext.Pools.OrderBy(pool => pool.Name).ToListAsync(cancellationToken);

    public Task<PoolDetails?> GetDetailsAsync(Guid poolId, CancellationToken cancellationToken) =>
        dbContext.PoolDetails.FirstOrDefaultAsync(details => details.PoolId == poolId, cancellationToken);

    public async Task<IReadOnlyList<PoolDetails>> ListDetailsAsync(CancellationToken cancellationToken) =>
        await dbContext.PoolDetails.ToListAsync(cancellationToken);

    public async Task<IReadOnlyList<PoolCoin>> ListPoolCoinsAsync(CancellationToken cancellationToken) =>
        await dbContext.PoolCoins.ToListAsync(cancellationToken);

    public async Task<IReadOnlyList<PoolCoin>> ListPoolCoinsAsync(Guid poolId, CancellationToken cancellationToken) =>
        await dbContext.PoolCoins
            .Where(poolCoin => poolCoin.PoolId == poolId)
            .ToListAsync(cancellationToken);

    public Task<bool> SupportsCoinAsync(Guid poolId, Guid coinId, CancellationToken cancellationToken) =>
        dbContext.PoolCoins.AnyAsync(
            poolCoin => poolCoin.PoolId == poolId && poolCoin.CoinId == coinId,
            cancellationToken);

    public void Add(Pool pool) => dbContext.Pools.Add(pool);
    public void AddDetails(PoolDetails details) => dbContext.PoolDetails.Add(details);

    public void AddPoolCoin(PoolCoin poolCoin) => dbContext.PoolCoins.Add(poolCoin);

    public void RemovePoolCoins(IEnumerable<PoolCoin> poolCoins) => dbContext.PoolCoins.RemoveRange(poolCoins);
}
