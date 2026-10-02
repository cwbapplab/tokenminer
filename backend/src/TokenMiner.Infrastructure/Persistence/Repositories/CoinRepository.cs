using Microsoft.EntityFrameworkCore;
using TokenMiner.Application.Mining.Abstractions;
using TokenMiner.Domain.Mining;

namespace TokenMiner.Infrastructure.Persistence.Repositories;

internal sealed class CoinRepository(AppDbContext dbContext) : ICoinRepository
{
    public Task<Coin?> GetByIdAsync(Guid id, CancellationToken cancellationToken) =>
        dbContext.Coins.FirstOrDefaultAsync(coin => coin.Id == id, cancellationToken);

    public Task<Coin?> GetByCodeAsync(string code, CancellationToken cancellationToken) =>
        dbContext.Coins.FirstOrDefaultAsync(coin => coin.Code == code, cancellationToken);

    public async Task<IReadOnlyList<Coin>> ListAsync(CancellationToken cancellationToken) =>
        await dbContext.Coins.OrderBy(coin => coin.Code).ToListAsync(cancellationToken);

    public void Add(Coin coin) => dbContext.Coins.Add(coin);

    public void AddPriceHistory(CoinPriceHistory pricePoint) => dbContext.CoinPriceHistory.Add(pricePoint);
}
