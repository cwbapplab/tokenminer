using Microsoft.EntityFrameworkCore;
using TokenMiner.Application.Treasury.Abstractions;
using TokenMiner.Domain.Treasury;
using TokenMiner.Domain.Treasury.Enums;

namespace TokenMiner.Infrastructure.Persistence.Repositories;

internal sealed class ConversionRepository(AppDbContext dbContext) : IConversionRepository
{
    public void AddProvider(ConversionProvider provider) => dbContext.ConversionProviders.Add(provider);

    public void AddRoute(ConversionRoute route) => dbContext.ConversionRoutes.Add(route);

    public void AddTransaction(ConversionTransaction transaction) =>
        dbContext.ConversionTransactions.Add(transaction);

    public Task<ConversionProvider?> GetProviderByIdAsync(Guid id, CancellationToken cancellationToken) =>
        dbContext.ConversionProviders.FirstOrDefaultAsync(provider => provider.Id == id, cancellationToken);

    public Task<ConversionProvider?> GetProviderByNameAsync(string name, CancellationToken cancellationToken) =>
        dbContext.ConversionProviders.FirstOrDefaultAsync(provider => provider.Name == name, cancellationToken);

    public async Task<IReadOnlyList<ConversionProvider>> ListProvidersAsync(CancellationToken cancellationToken) =>
        await dbContext.ConversionProviders.OrderBy(provider => provider.Name).ToListAsync(cancellationToken);

    public Task<ConversionRoute?> GetRouteByIdAsync(Guid id, CancellationToken cancellationToken) =>
        dbContext.ConversionRoutes.FirstOrDefaultAsync(route => route.Id == id, cancellationToken);

    public async Task<IReadOnlyList<ConversionRoute>> ListRoutesAsync(CancellationToken cancellationToken) =>
        await dbContext.ConversionRoutes.ToListAsync(cancellationToken);

    public Task<ConversionTransaction?> GetTransactionByIdAsync(Guid id, CancellationToken cancellationToken) =>
        dbContext.ConversionTransactions.FirstOrDefaultAsync(transaction => transaction.Id == id, cancellationToken);

    public Task<ConversionTransaction?> GetTransactionByIdempotencyKeyAsync(
        string idempotencyKey,
        CancellationToken cancellationToken) =>
        dbContext.ConversionTransactions.FirstOrDefaultAsync(
            transaction => transaction.IdempotencyKey == idempotencyKey,
            cancellationToken);

    public async Task<IReadOnlyList<ConversionTransaction>> ListTransactionsInFlightAsync(
        int limit,
        CancellationToken cancellationToken) =>
        await dbContext.ConversionTransactions
            .Where(transaction => transaction.Status == ConversionTransactionStatus.Pending
                || transaction.Status == ConversionTransactionStatus.Processing)
            .OrderBy(transaction => transaction.CreatedAt)
            .Take(limit)
            .ToListAsync(cancellationToken);

    public async Task<IReadOnlyList<ConversionTransaction>> ListRecentTransactionsAsync(
        int limit,
        CancellationToken cancellationToken) =>
        await dbContext.ConversionTransactions
            .OrderByDescending(transaction => transaction.CreatedAt)
            .Take(limit)
            .ToListAsync(cancellationToken);

    public async Task<IReadOnlyList<ConversionTransaction>> ListCompletedWithoutDepositAsync(
        CancellationToken cancellationToken)
    {
        var completed = await dbContext.ConversionTransactions
            .Where(transaction => transaction.Status == ConversionTransactionStatus.Completed)
            .OrderBy(transaction => transaction.CompletedAt)
            .ToListAsync(cancellationToken);

        // Filtering in memory avoids relying on string concatenation translating to SQL.
        var depositedKeys = (await dbContext.ProviderDeposits
                .Select(deposit => deposit.IdempotencyKey)
                .ToListAsync(cancellationToken))
            .ToHashSet(StringComparer.Ordinal);

        return completed
            .Where(transaction => !depositedKeys.Contains($"conversion:{transaction.Id}"))
            .ToList();
    }

    public Task<int> CountInFlightAsync(CancellationToken cancellationToken) =>
        dbContext.ConversionTransactions.CountAsync(
            transaction => transaction.Status == ConversionTransactionStatus.Pending
                || transaction.Status == ConversionTransactionStatus.Processing,
            cancellationToken);
}
