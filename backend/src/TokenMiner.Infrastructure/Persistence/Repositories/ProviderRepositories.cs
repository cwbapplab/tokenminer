using Microsoft.EntityFrameworkCore;
using TokenMiner.Application.Providers.Abstractions;
using TokenMiner.Domain.Providers;
using TokenMiner.Domain.Providers.Enums;

namespace TokenMiner.Infrastructure.Persistence.Repositories;

internal sealed class LlmProviderRepository(AppDbContext dbContext) : ILlmProviderRepository
{
    public void AddProvider(LlmProvider provider) => dbContext.LlmProviders.Add(provider);

    public void AddDepositAccount(LlmProviderDepositAccount account) =>
        dbContext.LlmProviderDepositAccounts.Add(account);

    public void AddModel(LlmProviderModel model) => dbContext.LlmProviderModels.Add(model);

    public void AddBalanceSnapshot(ProviderBalanceSnapshot snapshot) =>
        dbContext.ProviderBalanceSnapshots.Add(snapshot);

    public Task<LlmProvider?> GetProviderByIdAsync(Guid id, CancellationToken cancellationToken) =>
        dbContext.LlmProviders.FirstOrDefaultAsync(provider => provider.Id == id, cancellationToken);

    public Task<LlmProvider?> GetProviderByNameAsync(string name, CancellationToken cancellationToken) =>
        dbContext.LlmProviders.FirstOrDefaultAsync(provider => provider.Name == name, cancellationToken);

    public async Task<IReadOnlyList<LlmProvider>> ListProvidersAsync(CancellationToken cancellationToken) =>
        await dbContext.LlmProviders.OrderBy(provider => provider.Name).ToListAsync(cancellationToken);

    public Task<LlmProviderDepositAccount?> GetDepositAccountAsync(
        Guid llmProviderId,
        Guid coinId,
        string network,
        CancellationToken cancellationToken) =>
        dbContext.LlmProviderDepositAccounts.FirstOrDefaultAsync(
            account => account.LlmProviderId == llmProviderId
                && account.CoinId == coinId
                && account.Network == network,
            cancellationToken);

    public async Task<IReadOnlyList<LlmProviderDepositAccount>> ListDepositAccountsAsync(
        CancellationToken cancellationToken) =>
        await dbContext.LlmProviderDepositAccounts.ToListAsync(cancellationToken);

    public async Task<IReadOnlyList<LlmProviderModel>> ListModelsAsync(
        Guid llmProviderId,
        CancellationToken cancellationToken) =>
        await dbContext.LlmProviderModels
            .Where(model => model.LlmProviderId == llmProviderId)
            .ToListAsync(cancellationToken);

    public Task<ProviderBalanceSnapshot?> GetLatestBalanceAsync(
        Guid llmProviderId,
        CancellationToken cancellationToken) =>
        dbContext.ProviderBalanceSnapshots
            .Where(snapshot => snapshot.LlmProviderId == llmProviderId)
            .OrderByDescending(snapshot => snapshot.CheckedAt)
            .FirstOrDefaultAsync(cancellationToken);
}

internal sealed class ProviderDepositRepository(AppDbContext dbContext) : IProviderDepositRepository
{
    public void Add(ProviderDeposit deposit) => dbContext.ProviderDeposits.Add(deposit);

    public Task<ProviderDeposit?> GetByIdempotencyKeyAsync(
        string idempotencyKey,
        CancellationToken cancellationToken) =>
        dbContext.ProviderDeposits.FirstOrDefaultAsync(
            deposit => deposit.IdempotencyKey == idempotencyKey,
            cancellationToken);

    public async Task<IReadOnlyList<ProviderDeposit>> ListInFlightAsync(CancellationToken cancellationToken) =>
        await dbContext.ProviderDeposits
            .Where(deposit => deposit.Status == ProviderDepositStatus.Pending
                || deposit.Status == ProviderDepositStatus.Broadcast
                || deposit.Status == ProviderDepositStatus.AwaitingConfirmations
                || deposit.Status == ProviderDepositStatus.Confirmed)
            .OrderBy(deposit => deposit.CreatedAt)
            .ToListAsync(cancellationToken);

    public Task<int> CountInFlightAsync(CancellationToken cancellationToken) =>
        dbContext.ProviderDeposits.CountAsync(
            deposit => deposit.Status == ProviderDepositStatus.Pending
                || deposit.Status == ProviderDepositStatus.Broadcast
                || deposit.Status == ProviderDepositStatus.AwaitingConfirmations
                || deposit.Status == ProviderDepositStatus.Confirmed,
            cancellationToken);

    public async Task<IReadOnlyList<ProviderDeposit>> ListRecentAsync(
        int limit,
        CancellationToken cancellationToken) =>
        await dbContext.ProviderDeposits
            .OrderByDescending(deposit => deposit.CreatedAt)
            .Take(limit)
            .ToListAsync(cancellationToken);
}
