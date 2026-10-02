using System.Text.Json;
using MediatR;
using Microsoft.Extensions.Logging;
using TokenMiner.Application.Common.Abstractions;
using TokenMiner.Application.Mining.Abstractions;
using TokenMiner.Application.Providers.Abstractions;
using TokenMiner.Application.Treasury.Abstractions;
using TokenMiner.Domain.Mining.Enums;
using TokenMiner.Domain.Providers;
using TokenMiner.Domain.Providers.Enums;

namespace TokenMiner.Application.Providers.Commands;

// --- Catalogue and balance ---------------------------------------------------------------

public sealed record SyncModelCatalogCommand : IRequest<int>;

/// <summary>
/// Mirrors each provider's model catalogue locally. Models the provider no longer lists are
/// disabled rather than deleted, so historical references stay resolvable.
/// </summary>
internal sealed class SyncModelCatalogCommandHandler(
    ILlmProviderRepository providers,
    ILlmProviderClient client,
    ISecretResolver secrets,
    IUnitOfWork unitOfWork,
    TimeProvider timeProvider,
    ILogger<SyncModelCatalogCommandHandler> logger)
    : IRequestHandler<SyncModelCatalogCommand, int>
{
    public async Task<int> Handle(SyncModelCatalogCommand request, CancellationToken cancellationToken)
    {
        var now = timeProvider.GetUtcNow();
        var synced = 0;

        foreach (var provider in await providers.ListProvidersAsync(cancellationToken))
        {
            if (provider.Status != MiningStatus.Active)
            {
                continue;
            }

            IReadOnlyList<LlmCatalogModel> catalogue;

            try
            {
                catalogue = await client.GetModelsAsync(
                    await ProviderContext.CreateAsync(provider, secrets, cancellationToken),
                    cancellationToken);
            }
            catch (Exception exception)
            {
                logger.LogWarning(exception, "Syncing the model catalogue for {Provider} failed.", provider.Name);
                continue;
            }

            var known = (await providers.ListModelsAsync(provider.Id, cancellationToken))
                .ToDictionary(model => model.ModelId, StringComparer.Ordinal);

            var seen = new HashSet<string>(StringComparer.Ordinal);

            foreach (var entry in catalogue)
            {
                seen.Add(entry.ModelId);

                var capabilities = entry.Capabilities.Count == 0
                    ? null
                    : JsonSerializer.Serialize(entry.Capabilities);

                if (known.TryGetValue(entry.ModelId, out var existing))
                {
                    existing.Sync(
                        entry.Name,
                        entry.InputCost,
                        entry.OutputCost,
                        entry.CachedInputCost,
                        entry.Currency,
                        entry.ContextLength,
                        capabilities,
                        now);
                }
                else
                {
                    providers.AddModel(new LlmProviderModel(
                        Guid.NewGuid(),
                        provider.Id,
                        entry.ModelId,
                        entry.Name,
                        entry.InputCost,
                        entry.OutputCost,
                        entry.CachedInputCost,
                        entry.Currency,
                        entry.ContextLength,
                        capabilities,
                        now));
                }

                synced++;
            }

            foreach (var stale in known.Values.Where(model => !seen.Contains(model.ModelId)))
            {
                stale.Disable(now);
            }
        }

        if (synced > 0)
        {
            await unitOfWork.SaveChangesAsync(cancellationToken);
        }

        return synced;
    }
}

/// <summary>Builds a client context, resolving the provider's API key from the secret store.</summary>
internal static class ProviderContext
{
    public static async Task<LlmProviderContext> CreateAsync(
        LlmProvider provider,
        ISecretResolver secrets,
        CancellationToken cancellationToken) =>
        new(
            provider.Name,
            provider.Endpoint,
            string.IsNullOrWhiteSpace(provider.CredentialRef)
                ? null
                : await secrets.GetSecretAsync(provider.CredentialRef, cancellationToken));
}

public sealed record RefreshProviderBalanceCommand : IRequest<int>;

/// <summary>Snapshots each provider's prepaid balance so top-up decisions have fresh data.</summary>
internal sealed class RefreshProviderBalanceCommandHandler(
    ILlmProviderRepository providers,
    ILlmProviderClient client,
    ISecretResolver secrets,
    IUnitOfWork unitOfWork,
    TimeProvider timeProvider,
    ILogger<RefreshProviderBalanceCommandHandler> logger)
    : IRequestHandler<RefreshProviderBalanceCommand, int>
{
    public async Task<int> Handle(RefreshProviderBalanceCommand request, CancellationToken cancellationToken)
    {
        var now = timeProvider.GetUtcNow();
        var refreshed = 0;

        foreach (var provider in await providers.ListProvidersAsync(cancellationToken))
        {
            if (provider.Status != MiningStatus.Active)
            {
                continue;
            }

            LlmBalance balance;

            try
            {
                balance = await client.GetBalanceAsync(
                    await ProviderContext.CreateAsync(provider, secrets, cancellationToken),
                    cancellationToken);
            }
            catch (Exception exception)
            {
                logger.LogWarning(exception, "Reading the balance for {Provider} failed.", provider.Name);
                continue;
            }

            providers.AddBalanceSnapshot(new ProviderBalanceSnapshot(
                Guid.NewGuid(),
                provider.Id,
                balance.Balance,
                balance.ReservedBalance,
                balance.TotalBalance,
                now));

            refreshed++;
        }

        if (refreshed > 0)
        {
            await unitOfWork.SaveChangesAsync(cancellationToken);
        }

        return refreshed;
    }
}

// --- Deposit pipeline ---------------------------------------------------------------------

public sealed record ProviderPipelineResult(int DepositsQueued, int DepositsAdvanced);

public sealed record RunProviderDepositPipelineCommand : IRequest<ProviderPipelineResult>;

/// <summary>
/// Queues a transfer for every completed conversion that does not have one, then advances the
/// transfers already in flight.
/// </summary>
/// <remarks>
/// Broadcasting and being credited are deliberately separate steps: the chain confirming a
/// transfer says nothing about whether the provider has credited the wallet, so a deposit only
/// becomes <c>credited</c> once the provider's own balance has grown.
/// </remarks>
internal sealed class RunProviderDepositPipelineCommandHandler(
    IConversionRepository conversions,
    ILlmProviderRepository providers,
    IProviderDepositRepository deposits,
    ICoinRepository coins,
    IBlockchainTransferProvider blockchain,
    ILlmProviderClient client,
    ISecretResolver secrets,
    IUnitOfWork unitOfWork,
    ProviderOptions options,
    TimeProvider timeProvider,
    ILogger<RunProviderDepositPipelineCommandHandler> logger)
    : IRequestHandler<RunProviderDepositPipelineCommand, ProviderPipelineResult>
{
    /// <summary>Balances move in fractions of a cent, so exact equality is too strict.</summary>
    private const decimal Tolerance = 0.0001m;

    public async Task<ProviderPipelineResult> Handle(
        RunProviderDepositPipelineCommand request,
        CancellationToken cancellationToken)
    {
        var queued = await QueueAsync(cancellationToken);
        var advanced = await AdvanceAsync(cancellationToken);

        if (queued > 0 || advanced > 0)
        {
            await unitOfWork.SaveChangesAsync(cancellationToken);
        }

        return new ProviderPipelineResult(queued, advanced);
    }

    private async Task<int> QueueAsync(CancellationToken cancellationToken)
    {
        var now = timeProvider.GetUtcNow();
        var queued = 0;

        var activeProviders = await providers.ListProvidersAsync(cancellationToken);
        var provider = activeProviders.FirstOrDefault(candidate => candidate.Status == MiningStatus.Active);

        if (provider is null)
        {
            return 0;
        }

        foreach (var transaction in await conversions.ListCompletedWithoutDepositAsync(cancellationToken))
        {
            var amount = transaction.DestinationAmount ?? 0m;
            if (amount <= 0)
            {
                continue;
            }

            var idempotencyKey = $"conversion:{transaction.Id}";

            if (await deposits.GetByIdempotencyKeyAsync(idempotencyKey, cancellationToken) is not null)
            {
                continue;
            }

            var coin = await coins.GetByIdAsync(transaction.DestinationCoinId, cancellationToken);
            if (coin is null)
            {
                continue;
            }

            // The rail is identified by coin and network, never by coin alone.
            var account = await providers.GetDepositAccountAsync(
                provider.Id,
                coin.Id,
                coin.Network,
                cancellationToken);

            if (account is null)
            {
                logger.LogDebug(
                    "Provider {Provider} has no deposit account for {Coin} on {Network}.",
                    provider.Name,
                    coin.Code,
                    coin.Network);
                continue;
            }

            var deposit = new ProviderDeposit(
                Guid.NewGuid(),
                provider.Id,
                coin.Id,
                account.Network,
                account.DepositAddress,
                amount,
                idempotencyKey,
                now);

            var balanceBefore = await TryReadBalanceAsync(provider, cancellationToken);

            var transfer = await blockchain.SendAsync(
                new BlockchainTransferRequest(
                    deposit.Id,
                    idempotencyKey,
                    account.Network,
                    account.DepositAddress,
                    coin.Code,
                    amount),
                cancellationToken);

            if (string.IsNullOrWhiteSpace(transfer.TransactionHash))
            {
                deposit.Fail(transfer.Error ?? "The transfer could not be broadcast.", now);
            }
            else
            {
                deposit.Broadcast(transfer.TransactionHash, balanceBefore ?? 0m, now);
            }

            deposits.Add(deposit);
            queued++;
        }

        return queued;
    }

    private async Task<int> AdvanceAsync(CancellationToken cancellationToken)
    {
        var now = timeProvider.GetUtcNow();
        var advanced = 0;

        foreach (var deposit in await deposits.ListInFlightAsync(cancellationToken))
        {
            if (deposit.TransactionHash is null)
            {
                continue;
            }

            if (deposit.Status is ProviderDepositStatus.Broadcast or ProviderDepositStatus.AwaitingConfirmations)
            {
                var confirmations = await blockchain.GetConfirmationsAsync(
                    deposit.Network,
                    deposit.TransactionHash,
                    cancellationToken);

                deposit.ObserveConfirmations(confirmations, now);

                if (confirmations >= options.ConfirmationThreshold)
                {
                    deposit.Confirm(now);
                }

                advanced++;
                continue;
            }

            if (deposit.Status != ProviderDepositStatus.Confirmed)
            {
                continue;
            }

            // On chain but not necessarily credited: only the provider's own balance can say so.
            var provider = await providers.GetProviderByIdAsync(deposit.LlmProviderId, cancellationToken);
            if (provider is null)
            {
                continue;
            }

            var current = await TryReadBalanceAsync(provider, cancellationToken);
            if (current is null)
            {
                continue;
            }

            var credited = current.Value - (deposit.ProviderCreditBefore ?? 0m);

            if (credited + Tolerance >= deposit.Amount)
            {
                deposit.MarkCredited(current.Value, now);
                advanced++;
            }
        }

        return advanced;
    }

    private async Task<decimal?> TryReadBalanceAsync(LlmProvider provider, CancellationToken cancellationToken)
    {
        try
        {
            var balance = await client.GetBalanceAsync(
                await ProviderContext.CreateAsync(provider, secrets, cancellationToken),
                cancellationToken);

            return balance.TotalBalance;
        }
        catch (Exception exception)
        {
            logger.LogWarning(exception, "Reading the balance for {Provider} failed.", provider.Name);
            return null;
        }
    }
}
