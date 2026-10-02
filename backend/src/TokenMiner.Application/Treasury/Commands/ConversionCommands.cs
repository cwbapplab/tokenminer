using MediatR;
using Microsoft.Extensions.Logging;
using TokenMiner.Application.Common.Abstractions;
using TokenMiner.Application.Mining.Abstractions;
using TokenMiner.Application.Treasury.Abstractions;
using TokenMiner.Domain.Treasury;
using TokenMiner.Domain.Treasury.Enums;

namespace TokenMiner.Application.Treasury.Commands;

// --- Route selection ----------------------------------------------------------------------

/// <summary>
/// Picks the conversion route for a source coin: highest priority first, then the cheapest
/// minimum, and only routes whose provider is still active.
/// </summary>
public sealed class ConversionRouteSelector(IConversionRepository conversions)
{
    public async Task<(ConversionRoute Route, ConversionProvider Provider)?> SelectAsync(
        Guid sourceCoinId,
        decimal amount,
        CancellationToken cancellationToken)
    {
        var routes = await conversions.ListRoutesAsync(cancellationToken);

        var candidates = routes
            .Where(route => route.Enabled
                && route.SourceCoinId == sourceCoinId
                && route.MinimumAmount <= amount)
            .OrderByDescending(route => route.Priority)
            .ThenBy(route => route.MinimumAmount);

        foreach (var route in candidates)
        {
            var provider = await conversions.GetProviderByIdAsync(route.ConversionProviderId, cancellationToken);

            if (provider is { Status: TreasuryStatus.Active })
            {
                return (route, provider);
            }
        }

        return null;
    }
}

// --- Queue ------------------------------------------------------------------------------

public sealed record QueuePayoutConversionsCommand : IRequest<int>;

/// <summary>
/// Creates a conversion for every confirmed pool payout that does not have one yet. Idempotent
/// by construction: the idempotency key is derived from the payout, so a payout converts once.
/// </summary>
internal sealed class QueuePayoutConversionsCommandHandler(
    IPoolPayoutRepository payouts,
    IConversionRepository conversions,
    ConversionRouteSelector selector,
    IUnitOfWork unitOfWork,
    TimeProvider timeProvider,
    ILogger<QueuePayoutConversionsCommandHandler> logger)
    : IRequestHandler<QueuePayoutConversionsCommand, int>
{
    public async Task<int> Handle(
        QueuePayoutConversionsCommand request,
        CancellationToken cancellationToken)
    {
        var now = timeProvider.GetUtcNow();
        var queued = 0;

        foreach (var payout in await payouts.ListConfirmedAsync(cancellationToken))
        {
            var idempotencyKey = $"payout:{payout.Id}";

            if (await conversions.GetTransactionByIdempotencyKeyAsync(idempotencyKey, cancellationToken) is not null)
            {
                continue;
            }

            if (payout.Amount <= 0)
            {
                continue;
            }

            var selection = await selector.SelectAsync(payout.CoinId, payout.Amount, cancellationToken);

            if (selection is null)
            {
                logger.LogDebug(
                    "No conversion route for payout {Payout}; it stays unconverted until one is configured.",
                    payout.Id);
                continue;
            }

            var (route, _) = selection.Value;

            conversions.AddTransaction(new ConversionTransaction(
                Guid.NewGuid(),
                route.ConversionProviderId,
                payout.CoinId,
                payout.Amount,
                route.DestinationCoinId,
                idempotencyKey,
                now));

            queued++;
        }

        if (queued > 0)
        {
            await unitOfWork.SaveChangesAsync(cancellationToken);
        }

        return queued;
    }
}

// --- Advance ----------------------------------------------------------------------------

public sealed record AdvanceConversionsCommand(int BatchSize) : IRequest<int>;

/// <summary>
/// Drives in-flight conversions through their provider. Providers are required to be idempotent
/// on the transaction's key, so re-driving a transaction cannot double-sell.
/// </summary>
internal sealed class AdvanceConversionsCommandHandler(
    IConversionRepository conversions,
    ICoinRepository coins,
    IEnumerable<IConversionProvider> providers,
    IUnitOfWork unitOfWork,
    TimeProvider timeProvider,
    ILogger<AdvanceConversionsCommandHandler> logger)
    : IRequestHandler<AdvanceConversionsCommand, int>
{
    public async Task<int> Handle(AdvanceConversionsCommand request, CancellationToken cancellationToken)
    {
        var providerLookup = providers.ToDictionary(
            provider => provider.Name,
            StringComparer.OrdinalIgnoreCase);

        var inFlight = await conversions.ListTransactionsInFlightAsync(
            Math.Clamp(request.BatchSize, 1, 500),
            cancellationToken);

        var now = timeProvider.GetUtcNow();
        var advanced = 0;

        foreach (var transaction in inFlight)
        {
            var providerRecord = await conversions.GetProviderByIdAsync(
                transaction.ConversionProviderId,
                cancellationToken);

            if (providerRecord is null
                || !providerLookup.TryGetValue(providerRecord.Name, out var provider))
            {
                logger.LogWarning(
                    "Conversion {Transaction} uses provider {Provider}, which has no registered adapter.",
                    transaction.Id,
                    providerRecord?.Name ?? "(unknown)");
                continue;
            }

            var sourceCoin = await coins.GetByIdAsync(transaction.SourceCoinId, cancellationToken);
            var destinationCoin = await coins.GetByIdAsync(transaction.DestinationCoinId, cancellationToken);

            if (sourceCoin is null || destinationCoin is null)
            {
                transaction.Fail("Source or destination coin no longer exists.", now);
                advanced++;
                continue;
            }

            ConversionResult result;

            try
            {
                result = await provider.ExecuteAsync(
                    new ConversionExecutionRequest(
                        transaction.Id,
                        transaction.IdempotencyKey,
                        sourceCoin.Code,
                        destinationCoin.Code,
                        transaction.SourceAmount),
                    cancellationToken);
            }
            catch (Exception exception)
            {
                // A provider failure is retryable: the transaction stays in flight.
                logger.LogWarning(exception, "Conversion {Transaction} failed at the provider.", transaction.Id);
                transaction.MarkProcessing(null, now);
                advanced++;
                continue;
            }

            switch (result.Outcome)
            {
                case ConversionOutcome.Completed:
                    transaction.CompleteFromProvider(
                        result.DestinationAmount,
                        result.ExchangeRate,
                        result.Fees,
                        result.DestinationTransactionId,
                        now);
                    advanced++;
                    break;

                case ConversionOutcome.Failed:
                    transaction.Fail(result.Error ?? "The provider reported a failure.", now);
                    advanced++;
                    break;

                default:
                    transaction.MarkProcessing(result.SourceTransactionId, now);
                    advanced++;
                    break;
            }
        }

        if (advanced > 0)
        {
            await unitOfWork.SaveChangesAsync(cancellationToken);
        }

        return advanced;
    }
}

// --- Manual settlement -------------------------------------------------------------------

public enum ConversionSettlementOutcome
{
    Completed = 0,
    Failed = 1,
    Cancelled = 2,
}

public sealed record SettleConversionCommand(
    Guid TransactionId,
    ConversionSettlementOutcome Outcome,
    decimal? DestinationAmount,
    decimal? ExchangeRate,
    decimal? Fees,
    string? DestinationTransactionId,
    string? Reason) : IRequest<bool>;

/// <summary>
/// Records the outcome of a conversion an operator performed off-system. This is how the manual
/// provider is settled; it is also the escape hatch for a conversion that failed at a live
/// provider.
/// </summary>
internal sealed class SettleConversionCommandHandler(
    IConversionRepository conversions,
    IUnitOfWork unitOfWork,
    TimeProvider timeProvider) : IRequestHandler<SettleConversionCommand, bool>
{
    public async Task<bool> Handle(SettleConversionCommand request, CancellationToken cancellationToken)
    {
        var transaction = await conversions.GetTransactionByIdAsync(request.TransactionId, cancellationToken);

        if (transaction is null)
        {
            return false;
        }

        var now = timeProvider.GetUtcNow();

        switch (request.Outcome)
        {
            case ConversionSettlementOutcome.Completed:
                transaction.CompleteFromProvider(
                    request.DestinationAmount,
                    request.ExchangeRate,
                    request.Fees,
                    request.DestinationTransactionId,
                    now);
                break;

            case ConversionSettlementOutcome.Failed:
                transaction.Fail(request.Reason ?? "Marked as failed.", now);
                break;

            default:
                transaction.Cancel(request.Reason ?? "Cancelled.", now);
                break;
        }

        await unitOfWork.SaveChangesAsync(cancellationToken);

        return true;
    }
}
