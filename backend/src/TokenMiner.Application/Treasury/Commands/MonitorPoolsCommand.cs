using System.Text.Json;
using MediatR;
using Microsoft.Extensions.Logging;
using TokenMiner.Application.Common.Abstractions;
using TokenMiner.Application.Mining.Abstractions;
using TokenMiner.Application.Treasury.Abstractions;
using TokenMiner.Domain.Mining.Enums;
using TokenMiner.Domain.Treasury;
using TokenMiner.Domain.Treasury.Enums;

namespace TokenMiner.Application.Treasury.Commands;

public sealed record MonitorPoolsCommand : IRequest<PoolMonitorResult>;

/// <summary>
/// Polls every active pool for its payout history and — only for pools we withdraw from
/// ourselves — its balance. Newly observed payouts are recorded; nothing is inferred.
/// </summary>
internal sealed class MonitorPoolsCommandHandler(
    IPoolRepository pools,
    ICoinRepository coins,
    IPoolPayoutRepository payouts,
    IEnumerable<IPoolPayoutProvider> providers,
    IUnitOfWork unitOfWork,
    TimeProvider timeProvider,
    TreasuryOptions options,
    ILogger<MonitorPoolsCommandHandler> logger)
    : IRequestHandler<MonitorPoolsCommand, PoolMonitorResult>
{
    public async Task<PoolMonitorResult> Handle(
        MonitorPoolsCommand request,
        CancellationToken cancellationToken)
    {
        var providerLookup = providers.ToDictionary(
            provider => provider.Name,
            StringComparer.OrdinalIgnoreCase);

        var allPools = await pools.ListAsync(cancellationToken);
        var allDetails = await pools.ListDetailsAsync(cancellationToken);
        var detailsByPool = allDetails.ToDictionary(details => details.PoolId);

        var now = timeProvider.GetUtcNow();
        var recorded = 0;
        var requested = 0;

        foreach (var pool in allPools.Where(pool => pool.Status == MiningStatus.Active))
        {
            if (!detailsByPool.TryGetValue(pool.Id, out var details))
            {
                continue;
            }

            if (!providerLookup.TryGetValue(pool.Provider, out var provider))
            {
                logger.LogWarning(
                    "Pool {Pool} uses provider {Provider}, which has no registered integration.",
                    pool.SystemPoolId,
                    pool.Provider);
                continue;
            }

            var coin = await coins.GetByIdAsync(details.CoinId, cancellationToken);
            if (coin is null)
            {
                continue;
            }

            var context = new PoolPayoutContext(
                pool.Id,
                pool.SystemPoolId,
                pool.Provider,
                coin.Code,
                details.PayoutAddress,
                details.BaseUrl);

            recorded += await RecordNewPayoutsAsync(provider, context, pool.Id, coin.Id, now, cancellationToken);

            // Only pools we withdraw from ourselves need a balance check and a withdrawal request.
            if (details.PayoutMode == PayoutMode.ManagedRequest)
            {
                requested += await RequestWithdrawalIfDueAsync(
                    provider,
                    context,
                    pool.Id,
                    coin.Id,
                    now,
                    cancellationToken);
            }
        }

        if (recorded > 0 || requested > 0)
        {
            await unitOfWork.SaveChangesAsync(cancellationToken);
        }

        return new PoolMonitorResult(recorded, requested);
    }

    private async Task<int> RecordNewPayoutsAsync(
        IPoolPayoutProvider provider,
        PoolPayoutContext context,
        Guid poolId,
        Guid coinId,
        DateTimeOffset now,
        CancellationToken cancellationToken)
    {
        IReadOnlyList<PoolPayoutRecord> reported;

        try
        {
            reported = await provider.GetPayoutsAsync(context, cancellationToken);
        }
        catch (Exception exception)
        {
            // A pool being unreachable must not stop the other pools from being monitored.
            logger.LogWarning(exception, "Reading payouts for pool {Pool} failed.", context.SystemPoolId);
            return 0;
        }

        var recorded = 0;

        foreach (var record in reported)
        {
            // A payout without a hash cannot be de-duplicated, so it is not recorded as history.
            if (string.IsNullOrWhiteSpace(record.TransactionHash))
            {
                continue;
            }

            if (await payouts.ExistsByTransactionHashAsync(poolId, record.TransactionHash, cancellationToken))
            {
                continue;
            }

            var payout = new PoolPayout(
                Guid.NewGuid(),
                poolId,
                coinId,
                context.PayoutAddress,
                record.Amount,
                record.TransactionHash,
                record.RequestedAt,
                record.ReceivedAt,
                record.Confirmations,
                JsonSerializer.Serialize(record),
                now);

            // A pool that reports its own final status needs no confirmation counting.
            if (record.IsSettled)
            {
                payout.Confirm(record.ReceivedAt ?? now, now);
            }

            payouts.Add(payout);

            recorded++;
        }

        return recorded;
    }

    private async Task<int> RequestWithdrawalIfDueAsync(
        IPoolPayoutProvider provider,
        PoolPayoutContext context,
        Guid poolId,
        Guid coinId,
        DateTimeOffset now,
        CancellationToken cancellationToken)
    {
        PoolBalance balance;

        try
        {
            balance = await provider.GetBalanceAsync(context, cancellationToken);
        }
        catch (Exception exception)
        {
            logger.LogWarning(exception, "Reading the balance for pool {Pool} failed.", context.SystemPoolId);
            return 0;
        }

        // Pools report their own threshold; fall back to the configured floor when they do not.
        var minimum = balance.Threshold ?? options.MinimumPayoutAmount;

        if (balance.Confirmed < minimum || minimum <= 0)
        {
            return 0;
        }

        try
        {
            var result = await provider.RequestWithdrawalAsync(context, balance.Confirmed, cancellationToken);

            if (!result.Requested)
            {
                logger.LogInformation(
                    "Withdrawal from {Pool} was not placed: {Detail}",
                    context.SystemPoolId,
                    result.Detail ?? "no reason given");
                return 0;
            }

            // Tracked so the payout can be matched to its transaction once the pool publishes it.
            payouts.Add(new PoolPayout(
                Guid.NewGuid(),
                poolId,
                coinId,
                context.PayoutAddress,
                balance.Confirmed,
                result.TransactionHash,
                now,
                null,
                0,
                result.Detail,
                now));

            return 1;
        }
        catch (Exception exception)
        {
            logger.LogError(exception, "Requesting a withdrawal from {Pool} failed.", context.SystemPoolId);
            return 0;
        }
    }
}
