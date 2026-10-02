using MediatR;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using TokenMiner.Application.Treasury;
using TokenMiner.Application.Treasury.Commands;
using TokenMiner.Infrastructure.BackgroundJobs;

namespace TokenMiner.Infrastructure.BackgroundJobs;

/// <summary>Polls pools for balances and payout history.</summary>
internal sealed class PoolMonitorJob(
    IServiceScopeFactory scopeFactory,
    TreasuryOptions options,
    ILogger<PoolMonitorJob> logger)
    : PeriodicJob(scopeFactory, TimeSpan.FromSeconds(Math.Max(5, options.PoolMonitorIntervalSeconds)), logger)
{
    protected override async Task RunOnceAsync(IServiceProvider services, CancellationToken cancellationToken)
    {
        var sender = services.GetRequiredService<ISender>();

        var result = await sender.Send(new MonitorPoolsCommand(), cancellationToken);

        if (result.PayoutsRecorded > 0 || result.WithdrawalsRequested > 0)
        {
            logger.LogInformation(
                "Pool monitor recorded {Recorded} payout(s) and requested {Requested} withdrawal(s).",
                result.PayoutsRecorded,
                result.WithdrawalsRequested);
        }
    }
}

/// <summary>Settles confirmed payouts and attributes them to earned shares.</summary>
internal sealed class PayoutReconciliationJob(
    IServiceScopeFactory scopeFactory,
    TreasuryOptions options,
    ILogger<PayoutReconciliationJob> logger)
    : PeriodicJob(
        scopeFactory,
        TimeSpan.FromSeconds(Math.Max(5, options.PayoutReconciliationIntervalSeconds)),
        logger)
{
    protected override async Task RunOnceAsync(IServiceProvider services, CancellationToken cancellationToken)
    {
        var sender = services.GetRequiredService<ISender>();

        var result = await sender.Send(new ReconcilePayoutsCommand(), cancellationToken);

        if (result.PayoutsConfirmed > 0 || result.SharesRedeemed > 0)
        {
            logger.LogInformation(
                "Payout reconciliation confirmed {Confirmed} payout(s) and redeemed {Redeemed} share(s).",
                result.PayoutsConfirmed,
                result.SharesRedeemed);
        }
    }
}

/// <summary>Refreshes cached coin prices and appends history.</summary>
internal sealed class CoinPriceJob(
    IServiceScopeFactory scopeFactory,
    TreasuryOptions options,
    ILogger<CoinPriceJob> logger)
    : PeriodicJob(scopeFactory, TimeSpan.FromSeconds(Math.Max(30, options.CoinPriceIntervalSeconds)), logger)
{
    protected override async Task RunOnceAsync(IServiceProvider services, CancellationToken cancellationToken)
    {
        var sender = services.GetRequiredService<ISender>();

        var updated = await sender.Send(new RefreshCoinPricesCommand(), cancellationToken);

        if (updated > 0)
        {
            logger.LogInformation("Refreshed {Count} coin price(s).", updated);
        }
    }
}
