using MediatR;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using TokenMiner.Application.Configuration.Commands;
using TokenMiner.Application.Providers;

namespace TokenMiner.Infrastructure.BackgroundJobs;

/// <summary>
/// The auto top-up state machine's tick: act only when the provider balance has fallen below the
/// configured minimum, then advance each treasury stage in turn.
/// </summary>
internal sealed class AutoTopUpJob(
    IServiceScopeFactory scopeFactory,
    ProviderOptions options,
    ILogger<AutoTopUpJob> logger)
    : PeriodicJob(scopeFactory, TimeSpan.FromSeconds(Math.Max(15, options.TopUpIntervalSeconds)), logger)
{
    protected override async Task RunOnceAsync(IServiceProvider services, CancellationToken cancellationToken)
    {
        var sender = services.GetRequiredService<ISender>();

        var status = await sender.Send(new RunAutoTopUpCommand(), cancellationToken);

        switch (status.Outcome)
        {
            case TopUpOutcome.Driven:
                logger.LogInformation(
                    "Auto top-up drove the pipeline: {Payouts} payout(s), {Shares} share(s) redeemed, "
                    + "{Queued} conversion(s) queued, {Advanced} advanced, {Deposits} deposit(s) queued.",
                    status.PayoutsRecorded,
                    status.SharesRedeemed,
                    status.ConversionsQueued,
                    status.ConversionsAdvanced,
                    status.DepositsQueued);
                break;

            case TopUpOutcome.Disabled:
                logger.LogDebug("Auto top-up is disabled by configuration.");
                break;

            default:
                logger.LogDebug(
                    "Provider balance {Balance} is above the minimum {Minimum}; nothing to do.",
                    status.Balance,
                    status.Minimum);
                break;
        }
    }
}
