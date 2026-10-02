using MediatR;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using TokenMiner.Application.Mining;
using TokenMiner.Application.Mining.Shares;
using TokenMiner.Application.Mining.Statistics;

namespace TokenMiner.Infrastructure.BackgroundJobs;

/// <summary>Advances recorded mining shares through the reward lifecycle.</summary>
internal sealed class ShareRewardProcessorJob(
    IServiceScopeFactory scopeFactory,
    MiningOptions options,
    ILogger<ShareRewardProcessorJob> logger)
    : PeriodicJob(scopeFactory, TimeSpan.FromSeconds(Math.Max(1, options.ShareProcessorIntervalSeconds)), logger)
{
    protected override async Task RunOnceAsync(IServiceProvider services, CancellationToken cancellationToken)
    {
        var sender = services.GetRequiredService<ISender>();

        var processed = await sender.Send(
            new ProcessPendingSharesCommand(options.ShareProcessorBatchSize),
            cancellationToken);

        if (processed > 0)
        {
            logger.LogInformation("Processed {Count} mining share(s).", processed);
        }
    }
}

/// <summary>Rebuilds the materialised mining totals that analytics reads.</summary>
internal sealed class MiningStatisticsRollupJob(
    IServiceScopeFactory scopeFactory,
    MiningOptions options,
    ILogger<MiningStatisticsRollupJob> logger)
    : PeriodicJob(scopeFactory, TimeSpan.FromSeconds(Math.Max(1, options.StatisticsIntervalSeconds)), logger)
{
    protected override async Task RunOnceAsync(IServiceProvider services, CancellationToken cancellationToken)
    {
        var sender = services.GetRequiredService<ISender>();

        var rows = await sender.Send(new RecalculateMiningStatisticsCommand(), cancellationToken);

        if (rows > 0)
        {
            logger.LogInformation("Rebuilt {Count} mining statistics row(s).", rows);
        }
    }
}
