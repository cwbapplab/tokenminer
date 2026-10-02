using MediatR;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using TokenMiner.Application.Mining;
using TokenMiner.Application.Mining.Sessions;

namespace TokenMiner.Infrastructure.BackgroundJobs;

/// <summary>
/// Pauses mining sessions that stop sending heartbeats. A raw WebSocket cannot schedule its
/// own timeout, so liveness is enforced from the outside.
/// </summary>
internal sealed class MiningSessionWatchdog(
    IServiceScopeFactory scopeFactory,
    MiningOptions options,
    ILogger<MiningSessionWatchdog> logger)
    : PeriodicJob(scopeFactory, TimeSpan.FromSeconds(Math.Max(1, options.WatchdogIntervalSeconds)), logger)
{
    protected override async Task RunOnceAsync(IServiceProvider services, CancellationToken cancellationToken)
    {
        var sender = services.GetRequiredService<ISender>();

        var pausedCount = await sender.Send(
            new PauseStaleSessionsCommand(options.HeartbeatGraceSeconds),
            cancellationToken);

        if (pausedCount > 0)
        {
            logger.LogInformation(
                "Paused {Count} mining session(s) that stopped sending heartbeats.",
                pausedCount);
        }
    }
}
