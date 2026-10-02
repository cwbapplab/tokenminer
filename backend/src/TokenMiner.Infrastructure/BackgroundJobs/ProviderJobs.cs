using MediatR;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using TokenMiner.Application.Providers;
using TokenMiner.Application.Providers.Commands;

namespace TokenMiner.Infrastructure.BackgroundJobs;

/// <summary>Snapshots each provider's prepaid balance.</summary>
internal sealed class ProviderBalanceJob(
    IServiceScopeFactory scopeFactory,
    ProviderOptions options,
    ILogger<ProviderBalanceJob> logger)
    : PeriodicJob(scopeFactory, TimeSpan.FromSeconds(Math.Max(30, options.BalanceIntervalSeconds)), logger)
{
    protected override async Task RunOnceAsync(IServiceProvider services, CancellationToken cancellationToken)
    {
        var sender = services.GetRequiredService<ISender>();

        var refreshed = await sender.Send(new RefreshProviderBalanceCommand(), cancellationToken);

        if (refreshed > 0)
        {
            logger.LogInformation("Snapshotted {Count} provider balance(s).", refreshed);
        }
    }
}

/// <summary>Mirrors the providers' model catalogues.</summary>
internal sealed class ModelCatalogSyncJob(
    IServiceScopeFactory scopeFactory,
    ProviderOptions options,
    ILogger<ModelCatalogSyncJob> logger)
    : PeriodicJob(
        scopeFactory,
        TimeSpan.FromSeconds(Math.Max(300, options.ModelCatalogIntervalSeconds)),
        logger)
{
    protected override async Task RunOnceAsync(IServiceProvider services, CancellationToken cancellationToken)
    {
        var sender = services.GetRequiredService<ISender>();

        var synced = await sender.Send(new SyncModelCatalogCommand(), cancellationToken);

        if (synced > 0)
        {
            logger.LogInformation("Synced {Count} provider model(s).", synced);
        }
    }
}

/// <summary>Queues and advances stablecoin transfers into the provider's wallet.</summary>
internal sealed class ProviderDepositPipelineJob(
    IServiceScopeFactory scopeFactory,
    ProviderOptions options,
    ILogger<ProviderDepositPipelineJob> logger)
    : PeriodicJob(scopeFactory, TimeSpan.FromSeconds(Math.Max(30, options.DepositIntervalSeconds)), logger)
{
    protected override async Task RunOnceAsync(IServiceProvider services, CancellationToken cancellationToken)
    {
        var sender = services.GetRequiredService<ISender>();

        var result = await sender.Send(new RunProviderDepositPipelineCommand(), cancellationToken);

        if (result.DepositsQueued > 0 || result.DepositsAdvanced > 0)
        {
            logger.LogInformation(
                "Provider deposit pipeline queued {Queued} and advanced {Advanced} transfer(s).",
                result.DepositsQueued,
                result.DepositsAdvanced);
        }
    }
}
