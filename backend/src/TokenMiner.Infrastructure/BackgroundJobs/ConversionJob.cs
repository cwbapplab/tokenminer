using MediatR;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using TokenMiner.Application.Treasury;
using TokenMiner.Application.Treasury.Commands;

namespace TokenMiner.Infrastructure.BackgroundJobs;

/// <summary>
/// Queues conversions for settled payouts and drives the in-flight ones.
/// </summary>
/// <remarks>
/// A manual conversion deliberately stays in flight until an operator settles it, so this job
/// will pick it up on every pass. That is harmless: the work is a no-op until the provider says
/// otherwise, and every transition is guarded by the transaction's state and idempotency key.
/// </remarks>
internal sealed class ConversionJob(
    IServiceScopeFactory scopeFactory,
    TreasuryOptions options,
    ILogger<ConversionJob> logger)
    : PeriodicJob(scopeFactory, TimeSpan.FromSeconds(Math.Max(10, options.ConversionIntervalSeconds)), logger)
{
    protected override async Task RunOnceAsync(IServiceProvider services, CancellationToken cancellationToken)
    {
        var sender = services.GetRequiredService<ISender>();

        var result = await sender.Send(
            new RunConversionPipelineCommand(options.ConversionBatchSize),
            cancellationToken);

        if (result.Queued > 0 || result.Advanced > 0)
        {
            logger.LogInformation(
                "Conversion pipeline queued {Queued} and advanced {Advanced} transaction(s).",
                result.Queued,
                result.Advanced);
        }
    }
}
