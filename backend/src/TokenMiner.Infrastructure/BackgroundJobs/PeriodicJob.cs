using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;

namespace TokenMiner.Infrastructure.BackgroundJobs;

/// <summary>
/// A job that runs on a fixed interval in its own dependency-injection scope. A failing tick is
/// logged and swallowed so one bad run cannot kill the loop.
/// </summary>
/// <remarks>
/// These move to the durable job runner alongside the remaining background services; today the
/// set is small enough that a hosted service per job is clearer than the extra machinery.
/// </remarks>
internal abstract class PeriodicJob(
    IServiceScopeFactory scopeFactory,
    TimeSpan interval,
    ILogger logger) : BackgroundService
{
    protected override async Task ExecuteAsync(CancellationToken stoppingToken)
    {
        using var timer = new PeriodicTimer(interval);

        try
        {
            while (await timer.WaitForNextTickAsync(stoppingToken))
            {
                try
                {
                    using var scope = scopeFactory.CreateScope();
                    await RunOnceAsync(scope.ServiceProvider, stoppingToken);
                }
                catch (OperationCanceledException) when (stoppingToken.IsCancellationRequested)
                {
                    break;
                }
                catch (Exception exception)
                {
                    logger.LogError(exception, "{Job} tick failed.", GetType().Name);
                }
            }
        }
        catch (OperationCanceledException) when (stoppingToken.IsCancellationRequested)
        {
            // Waiting on the timer is where shutdown surfaces. Swallowing it here keeps a normal
            // stop from looking like a crash: the host otherwise reports "a BackgroundService has
            // thrown an unhandled exception, and the IHost instance is stopping".
        }
    }

    protected abstract Task RunOnceAsync(IServiceProvider services, CancellationToken cancellationToken);
}
