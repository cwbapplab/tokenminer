using MediatR;
using TokenMiner.Application.Common.Abstractions;
using TokenMiner.Application.Mining.Abstractions;
using TokenMiner.Domain.Mining;
using TokenMiner.Domain.Mining.Enums;

namespace TokenMiner.Application.Mining.Sessions;

public sealed record PauseStaleSessionsCommand(int GraceSeconds) : IRequest<int>;

/// <summary>
/// Pauses sessions that have stopped sending heartbeats. Because only <c>running</c> sessions
/// are considered, one outage produces exactly one pause event no matter how often this runs.
/// </summary>
internal sealed class PauseStaleSessionsCommandHandler(
    IMiningSessionRepository sessions,
    IUserHardwareRepository hardware,
    IMiningLogRepository logs,
    IUnitOfWork unitOfWork,
    TimeProvider timeProvider) : IRequestHandler<PauseStaleSessionsCommand, int>
{
    public async Task<int> Handle(PauseStaleSessionsCommand request, CancellationToken cancellationToken)
    {
        var now = timeProvider.GetUtcNow();
        var threshold = now.AddSeconds(-Math.Max(1, request.GraceSeconds));

        var stale = await sessions.ListRunningIdleSinceAsync(threshold, cancellationToken);
        if (stale.Count == 0)
        {
            return 0;
        }

        // One lookup for every affected device, rather than one query per session.
        var devices = await hardware.ListByIdsAsync(
            stale.Select(session => session.UserHardwareId).Distinct(),
            cancellationToken);

        var devicesById = devices.ToDictionary(device => device.Id);

        var pausedCount = 0;

        foreach (var session in stale)
        {
            if (!session.Pause(MiningStopReasonDbValues.ConnectionFailure, now))
            {
                continue;
            }

            if (!devicesById.TryGetValue(session.UserHardwareId, out var device))
            {
                continue;
            }

            logs.Add(new MiningLog(
                Guid.NewGuid(),
                device.UserId,
                device.Id,
                session.Id,
                session.PoolId,
                session.CoinId,
                MiningEventType.MiningPaused,
                reason: MiningStopReasonDbValues.ConnectionFailure,
                metadata: null,
                now));

            pausedCount++;
        }

        if (pausedCount > 0)
        {
            await unitOfWork.SaveChangesAsync(cancellationToken);
        }

        return pausedCount;
    }
}
