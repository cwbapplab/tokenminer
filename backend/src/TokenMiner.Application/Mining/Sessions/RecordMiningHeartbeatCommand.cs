using MediatR;
using TokenMiner.Application.Common.Abstractions;
using TokenMiner.Application.Mining.Abstractions;
using TokenMiner.Domain.Mining;
using TokenMiner.Domain.Mining.Enums;

namespace TokenMiner.Application.Mining.Sessions;

public sealed record MiningHeartbeatResult(bool SessionFound, string? Status, bool Resumed, DateTimeOffset At);

public sealed record RecordMiningHeartbeatCommand(Guid UserId, Guid HardwareId)
    : IRequest<MiningHeartbeatResult>;

/// <summary>
/// Records liveness for a device's session. A session that was paused after a connection
/// failure resumes on the next heartbeat, which is how the client's reconnect is detected.
/// </summary>
internal sealed class RecordMiningHeartbeatCommandHandler(
    IUserHardwareRepository hardware,
    IMiningSessionRepository sessions,
    IMiningLogRepository logs,
    IUnitOfWork unitOfWork,
    TimeProvider timeProvider) : IRequestHandler<RecordMiningHeartbeatCommand, MiningHeartbeatResult>
{
    public async Task<MiningHeartbeatResult> Handle(
        RecordMiningHeartbeatCommand request,
        CancellationToken cancellationToken)
    {
        var now = timeProvider.GetUtcNow();

        var device = await hardware.GetByUserAndHardwareIdAsync(
            request.UserId,
            request.HardwareId,
            cancellationToken);

        if (device is null)
        {
            return new MiningHeartbeatResult(SessionFound: false, Status: null, Resumed: false, At: now);
        }

        var session = await sessions.GetActiveByHardwareAsync(device.Id, cancellationToken);
        if (session is null)
        {
            return new MiningHeartbeatResult(SessionFound: false, Status: null, Resumed: false, At: now);
        }

        var resumed = false;

        if (session.Status == MiningSessionStatus.Paused
            && string.Equals(session.PauseReason, MiningStopReasonDbValues.ConnectionFailure, StringComparison.Ordinal))
        {
            resumed = session.Resume(now);

            if (resumed)
            {
                logs.Add(new MiningLog(
                    Guid.NewGuid(),
                    request.UserId,
                    device.Id,
                    session.Id,
                    session.PoolId,
                    session.CoinId,
                    MiningEventType.MiningResumed,
                    reason: null,
                    metadata: null,
                    now));
            }
        }

        session.Touch(now);
        device.Touch(now);

        await unitOfWork.SaveChangesAsync(cancellationToken);

        return new MiningHeartbeatResult(
            SessionFound: true,
            Status: session.Status.ToDbValue(),
            Resumed: resumed,
            At: now);
    }
}
