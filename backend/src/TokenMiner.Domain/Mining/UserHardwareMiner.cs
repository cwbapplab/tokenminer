using TokenMiner.Domain.Mining.Enums;

namespace TokenMiner.Domain.Mining;

/// <summary>
/// The current mining state of one device. At most one session may be <c>running</c> or
/// <c>paused</c> per device; stopped sessions accumulate as history.
/// </summary>
public sealed class UserHardwareMiner
{
    private UserHardwareMiner()
    {
    }

    public UserHardwareMiner(
        Guid id,
        Guid userHardwareId,
        Guid poolId,
        Guid coinId,
        Guid miningAlgoId,
        string workerIdentifier,
        DateTimeOffset now)
    {
        Id = id;
        UserHardwareId = userHardwareId;
        PoolId = poolId;
        CoinId = coinId;
        MiningAlgoId = miningAlgoId;
        WorkerIdentifier = workerIdentifier;
        Status = MiningSessionStatus.Running;
        StartedAt = now;
        LastActivityAt = now;
    }

    public Guid Id { get; private set; }

    public Guid UserHardwareId { get; private set; }

    public Guid PoolId { get; private set; }

    public Guid CoinId { get; private set; }

    public Guid MiningAlgoId { get; private set; }

    /// <summary>Stratum worker identity, <c>{userId}-{userHardwareId}</c>.</summary>
    public string WorkerIdentifier { get; private set; } = null!;

    public MiningSessionStatus Status { get; private set; }

    public DateTimeOffset StartedAt { get; private set; }

    public DateTimeOffset? PausedAt { get; private set; }

    public DateTimeOffset? StoppedAt { get; private set; }

    public DateTimeOffset LastActivityAt { get; private set; }

    public string? PauseReason { get; private set; }

    public string? StopReason { get; private set; }

    public bool IsActive => Status is MiningSessionStatus.Running or MiningSessionStatus.Paused;

    public void Touch(DateTimeOffset now) => LastActivityAt = now;

    /// <returns><c>true</c> when the session actually transitioned to paused.</returns>
    public bool Pause(string reason, DateTimeOffset now)
    {
        if (Status != MiningSessionStatus.Running)
        {
            return false;
        }

        Status = MiningSessionStatus.Paused;
        PausedAt = now;
        PauseReason = reason;
        LastActivityAt = now;

        return true;
    }

    /// <returns><c>true</c> when the session actually transitioned back to running.</returns>
    public bool Resume(DateTimeOffset now)
    {
        if (Status != MiningSessionStatus.Paused)
        {
            return false;
        }

        Status = MiningSessionStatus.Running;
        PausedAt = null;
        PauseReason = null;
        LastActivityAt = now;

        return true;
    }

    /// <returns><c>true</c> when the session actually transitioned to stopped.</returns>
    public bool Stop(string reason, DateTimeOffset now)
    {
        if (Status == MiningSessionStatus.Stopped)
        {
            return false;
        }

        Status = MiningSessionStatus.Stopped;
        StoppedAt = now;
        StopReason = reason;

        return true;
    }
}
