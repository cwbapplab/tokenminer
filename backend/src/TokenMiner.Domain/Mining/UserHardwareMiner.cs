namespace TokenMiner.Domain.Mining;

/// <summary>
/// The open mining session of one device. At most one session may be open (not yet stopped) per
/// device; stopped sessions accumulate as history. Whether the device is mining right now is
/// derived from its share activity, not stored here.
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

    public DateTimeOffset StartedAt { get; private set; }

    public DateTimeOffset? StoppedAt { get; private set; }

    public DateTimeOffset LastActivityAt { get; private set; }

    public string? StopReason { get; private set; }

    /// <summary>An open session, i.e. one that has not been stopped yet.</summary>
    public bool IsActive => StoppedAt is null;

    public void Touch(DateTimeOffset now) => LastActivityAt = now;

    /// <returns><c>true</c> when the session actually transitioned to stopped.</returns>
    public bool Stop(string reason, DateTimeOffset now)
    {
        if (StoppedAt is not null)
        {
            return false;
        }

        StoppedAt = now;
        StopReason = reason;

        return true;
    }
}
