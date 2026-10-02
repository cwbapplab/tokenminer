using TokenMiner.Domain.Mining.Enums;

namespace TokenMiner.Domain.Mining;

/// <summary>
/// Auditable mining lifecycle event. <see cref="Metadata"/> carries event-specific detail as
/// raw JSON so new event types do not require schema changes.
/// </summary>
public sealed class MiningLog
{
    private MiningLog()
    {
    }

    public MiningLog(
        Guid id,
        Guid userId,
        Guid userHardwareId,
        Guid userHardwareMinerId,
        Guid poolId,
        Guid coinId,
        MiningEventType eventType,
        string? reason,
        string? metadata,
        DateTimeOffset now)
    {
        Id = id;
        UserId = userId;
        UserHardwareId = userHardwareId;
        UserHardwareMinerId = userHardwareMinerId;
        PoolId = poolId;
        CoinId = coinId;
        EventType = eventType;
        Reason = reason;
        Metadata = metadata;
        CreatedAt = now;
    }

    public Guid Id { get; private set; }

    public Guid UserId { get; private set; }

    public Guid UserHardwareId { get; private set; }

    public Guid UserHardwareMinerId { get; private set; }

    public Guid PoolId { get; private set; }

    public Guid CoinId { get; private set; }

    public MiningEventType EventType { get; private set; }

    public string? Reason { get; private set; }

    public string? Metadata { get; private set; }

    public DateTimeOffset CreatedAt { get; private set; }
}
