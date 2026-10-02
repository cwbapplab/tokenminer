using TokenMiner.Domain.Mining.Enums;

namespace TokenMiner.Domain.Mining;

/// <summary>
/// A physical mining device owned by a user. <see cref="HardwareId"/> is the GUID the client
/// generates; the pair (user, hardware) is unique.
/// </summary>
public sealed class UserHardware
{
    private UserHardware()
    {
    }

    public UserHardware(Guid id, Guid userId, Guid hardwareId, string? name, DateTimeOffset now)
    {
        Id = id;
        UserId = userId;
        HardwareId = hardwareId;
        Name = name;
        CreatedAt = now;
        LastSeenAt = now;
        Status = MiningStatus.Active;
    }

    public Guid Id { get; private set; }

    public Guid UserId { get; private set; }

    public Guid HardwareId { get; private set; }

    public string? Name { get; private set; }

    public DateTimeOffset CreatedAt { get; private set; }

    public DateTimeOffset? LastSeenAt { get; private set; }

    public MiningStatus Status { get; private set; }

    public void Touch(DateTimeOffset now) => LastSeenAt = now;

    public void Update(string? name, MiningStatus status, DateTimeOffset now)
    {
        Name = name;
        Status = status;
        LastSeenAt = now;
    }
}
