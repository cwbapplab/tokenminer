using TokenMiner.Domain.Mining.Enums;

namespace TokenMiner.Domain.Mining;

/// <summary>
/// A supported mining pool. <see cref="SystemPoolId"/> is our stable identifier for the
/// pool account, independent of the database key.
/// </summary>
public sealed class Pool
{
    private Pool()
    {
    }

    public Pool(Guid id, string systemPoolId, string name, string provider, DateTimeOffset now)
    {
        Id = id;
        SystemPoolId = systemPoolId;
        Name = name;
        Provider = provider;
        Status = MiningStatus.Active;
        CreatedAt = now;
        UpdatedAt = now;
    }

    public Guid Id { get; private set; }

    public string SystemPoolId { get; private set; } = null!;

    public string Name { get; private set; } = null!;

    /// <summary>
    /// Which payout integration serves this pool, e.g. <c>kryptex</c>. Kept as data so a second
    /// pool vendor needs no code changes outside its own provider.
    /// </summary>
    public string Provider { get; private set; } = null!;

    public MiningStatus Status { get; private set; }

    public DateTimeOffset CreatedAt { get; private set; }

    public DateTimeOffset UpdatedAt { get; private set; }

    public void UpdateDetails(string name, string provider, MiningStatus status, DateTimeOffset now)
    {
        Name = name;
        Provider = provider;
        Status = status;
        UpdatedAt = now;
    }
}
