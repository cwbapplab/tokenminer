using TokenMiner.Domain.Mining.Enums;

namespace TokenMiner.Domain.Mining;

/// <summary>
/// A mining algorithm the system can hand to a client. Selection metadata lives here so the
/// API never hard-codes which algorithm to use. <see cref="Configuration"/> holds the
/// algorithm-specific launch template as raw JSON.
/// </summary>
public sealed class MiningAlgo
{
    private MiningAlgo()
    {
    }

    public MiningAlgo(
        Guid id,
        string code,
        string name,
        int priority,
        string? configuration,
        DateTimeOffset now)
    {
        Id = id;
        Code = code;
        Name = name;
        Priority = priority;
        Configuration = configuration;
        Status = MiningStatus.Active;
        CreatedAt = now;
        UpdatedAt = now;
    }

    public Guid Id { get; private set; }

    public string Code { get; private set; } = null!;

    public string Name { get; private set; } = null!;

    public MiningStatus Status { get; private set; }

    /// <summary>Higher values win when the API selects the best algorithm.</summary>
    public int Priority { get; private set; }

    public string? Configuration { get; private set; }

    public DateTimeOffset CreatedAt { get; private set; }

    public DateTimeOffset UpdatedAt { get; private set; }

    public void Update(string name, int priority, string? configuration, MiningStatus status, DateTimeOffset now)
    {
        Name = name;
        Priority = priority;
        Configuration = configuration;
        Status = status;
        UpdatedAt = now;
    }
}
