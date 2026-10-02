namespace TokenMiner.Domain.Mining.Enums;

/// <summary>Auditable mining lifecycle events recorded in the mining log.</summary>
public enum MiningEventType
{
    MiningStarted = 0,
    MiningPaused = 1,
    MiningResumed = 2,
    MiningStopped = 3,
    ConnectionFailure = 4,
    AlgorithmSelected = 5,
    PoolChanged = 6,
}

public static class MiningEventTypeDbValues
{
    public const string MiningStarted = "mining_started";
    public const string MiningPaused = "mining_paused";
    public const string MiningResumed = "mining_resumed";
    public const string MiningStopped = "mining_stopped";
    public const string ConnectionFailure = "connection_failure";
    public const string AlgorithmSelected = "algorithm_selected";
    public const string PoolChanged = "pool_changed";

    public static string ToDbValue(this MiningEventType eventType) => eventType switch
    {
        MiningEventType.MiningStarted => MiningStarted,
        MiningEventType.MiningPaused => MiningPaused,
        MiningEventType.MiningResumed => MiningResumed,
        MiningEventType.MiningStopped => MiningStopped,
        MiningEventType.ConnectionFailure => ConnectionFailure,
        MiningEventType.AlgorithmSelected => AlgorithmSelected,
        MiningEventType.PoolChanged => PoolChanged,
        _ => throw new ArgumentOutOfRangeException(nameof(eventType), eventType, "Unknown mining event type."),
    };

    public static MiningEventType FromDbValue(string value) => value switch
    {
        MiningStarted => MiningEventType.MiningStarted,
        MiningPaused => MiningEventType.MiningPaused,
        MiningResumed => MiningEventType.MiningResumed,
        MiningStopped => MiningEventType.MiningStopped,
        ConnectionFailure => MiningEventType.ConnectionFailure,
        AlgorithmSelected => MiningEventType.AlgorithmSelected,
        PoolChanged => MiningEventType.PoolChanged,
        _ => throw new ArgumentOutOfRangeException(nameof(value), value, "Unknown mining event type value."),
    };
}
