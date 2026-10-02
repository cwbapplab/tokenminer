namespace TokenMiner.Domain.Mining.Enums;

/// <summary>Lifecycle of a single mining session on one device.</summary>
public enum MiningSessionStatus
{
    Running = 0,
    Paused = 1,
    Stopped = 2,
}

public static class MiningSessionStatusDbValues
{
    public const string Running = "running";
    public const string Paused = "paused";
    public const string Stopped = "stopped";

    /// <summary>Statuses that count as an in-flight session, i.e. an active one.</summary>
    public static readonly string[] Active = [Running, Paused];

    public static string ToDbValue(this MiningSessionStatus status) => status switch
    {
        MiningSessionStatus.Running => Running,
        MiningSessionStatus.Paused => Paused,
        MiningSessionStatus.Stopped => Stopped,
        _ => throw new ArgumentOutOfRangeException(nameof(status), status, "Unknown session status."),
    };

    public static MiningSessionStatus FromDbValue(string value) => value switch
    {
        Running => MiningSessionStatus.Running,
        Paused => MiningSessionStatus.Paused,
        Stopped => MiningSessionStatus.Stopped,
        _ => throw new ArgumentOutOfRangeException(nameof(value), value, "Unknown session status value."),
    };
}
