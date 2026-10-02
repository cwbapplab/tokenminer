namespace TokenMiner.Domain.Mining.Enums;

/// <summary>Lifecycle state shared by the mining catalogue aggregates.</summary>
public enum MiningStatus
{
    Active = 0,
    Disabled = 1,
}

/// <summary>snake_case database representation, e.g. <c>disabled</c>.</summary>
public static class MiningStatusDbValues
{
    public const string Active = "active";
    public const string Disabled = "disabled";

    public static string ToDbValue(this MiningStatus status) => status switch
    {
        MiningStatus.Active => Active,
        MiningStatus.Disabled => Disabled,
        _ => throw new ArgumentOutOfRangeException(nameof(status), status, "Unknown mining status."),
    };

    public static MiningStatus FromDbValue(string value) => value switch
    {
        Active => MiningStatus.Active,
        Disabled => MiningStatus.Disabled,
        _ => throw new ArgumentOutOfRangeException(nameof(value), value, "Unknown mining status value."),
    };

    /// <summary>Non-throwing parse, for validating wire input.</summary>
    public static bool TryParseDbValue(string? value, out MiningStatus status)
    {
        switch (value)
        {
            case Active:
                status = MiningStatus.Active;
                return true;
            case Disabled:
                status = MiningStatus.Disabled;
                return true;
            default:
                status = MiningStatus.Active;
                return false;
        }
    }
}
