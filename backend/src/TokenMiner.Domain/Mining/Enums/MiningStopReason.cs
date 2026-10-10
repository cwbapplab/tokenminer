namespace TokenMiner.Domain.Mining.Enums;

/// <summary>
/// Why a mining session stopped. These values are part of the public API
/// contract, so they use the documented hyphenated spelling rather than snake_case.
/// </summary>
public enum MiningStopReason
{
    ConnectionFailure = 0,
    DriverCrash = 1,
    MinerClosed = 2,
    UserTriggered = 3,
    Unknown = 4,
}

public static class MiningStopReasonDbValues
{
    public const string ConnectionFailure = "connection-failure";
    public const string DriverCrash = "driver-crash";
    public const string MinerClosed = "miner-closed";
    public const string UserTriggered = "user-triggered";
    public const string Unknown = "unknown";

    public static string ToDbValue(this MiningStopReason reason) => reason switch
    {
        MiningStopReason.ConnectionFailure => ConnectionFailure,
        MiningStopReason.DriverCrash => DriverCrash,
        MiningStopReason.MinerClosed => MinerClosed,
        MiningStopReason.UserTriggered => UserTriggered,
        MiningStopReason.Unknown => Unknown,
        _ => throw new ArgumentOutOfRangeException(nameof(reason), reason, "Unknown stop reason."),
    };

    public static MiningStopReason FromDbValue(string value) => value switch
    {
        ConnectionFailure => MiningStopReason.ConnectionFailure,
        DriverCrash => MiningStopReason.DriverCrash,
        MinerClosed => MiningStopReason.MinerClosed,
        UserTriggered => MiningStopReason.UserTriggered,
        Unknown => MiningStopReason.Unknown,
        _ => throw new ArgumentOutOfRangeException(nameof(value), value, "Unknown stop reason value."),
    };

    /// <summary>Non-throwing parse, for validating wire input.</summary>
    public static bool TryParseDbValue(string? value, out MiningStopReason reason)
    {
        switch (value)
        {
            case ConnectionFailure:
                reason = MiningStopReason.ConnectionFailure;
                return true;
            case DriverCrash:
                reason = MiningStopReason.DriverCrash;
                return true;
            case MinerClosed:
                reason = MiningStopReason.MinerClosed;
                return true;
            case UserTriggered:
                reason = MiningStopReason.UserTriggered;
                return true;
            case Unknown:
                reason = MiningStopReason.Unknown;
                return true;
            default:
                reason = MiningStopReason.Unknown;
                return false;
        }
    }
}
