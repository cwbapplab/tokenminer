namespace TokenMiner.Domain.Providers.Enums;

/// <summary>Lifecycle of a stablecoin transfer into an LLM provider's wallet.</summary>
public enum ProviderDepositStatus
{
    Pending = 0,
    Broadcast = 1,
    AwaitingConfirmations = 2,
    Confirmed = 3,
    Credited = 4,
    Failed = 5,
}

public static class ProviderDepositStatusDbValues
{
    public const string Pending = "pending";
    public const string Broadcast = "broadcast";
    public const string AwaitingConfirmations = "awaiting_confirmations";
    public const string Confirmed = "confirmed";
    public const string Credited = "credited";
    public const string Failed = "failed";

    public static string ToDbValue(this ProviderDepositStatus status) => status switch
    {
        ProviderDepositStatus.Pending => Pending,
        ProviderDepositStatus.Broadcast => Broadcast,
        ProviderDepositStatus.AwaitingConfirmations => AwaitingConfirmations,
        ProviderDepositStatus.Confirmed => Confirmed,
        ProviderDepositStatus.Credited => Credited,
        ProviderDepositStatus.Failed => Failed,
        _ => throw new ArgumentOutOfRangeException(nameof(status), status, "Unknown provider deposit status."),
    };

    public static ProviderDepositStatus FromDbValue(string value) => value switch
    {
        Pending => ProviderDepositStatus.Pending,
        Broadcast => ProviderDepositStatus.Broadcast,
        AwaitingConfirmations => ProviderDepositStatus.AwaitingConfirmations,
        Confirmed => ProviderDepositStatus.Confirmed,
        Credited => ProviderDepositStatus.Credited,
        Failed => ProviderDepositStatus.Failed,
        _ => throw new ArgumentOutOfRangeException(nameof(value), value, "Unknown provider deposit status value."),
    };

    /// <summary>Statuses that still need driving.</summary>
    public static readonly ProviderDepositStatus[] InFlight =
    [
        ProviderDepositStatus.Pending,
        ProviderDepositStatus.Broadcast,
        ProviderDepositStatus.AwaitingConfirmations,
        ProviderDepositStatus.Confirmed,
    ];
}
