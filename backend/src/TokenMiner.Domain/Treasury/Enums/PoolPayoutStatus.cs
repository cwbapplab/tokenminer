namespace TokenMiner.Domain.Treasury.Enums;

/// <summary>Lifecycle of a payout from a mining pool.</summary>
public enum PoolPayoutStatus
{
    Pending = 0,
    Processing = 1,
    AwaitingConfirmations = 2,
    Confirmed = 3,
    Failed = 4,
}

public static class PoolPayoutStatusDbValues
{
    public const string Pending = "pending";
    public const string Processing = "processing";
    public const string AwaitingConfirmations = "awaiting_confirmations";
    public const string Confirmed = "confirmed";
    public const string Failed = "failed";

    public static string ToDbValue(this PoolPayoutStatus status) => status switch
    {
        PoolPayoutStatus.Pending => Pending,
        PoolPayoutStatus.Processing => Processing,
        PoolPayoutStatus.AwaitingConfirmations => AwaitingConfirmations,
        PoolPayoutStatus.Confirmed => Confirmed,
        PoolPayoutStatus.Failed => Failed,
        _ => throw new ArgumentOutOfRangeException(nameof(status), status, "Unknown pool payout status."),
    };

    public static PoolPayoutStatus FromDbValue(string value) => value switch
    {
        Pending => PoolPayoutStatus.Pending,
        Processing => PoolPayoutStatus.Processing,
        AwaitingConfirmations => PoolPayoutStatus.AwaitingConfirmations,
        Confirmed => PoolPayoutStatus.Confirmed,
        Failed => PoolPayoutStatus.Failed,
        _ => throw new ArgumentOutOfRangeException(nameof(value), value, "Unknown pool payout status value."),
    };

    /// <summary>Statuses that still need watching.</summary>
    public static readonly PoolPayoutStatus[] Unsettled =
        [PoolPayoutStatus.Pending, PoolPayoutStatus.Processing, PoolPayoutStatus.AwaitingConfirmations];
}
