namespace TokenMiner.Domain.Mining.Enums;

/// <summary>
/// Reward lifecycle of a mining share. A share is recorded as <c>pending</c> when the pool
/// accepts it and is only redeemable once the resulting pool payout has been reconciled.
/// </summary>
public enum ShareRewardStatus
{
    Pending = 0,
    Processing = 1,
    Accepted = 2,
    Redeemed = 3,
    Rejected = 4,
}

public static class ShareRewardStatusDbValues
{
    public const string Pending = "pending";
    public const string Processing = "processing";
    public const string Accepted = "accepted";
    public const string Redeemed = "redeemed";
    public const string Rejected = "rejected";

    public static string ToDbValue(this ShareRewardStatus status) => status switch
    {
        ShareRewardStatus.Pending => Pending,
        ShareRewardStatus.Processing => Processing,
        ShareRewardStatus.Accepted => Accepted,
        ShareRewardStatus.Redeemed => Redeemed,
        ShareRewardStatus.Rejected => Rejected,
        _ => throw new ArgumentOutOfRangeException(nameof(status), status, "Unknown reward status."),
    };

    public static ShareRewardStatus FromDbValue(string value) => value switch
    {
        Pending => ShareRewardStatus.Pending,
        Processing => ShareRewardStatus.Processing,
        Accepted => ShareRewardStatus.Accepted,
        Redeemed => ShareRewardStatus.Redeemed,
        Rejected => ShareRewardStatus.Rejected,
        _ => throw new ArgumentOutOfRangeException(nameof(value), value, "Unknown reward status value."),
    };
}
