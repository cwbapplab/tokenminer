namespace TokenMiner.Domain.Treasury.Enums;

/// <summary>Lifecycle of a conversion from a mined coin into a stablecoin.</summary>
public enum ConversionTransactionStatus
{
    Pending = 0,
    Processing = 1,
    Completed = 2,
    Failed = 3,
    Cancelled = 4,
}

public static class ConversionTransactionStatusDbValues
{
    public const string Pending = "pending";
    public const string Processing = "processing";
    public const string Completed = "completed";
    public const string Failed = "failed";
    public const string Cancelled = "cancelled";

    public static string ToDbValue(this ConversionTransactionStatus status) => status switch
    {
        ConversionTransactionStatus.Pending => Pending,
        ConversionTransactionStatus.Processing => Processing,
        ConversionTransactionStatus.Completed => Completed,
        ConversionTransactionStatus.Failed => Failed,
        ConversionTransactionStatus.Cancelled => Cancelled,
        _ => throw new ArgumentOutOfRangeException(nameof(status), status, "Unknown conversion status."),
    };

    public static ConversionTransactionStatus FromDbValue(string value) => value switch
    {
        Pending => ConversionTransactionStatus.Pending,
        Processing => ConversionTransactionStatus.Processing,
        Completed => ConversionTransactionStatus.Completed,
        Failed => ConversionTransactionStatus.Failed,
        Cancelled => ConversionTransactionStatus.Cancelled,
        _ => throw new ArgumentOutOfRangeException(nameof(value), value, "Unknown conversion status value."),
    };

    /// <summary>Statuses that still need driving.</summary>
    public static readonly ConversionTransactionStatus[] InFlight =
        [ConversionTransactionStatus.Pending, ConversionTransactionStatus.Processing];
}

/// <summary>Enablement state of treasury configuration records.</summary>
public enum TreasuryStatus
{
    Active = 0,
    Disabled = 1,
}

public static class TreasuryStatusDbValues
{
    public const string Active = "active";
    public const string Disabled = "disabled";

    public static string ToDbValue(this TreasuryStatus status) => status switch
    {
        TreasuryStatus.Active => Active,
        TreasuryStatus.Disabled => Disabled,
        _ => throw new ArgumentOutOfRangeException(nameof(status), status, "Unknown treasury status."),
    };

    public static TreasuryStatus FromDbValue(string value) => value switch
    {
        Active => TreasuryStatus.Active,
        Disabled => TreasuryStatus.Disabled,
        _ => throw new ArgumentOutOfRangeException(nameof(value), value, "Unknown treasury status value."),
    };

    public static bool TryParseDbValue(string? value, out TreasuryStatus status)
    {
        switch (value)
        {
            case Active:
                status = TreasuryStatus.Active;
                return true;
            case Disabled:
                status = TreasuryStatus.Disabled;
                return true;
            default:
                status = TreasuryStatus.Active;
                return false;
        }
    }
}
