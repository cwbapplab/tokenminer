namespace TokenMiner.Application.Providers;

/// <summary>Bound from the <c>Providers</c> configuration section.</summary>
public sealed class ProviderOptions
{
    public const string SectionName = "Providers";

    /// <summary>How often provider balances and the deposit pipeline are refreshed.</summary>
    public int BalanceIntervalSeconds { get; set; } = 180;

    /// <summary>How often the model catalogue is mirrored.</summary>
    public int ModelCatalogIntervalSeconds { get; set; } = 86400;

    /// <summary>How often the deposit pipeline runs.</summary>
    public int DepositIntervalSeconds { get; set; } = 120;

    /// <summary>How often the auto top-up state machine ticks.</summary>
    public int TopUpIntervalSeconds { get; set; } = 60;

    /// <summary>How many deposits one pass advances.</summary>
    public int DepositBatchSize { get; set; } = 25;

    /// <summary>Confirmations required before a deposit is treated as settled on chain.</summary>
    public int ConfirmationThreshold { get; set; } = 6;

    /// <summary>
    /// Top-up policy: the balance the system aims for, and the level that triggers a top-up.
    /// </summary>
    public decimal BalanceTarget { get; set; } = 20m;

    public decimal BalanceMinimum { get; set; } = 5m;

    /// <summary>Smallest conversion worth performing.</summary>
    public decimal MinimumConversionAmount { get; set; } = 1m;

    /// <summary>Smallest pool payout worth requesting.</summary>
    public decimal MinimumPayoutAmount { get; set; } = 1m;

    /// <summary>
    /// Confirmations the simulated chain reports, so the deposit lifecycle can be exercised
    /// without a network. Raise it to at or above <see cref="ConfirmationThreshold"/> to settle.
    /// </summary>
    public int SimulatedConfirmations { get; set; }
}
