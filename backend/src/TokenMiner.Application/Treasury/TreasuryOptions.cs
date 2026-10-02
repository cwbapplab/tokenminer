namespace TokenMiner.Application.Treasury;

/// <summary>Bound from the <c>Treasury</c> configuration section.</summary>
public sealed class TreasuryOptions
{
    public const string SectionName = "Treasury";

    /// <summary>How often pools are polled for balances and payout history.</summary>
    public int PoolMonitorIntervalSeconds { get; set; } = 300;

    /// <summary>How often payout confirmations are refreshed and shares attributed.</summary>
    public int PayoutReconciliationIntervalSeconds { get; set; } = 300;

    /// <summary>How often coin prices are refreshed.</summary>
    public int CoinPriceIntervalSeconds { get; set; } = 900;

    /// <summary>Fallback withdrawal floor when a pool does not report its own threshold.</summary>
    public decimal MinimumPayoutAmount { get; set; } = 1m;

    /// <summary>Confirmations required before a payout is treated as settled.</summary>
    public int PayoutConfirmationThreshold { get; set; } = 6;

    /// <summary>
    /// Enables the browser-automation withdrawal fallback. Off by default: it drives a real
    /// financial action through a web UI and needs Playwright browsers installed.
    /// </summary>
    public bool EnableBrowserWithdrawals { get; set; }

    /// <summary>Base URL used to fetch coin prices when a pool does not supply one.</summary>
    public string PriceSourceBaseUrl { get; set; } = "https://pool.kryptex.com";

    /// <summary>How often the conversion pipeline runs.</summary>
    public int ConversionIntervalSeconds { get; set; } = 120;

    /// <summary>How many in-flight conversions one pass drives.</summary>
    public int ConversionBatchSize { get; set; } = 50;

    /// <summary>Stablecoins per mined coin, used by the simulated conversion provider.</summary>
    public decimal SimulatedConversionRate { get; set; } = 1m;

    /// <summary>Fraction taken as fees by the simulated conversion provider.</summary>
    public decimal SimulatedConversionFeeRate { get; set; }

    /// <summary>Payout request page opened by the browser fallback.</summary>
    public string WithdrawalPageUrl { get; set; } = string.Empty;

    /// <summary>Destination wallet the browser fallback fills in.</summary>
    public string WithdrawalAddress { get; set; } = string.Empty;

    public string WithdrawalAmountSelector { get; set; } = "input[name='amount']";

    public string WithdrawalAddressSelector { get; set; } = "input[name='address']";

    public string WithdrawalSubmitSelector { get; set; } = "button[type='submit']";

    public int BrowserTimeoutSeconds { get; set; } = 30;
}
