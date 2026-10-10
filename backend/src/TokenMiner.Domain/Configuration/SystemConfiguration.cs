namespace TokenMiner.Domain.Configuration;

/// <summary>
/// A single operator-tunable setting. Values are stored as JSON so a setting can be a scalar or
/// a structured value without a schema change.
/// </summary>
public sealed class SystemConfiguration
{
    private SystemConfiguration()
    {
    }

    public SystemConfiguration(string key, string value, DateTimeOffset now)
    {
        Key = key;
        Value = value;
        UpdatedAt = now;
    }

    public string Key { get; private set; } = null!;

    public string Value { get; private set; } = null!;

    public DateTimeOffset UpdatedAt { get; private set; }

    public void Update(string value, DateTimeOffset now)
    {
        Value = value;
        UpdatedAt = now;
    }
}

/// <summary>
/// The settings the runtime reads. Keys are stable identifiers, so renaming one is a breaking
/// change for anything already deployed.
/// </summary>
public static class SystemConfigurationKeys
{
    public const string ActiveLlmProviderId = "active_llm_provider_id";
    public const string ActiveConversionProviderId = "active_conversion_provider_id";
    public const string ActivePoolId = "active_pool_id";
    public const string PoolMonitorInterval = "pool_monitor_interval_seconds";
    public const string CoinPriceRefreshInterval = "coin_price_refresh_interval_seconds";
    public const string PayoutConfirmationThreshold = "payout_confirmation_threshold";
    public const string ProviderBalanceTarget = "provider_balance_target";
    public const string ProviderBalanceMinimum = "provider_balance_minimum";
    public const string MinimumConversionAmount = "minimum_conversion_amount";
    public const string MinimumPayoutAmount = "minimum_payout_amount";
    public const string AutoTopUpEnabled = "auto_top_up_enabled";
}
