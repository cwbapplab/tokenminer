using TokenMiner.Domain.Mining.Enums;

namespace TokenMiner.Domain.Mining;

/// <summary>
/// Materialised mining totals for one device, coin and rolling window. Analytics reads these
/// rows rather than aggregating raw shares on every request.
/// </summary>
public sealed class MiningStatistic
{
    private MiningStatistic()
    {
    }

    public MiningStatistic(
        Guid id,
        Guid userId,
        Guid userHardwareId,
        Guid coinId,
        StatisticPeriod period,
        decimal amount,
        decimal usdValue,
        DateTimeOffset calculatedAt)
    {
        Id = id;
        UserId = userId;
        UserHardwareId = userHardwareId;
        CoinId = coinId;
        Period = period;
        Amount = amount;
        UsdValue = usdValue;
        CalculatedAt = calculatedAt;
    }

    public Guid Id { get; private set; }

    public Guid UserId { get; private set; }

    public Guid UserHardwareId { get; private set; }

    public Guid CoinId { get; private set; }

    public StatisticPeriod Period { get; private set; }

    /// <summary>Sum of mined coin amounts.</summary>
    public decimal Amount { get; private set; }

    /// <summary>Sum of USD valuations using each share's price snapshot.</summary>
    public decimal UsdValue { get; private set; }

    public DateTimeOffset CalculatedAt { get; private set; }

    public void Update(decimal amount, decimal usdValue, DateTimeOffset calculatedAt)
    {
        Amount = amount;
        UsdValue = usdValue;
        CalculatedAt = calculatedAt;
    }
}
