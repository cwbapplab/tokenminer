namespace TokenMiner.Domain.Mining;

/// <summary>
/// Append-only observation of a coin's USD price. Never updated or deleted, so historical
/// mining valuations stay reproducible.
/// </summary>
public sealed class CoinPriceHistory
{
    private CoinPriceHistory()
    {
    }

    public CoinPriceHistory(Guid id, Guid coinId, decimal usdValue, DateTimeOffset observedAt, string source)
    {
        Id = id;
        CoinId = coinId;
        UsdValue = usdValue;
        ObservedAt = observedAt;
        Source = source;
    }

    public Guid Id { get; private set; }

    public Guid CoinId { get; private set; }

    public decimal UsdValue { get; private set; }

    public DateTimeOffset ObservedAt { get; private set; }

    public string Source { get; private set; } = null!;
}
