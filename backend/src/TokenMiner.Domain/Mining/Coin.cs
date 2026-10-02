using TokenMiner.Domain.Mining.Enums;

namespace TokenMiner.Domain.Mining;

/// <summary>
/// A cryptocurrency that can participate in the mining and payout pipeline. <see cref="Code"/>
/// is the stable, human-facing identifier (for example <c>prl</c> or <c>qtc</c>).
/// </summary>
public sealed class Coin
{
    private Coin()
    {
    }

    public Coin(Guid id, string code, string name, string network, int decimals, DateTimeOffset now)
    {
        Id = id;
        Code = code;
        Name = name;
        Network = network;
        Decimals = decimals;
        Status = MiningStatus.Active;
        CreatedAt = now;
        UpdatedAt = now;
    }

    public Guid Id { get; private set; }

    public string Code { get; private set; } = null!;

    public string Name { get; private set; } = null!;

    public string Network { get; private set; } = null!;

    public int Decimals { get; private set; }

    /// <summary>Latest observed price, used for estimation snapshots.</summary>
    public decimal? LastKnownUsdValue { get; private set; }

    public DateTimeOffset? LastValueDate { get; private set; }

    public MiningStatus Status { get; private set; }

    public DateTimeOffset CreatedAt { get; private set; }

    public DateTimeOffset UpdatedAt { get; private set; }

    public void UpdateDetails(string name, string network, int decimals, MiningStatus status, DateTimeOffset now)
    {
        Name = name;
        Network = network;
        Decimals = decimals;
        Status = status;
        UpdatedAt = now;
    }

    /// <summary>Refreshes the cached valuation. Historical points live in <see cref="CoinPriceHistory"/>.</summary>
    public void RecordPrice(decimal usdValue, DateTimeOffset observedAt, DateTimeOffset now)
    {
        LastKnownUsdValue = usdValue;
        LastValueDate = observedAt;
        UpdatedAt = now;
    }
}
