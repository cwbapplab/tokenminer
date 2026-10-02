using TokenMiner.Domain.Mining.Enums;

namespace TokenMiner.Domain.Mining;

/// <summary>
/// Connection and payout configuration for a pool. One row per pool; <see cref="CoinId"/>
/// names the pool's default coin, while the full supported set lives in <see cref="PoolCoin"/>.
/// </summary>
public sealed class PoolDetails
{
    private PoolDetails()
    {
    }

    public PoolDetails(
        Guid id,
        Guid poolId,
        string baseUrl,
        string? statusEndpoint,
        string? stratumEndpoint,
        Guid coinId,
        PayoutMode payoutMode,
        string? payoutAddress,
        string? payoutNetwork,
        DateTimeOffset now)
    {
        Id = id;
        PoolId = poolId;
        BaseUrl = baseUrl;
        StatusEndpoint = statusEndpoint;
        StratumEndpoint = stratumEndpoint;
        CoinId = coinId;
        PayoutMode = payoutMode;
        PayoutAddress = payoutAddress;
        PayoutNetwork = payoutNetwork;
        CreatedAt = now;
        UpdatedAt = now;
    }

    public Guid Id { get; private set; }

    public Guid PoolId { get; private set; }

    public string BaseUrl { get; private set; } = null!;

    public string? StatusEndpoint { get; private set; }

    /// <summary>Upstream stratum endpoint (<c>host:port</c>) the proxy connects to.</summary>
    public string? StratumEndpoint { get; private set; }

    public Guid CoinId { get; private set; }

    public PayoutMode PayoutMode { get; private set; }

    /// <summary>Wallet the pool pays into. Required unless the mode is <c>managed_request</c>.</summary>
    public string? PayoutAddress { get; private set; }

    public string? PayoutNetwork { get; private set; }

    public DateTimeOffset CreatedAt { get; private set; }

    public DateTimeOffset UpdatedAt { get; private set; }

    public void Update(
        string baseUrl,
        string? statusEndpoint,
        string? stratumEndpoint,
        Guid coinId,
        PayoutMode payoutMode,
        string? payoutAddress,
        string? payoutNetwork,
        DateTimeOffset now)
    {
        BaseUrl = baseUrl;
        StatusEndpoint = statusEndpoint;
        StratumEndpoint = stratumEndpoint;
        CoinId = coinId;
        PayoutMode = payoutMode;
        PayoutAddress = payoutAddress;
        PayoutNetwork = payoutNetwork;
        UpdatedAt = now;
    }
}
