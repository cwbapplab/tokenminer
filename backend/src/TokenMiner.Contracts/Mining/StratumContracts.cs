namespace TokenMiner.Contracts.Mining;

/// <summary>A coin a pool accepts, as far as the stratum proxy is concerned.</summary>
public sealed record StratumCoinConfig(Guid CoinId, string Code);

/// <summary>Upstream routing information for one active pool.</summary>
public sealed record StratumPoolConfig(
    Guid PoolId,
    string SystemPoolId,
    string Name,
    string StratumEndpoint,
    string? PayoutAddress,
    IReadOnlyList<StratumCoinConfig> Coins);

/// <summary>
/// Everything the proxy needs to route stratum traffic. <paramref name="Version"/> changes
/// whenever any pool configuration changes, so a cached copy can be invalidated precisely.
/// </summary>
public sealed record StratumConfigResponse(string Version, IReadOnlyList<StratumPoolConfig> Pools);

/// <summary>Resolution of a worker identity into the pool/coin it is currently mining.</summary>
public sealed record StratumWorkerResponse(
    string WorkerId,
    Guid UserId,
    Guid UserHardwareId,
    Guid UserHardwareMinerId,
    Guid PoolId,
    string StratumEndpoint,
    string? PayoutAddress,
    Guid CoinId,
    string CoinCode);
