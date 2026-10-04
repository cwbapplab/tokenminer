using System.Text.Json;
using TokenMiner.Domain.Mining;
using TokenMiner.Domain.Mining.Enums;

namespace TokenMiner.Application.Mining.Models;

public sealed record CoinDto(
    Guid Id,
    string Code,
    string Name,
    string Network,
    int Decimals,
    decimal? LastKnownUsdValue,
    DateTimeOffset? LastValueDate,
    string Status,
    DateTimeOffset CreatedAt,
    DateTimeOffset UpdatedAt);

public sealed record PoolDetailsDto(
    string BaseUrl,
    string? StatusEndpoint,
    string? StratumEndpoint,
    Guid CoinId,
    string PayoutMode,
    string? PayoutAddress,
    string? PayoutNetwork);

public sealed record PoolDto(
    Guid Id,
    string SystemPoolId,
    string Name,
    string Provider,
    string Status,
    PoolDetailsDto? Details,
    IReadOnlyList<Guid> CoinIds,
    DateTimeOffset CreatedAt,
    DateTimeOffset UpdatedAt);

public sealed record MiningAlgoDto(
    Guid Id,
    string Code,
    string Name,
    string Status,
    int Priority,
    string? Configuration,
    DateTimeOffset CreatedAt,
    DateTimeOffset UpdatedAt);

public sealed record MiningSessionDto(
    Guid SessionId,
    Guid PoolId,
    string PoolName,
    Guid CoinId,
    string CoinCode,
    Guid AlgorithmId,
    string AlgorithmCode,
    string WorkerId,
    string MinerCommand,
    /// <summary>The rendered algorithm configuration (algo/endpoint/wallet/workerId/coin).</summary>
    JsonElement MinerConfig,
    string StratumEndpoint,
    string Status,
    DateTimeOffset StartedAt);

public static class MiningMappings
{
    public static CoinDto ToDto(this Coin coin) => new(
        coin.Id,
        coin.Code,
        coin.Name,
        coin.Network,
        coin.Decimals,
        coin.LastKnownUsdValue,
        coin.LastValueDate,
        coin.Status.ToDbValue(),
        coin.CreatedAt,
        coin.UpdatedAt);

    public static PoolDetailsDto ToDto(this PoolDetails details) => new(
        details.BaseUrl,
        details.StatusEndpoint,
        details.StratumEndpoint,
        details.CoinId,
        details.PayoutMode.ToDbValue(),
        details.PayoutAddress,
        details.PayoutNetwork);

    public static PoolDto ToDto(this Pool pool, PoolDetails? details, IReadOnlyList<Guid> coinIds) => new(
        pool.Id,
        pool.SystemPoolId,
        pool.Name,
        pool.Provider,
        pool.Status.ToDbValue(),
        details?.ToDto(),
        coinIds,
        pool.CreatedAt,
        pool.UpdatedAt);

    public static MiningAlgoDto ToDto(this MiningAlgo algo) => new(
        algo.Id,
        algo.Code,
        algo.Name,
        algo.Status.ToDbValue(),
        algo.Priority,
        algo.Configuration,
        algo.CreatedAt,
        algo.UpdatedAt);
}
