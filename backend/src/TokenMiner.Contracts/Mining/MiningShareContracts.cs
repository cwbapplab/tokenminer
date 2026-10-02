using System.Text.Json;

namespace TokenMiner.Contracts.Mining;

/// <summary>
/// An accepted share reported by the stratum proxy. Only that service may post this, and it
/// carries the pool-side evidence so the share can be re-validated later.
/// </summary>
public sealed record MiningShareRequest(
    Guid UserId,
    Guid UserHardwareId,
    Guid PoolId,
    Guid CoinId,
    string ShareIdentifier,
    string? JobId,
    string? Nonce,
    string? Extranonce,
    decimal? Difficulty,
    string? Target,
    string? Hash,
    DateTimeOffset Timestamp,
    decimal? CoinValue,
    JsonElement? PoolResponse);

public sealed record MiningShareResponse(
    Guid Id,
    string ShareIdentifier,
    string RewardStatus,
    decimal? CoinValue,
    decimal? ApproxUsdValueAtTime,
    bool AlreadyRecorded,
    DateTimeOffset CreatedAt);

// --- Analytics ---------------------------------------------------------------------------

public sealed record HardwareAnalyticsResponse(
    Guid UserHardwareId,
    decimal Last30Usd,
    decimal Last60Usd,
    decimal AllTimeUsd);

public sealed record AnalyticsTotalsResponse(decimal Last30Usd, decimal Last60Usd, decimal AllTimeUsd);

public sealed record MiningAnalyticsResponse(
    IReadOnlyList<HardwareAnalyticsResponse> Hardware,
    AnalyticsTotalsResponse Totals);
