using System.Text.Json;

namespace TokenMiner.Contracts.Mining;

// --- Coins -------------------------------------------------------------------------------

public sealed record CoinResponse(
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

public sealed record CreateCoinRequest(string Code, string Name, string Network, int Decimals);

public sealed record UpdateCoinRequest(string Name, string Network, int Decimals, string Status);

// --- Pools -------------------------------------------------------------------------------

/// <summary>
/// <paramref name="PayoutMode"/> is one of <c>managed_request</c>, <c>native_auto_payout</c>
/// or <c>direct_to_wallet</c>.
/// </summary>
public sealed record PoolDetailsRequest(
    string BaseUrl,
    string? StatusEndpoint,
    string? StratumEndpoint,
    Guid CoinId,
    string PayoutMode,
    string? PayoutAddress,
    string? PayoutNetwork);

/// <summary>
/// <paramref name="Provider"/> names the payout integration (e.g. <c>kryptex</c>) and defaults
/// to <c>kryptex</c> so existing callers need not supply it.
/// </summary>
public sealed record CreatePoolRequest(
    string SystemPoolId,
    string Name,
    PoolDetailsRequest Details,
    IReadOnlyList<Guid> CoinIds,
    string Provider = "kryptex");

/// <summary><paramref name="Provider"/> defaults to <c>kryptex</c> when omitted.</summary>
public sealed record UpdatePoolRequest(
    string Name,
    string Status,
    PoolDetailsRequest Details,
    IReadOnlyList<Guid> CoinIds,
    string Provider = "kryptex");

public sealed record PoolDetailsResponse(
    string BaseUrl,
    string? StatusEndpoint,
    string? StratumEndpoint,
    Guid CoinId,
    string PayoutMode,
    string? PayoutAddress,
    string? PayoutNetwork);

public sealed record PoolResponse(
    Guid Id,
    string SystemPoolId,
    string Name,
    string Provider,
    string Status,
    PoolDetailsResponse? Details,
    IReadOnlyList<Guid> CoinIds,
    DateTimeOffset CreatedAt,
    DateTimeOffset UpdatedAt);

// --- Mining algorithms -------------------------------------------------------------------

public sealed record MiningAlgoResponse(
    Guid Id,
    string Code,
    string Name,
    string Status,
    int Priority,
    JsonElement? Configuration,
    DateTimeOffset CreatedAt,
    DateTimeOffset UpdatedAt);

public sealed record CreateMiningAlgoRequest(
    string Code,
    string Name,
    int Priority,
    JsonElement? Configuration);

public sealed record UpdateMiningAlgoRequest(
    string Name,
    int Priority,
    string Status,
    JsonElement? Configuration);
