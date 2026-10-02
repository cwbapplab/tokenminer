namespace TokenMiner.Contracts.Treasury;

public sealed record PoolPayoutResponse(
    Guid Id,
    Guid PoolId,
    Guid CoinId,
    string? WalletAddress,
    decimal Amount,
    decimal RedeemedAmount,
    string? TransactionHash,
    DateTimeOffset? RequestedAt,
    DateTimeOffset? ReceivedAt,
    int ConfirmationsCount,
    string Status,
    DateTimeOffset CreatedAt,
    DateTimeOffset UpdatedAt);

// --- Conversion configuration ------------------------------------------------------------

public sealed record CreateConversionProviderRequest(
    string Name,
    string BaseUrl,
    int Priority,
    string? CredentialRef,
    string? SupportedFeatures);

public sealed record UpdateConversionProviderRequest(
    string BaseUrl,
    int Priority,
    string Status,
    string? CredentialRef,
    string? SupportedFeatures);

public sealed record ConversionProviderResponse(
    Guid Id,
    string Name,
    string BaseUrl,
    string Status,
    int Priority,
    string? CredentialRef,
    string? SupportedFeatures,
    DateTimeOffset CreatedAt,
    DateTimeOffset UpdatedAt);

public sealed record CreateConversionRouteRequest(
    Guid ConversionProviderId,
    Guid SourceCoinId,
    Guid DestinationCoinId,
    string? SourceNetwork,
    string? DestinationNetwork,
    int Priority,
    decimal MinimumAmount);

public sealed record UpdateConversionRouteRequest(
    bool Enabled,
    int Priority,
    decimal MinimumAmount);

public sealed record ConversionRouteResponse(
    Guid Id,
    Guid ConversionProviderId,
    string ConversionProviderName,
    Guid SourceCoinId,
    string SourceCoinCode,
    Guid DestinationCoinId,
    string DestinationCoinCode,
    string? SourceNetwork,
    string? DestinationNetwork,
    bool Enabled,
    int Priority,
    decimal MinimumAmount,
    DateTimeOffset CreatedAt,
    DateTimeOffset UpdatedAt);

// --- Conversion ledger -------------------------------------------------------------------

public sealed record ConversionTransactionResponse(
    Guid Id,
    Guid ConversionProviderId,
    string ConversionProviderName,
    string SourceCoinCode,
    decimal SourceAmount,
    string DestinationCoinCode,
    decimal? DestinationAmount,
    decimal? ExchangeRate,
    decimal? Fees,
    string? SourceTransactionId,
    string? DestinationTransactionId,
    string Status,
    string IdempotencyKey,
    string? Error,
    DateTimeOffset CreatedAt,
    DateTimeOffset UpdatedAt,
    DateTimeOffset? CompletedAt);

/// <summary><paramref name="Outcome"/> is <c>completed</c>, <c>failed</c> or <c>cancelled</c>.</summary>
public sealed record SettleConversionRequest(
    string Outcome,
    decimal? DestinationAmount,
    decimal? ExchangeRate,
    decimal? Fees,
    string? DestinationTransactionId,
    string? Reason);
