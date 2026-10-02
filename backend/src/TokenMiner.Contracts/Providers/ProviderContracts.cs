namespace TokenMiner.Contracts.Providers;

public sealed record LlmProviderDepositAccountResponse(
    Guid Id,
    Guid CoinId,
    string CoinCode,
    string Network,
    string DepositAddress,
    string Status);

public sealed record LlmProviderResponse(
    Guid Id,
    string Name,
    string Endpoint,
    string? CredentialRef,
    string Status,
    decimal? Balance,
    decimal? ReservedBalance,
    decimal? TotalBalance,
    DateTimeOffset? BalanceCheckedAt,
    IReadOnlyList<LlmProviderDepositAccountResponse> DepositAccounts,
    DateTimeOffset CreatedAt,
    DateTimeOffset UpdatedAt);

public sealed record CreateLlmProviderRequest(string Name, string Endpoint, string? CredentialRef);

public sealed record UpdateLlmProviderRequest(
    string Name,
    string Endpoint,
    string? CredentialRef,
    string Status);

public sealed record UpsertDepositAccountRequest(
    Guid CoinId,
    string Network,
    string DepositAddress,
    string Status);

public sealed record LlmProviderModelResponse(
    Guid Id,
    string ModelId,
    string? Name,
    decimal? InputCost,
    decimal? OutputCost,
    decimal? CachedInputCost,
    string Currency,
    int? ContextLength,
    string? Capabilities,
    string Status,
    DateTimeOffset LastSyncedAt);

public sealed record ProviderDepositResponse(
    Guid Id,
    Guid LlmProviderId,
    Guid CoinId,
    string Network,
    string Address,
    decimal Amount,
    string? TransactionHash,
    int Confirmations,
    string Status,
    decimal? ProviderCreditBefore,
    decimal? ProviderCreditAfter,
    string IdempotencyKey,
    string? Error,
    DateTimeOffset CreatedAt,
    DateTimeOffset UpdatedAt,
    DateTimeOffset? ConfirmedAt);
