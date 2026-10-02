using TokenMiner.Domain.Providers;

namespace TokenMiner.Application.Providers.Abstractions;

/// <summary>What a provider client needs in order to call the gateway.</summary>
public sealed record LlmProviderContext(string Name, string Endpoint, string? ApiKey);

public sealed record LlmBalance(decimal Balance, decimal ReservedBalance, decimal TotalBalance);

public sealed record LlmCatalogModel(
    string ModelId,
    string? Name,
    decimal? InputCost,
    decimal? OutputCost,
    decimal? CachedInputCost,
    string Currency,
    int? ContextLength,
    IReadOnlyList<string> Capabilities);

/// <summary>An LLM gateway: its prepaid balance and its model catalogue.</summary>
public interface ILlmProviderClient
{
    /// <summary>Matches <c>llm_providers.name</c>, e.g. <c>router-one</c>.</summary>
    string Name { get; }

    Task<LlmBalance> GetBalanceAsync(LlmProviderContext context, CancellationToken cancellationToken);

    Task<IReadOnlyList<LlmCatalogModel>> GetModelsAsync(
        LlmProviderContext context,
        CancellationToken cancellationToken);
}

/// <summary>Resolves a credential reference into the secret itself.</summary>
/// <remarks>
/// The shipped implementation reads from configuration. A vault or key-management adapter
/// replaces it without touching any caller, which is the point of the indirection.
/// </remarks>
public interface ISecretResolver
{
    Task<string?> GetSecretAsync(string credentialRef, CancellationToken cancellationToken);
}

// --- Blockchain ---------------------------------------------------------------------------

public sealed record BlockchainTransferRequest(
    Guid DepositId,
    string IdempotencyKey,
    string Network,
    string ToAddress,
    string CoinCode,
    decimal Amount);

public sealed record BlockchainTransferResult(string? TransactionHash, string? Error);

/// <summary>
/// Sends a stablecoin transfer and reports its confirmations.
/// </summary>
/// <remarks>
/// Kept behind a port because moving funds needs a funded hot wallet and a key-custody
/// decision, neither of which belongs in this code base's configuration
/// </remarks>
public interface IBlockchainTransferProvider
{
    string Name { get; }

    Task<BlockchainTransferResult> SendAsync(
        BlockchainTransferRequest request,
        CancellationToken cancellationToken);

    Task<int> GetConfirmationsAsync(
        string network,
        string transactionHash,
        CancellationToken cancellationToken);
}

// --- Repositories -------------------------------------------------------------------------

public interface ILlmProviderRepository
{
    void AddProvider(LlmProvider provider);

    void AddDepositAccount(LlmProviderDepositAccount account);

    void AddModel(LlmProviderModel model);

    void AddBalanceSnapshot(ProviderBalanceSnapshot snapshot);

    Task<LlmProvider?> GetProviderByIdAsync(Guid id, CancellationToken cancellationToken);

    Task<LlmProvider?> GetProviderByNameAsync(string name, CancellationToken cancellationToken);

    Task<IReadOnlyList<LlmProvider>> ListProvidersAsync(CancellationToken cancellationToken);

    Task<LlmProviderDepositAccount?> GetDepositAccountAsync(
        Guid llmProviderId,
        Guid coinId,
        string network,
        CancellationToken cancellationToken);

    Task<IReadOnlyList<LlmProviderDepositAccount>> ListDepositAccountsAsync(CancellationToken cancellationToken);

    Task<IReadOnlyList<LlmProviderModel>> ListModelsAsync(Guid llmProviderId, CancellationToken cancellationToken);

    Task<ProviderBalanceSnapshot?> GetLatestBalanceAsync(Guid llmProviderId, CancellationToken cancellationToken);
}

public interface IProviderDepositRepository
{
    void Add(ProviderDeposit deposit);

    Task<ProviderDeposit?> GetByIdempotencyKeyAsync(string idempotencyKey, CancellationToken cancellationToken);

    Task<IReadOnlyList<ProviderDeposit>> ListInFlightAsync(CancellationToken cancellationToken);

    Task<int> CountInFlightAsync(CancellationToken cancellationToken);

    Task<IReadOnlyList<ProviderDeposit>> ListRecentAsync(int limit, CancellationToken cancellationToken);
}
