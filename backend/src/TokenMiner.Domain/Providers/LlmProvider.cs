using TokenMiner.Domain.Mining.Enums;
using TokenMiner.Domain.Providers.Enums;

namespace TokenMiner.Domain.Providers;

/// <summary>An LLM gateway whose prepaid balance the system keeps topped up.</summary>
public sealed class LlmProvider
{
    private LlmProvider()
    {
    }

    public LlmProvider(Guid id, string name, string endpoint, string? credentialRef, DateTimeOffset now)
    {
        Id = id;
        Name = name;
        Endpoint = endpoint;
        CredentialRef = credentialRef;
        Status = MiningStatus.Active;
        CreatedAt = now;
        UpdatedAt = now;
    }

    public Guid Id { get; private set; }

    public string Name { get; private set; } = null!;

    /// <summary>OpenAI-compatible base URL, e.g. <c>https://api.router.one/v1</c>.</summary>
    public string Endpoint { get; private set; } = null!;

    /// <summary>Key into the secret resolver; never the API key itself.</summary>
    public string? CredentialRef { get; private set; }

    public MiningStatus Status { get; private set; }

    public DateTimeOffset CreatedAt { get; private set; }

    public DateTimeOffset UpdatedAt { get; private set; }

    public void Update(string name, string endpoint, string? credentialRef, MiningStatus status, DateTimeOffset now)
    {
        Name = name;
        Endpoint = endpoint;
        CredentialRef = credentialRef;
        Status = status;
        UpdatedAt = now;
    }
}

/// <summary>
/// The deposit rail for one coin and network. A provider's USDT address on BSC is a different
/// rail from its USDT address on Tron, so the pair is part of the identity.
/// </summary>
public sealed class LlmProviderDepositAccount
{
    private LlmProviderDepositAccount()
    {
    }

    public LlmProviderDepositAccount(
        Guid id,
        Guid llmProviderId,
        Guid coinId,
        string network,
        string depositAddress,
        DateTimeOffset now)
    {
        Id = id;
        LlmProviderId = llmProviderId;
        CoinId = coinId;
        Network = network;
        DepositAddress = depositAddress;
        Status = MiningStatus.Active;
        CreatedAt = now;
        UpdatedAt = now;
    }

    public Guid Id { get; private set; }

    public Guid LlmProviderId { get; private set; }

    public Guid CoinId { get; private set; }

    public string Network { get; private set; } = null!;

    public string DepositAddress { get; private set; } = null!;

    public MiningStatus Status { get; private set; }

    public DateTimeOffset CreatedAt { get; private set; }

    public DateTimeOffset UpdatedAt { get; private set; }

    public void Update(string depositAddress, MiningStatus status, DateTimeOffset now)
    {
        DepositAddress = depositAddress;
        Status = status;
        UpdatedAt = now;
    }
}

/// <summary>A model offered by a provider, with its posted prices, synced from the catalogue.</summary>
public sealed class LlmProviderModel
{
    private LlmProviderModel()
    {
    }

    public LlmProviderModel(
        Guid id,
        Guid llmProviderId,
        string modelId,
        string? name,
        decimal? inputCost,
        decimal? outputCost,
        decimal? cachedInputCost,
        string currency,
        int? contextLength,
        string? capabilities,
        DateTimeOffset now)
    {
        Id = id;
        LlmProviderId = llmProviderId;
        ModelId = modelId;
        Name = name;
        InputCost = inputCost;
        OutputCost = outputCost;
        CachedInputCost = cachedInputCost;
        Currency = currency;
        ContextLength = contextLength;
        Capabilities = capabilities;
        Status = MiningStatus.Active;
        LastSyncedAt = now;
        CreatedAt = now;
        UpdatedAt = now;
    }

    public Guid Id { get; private set; }

    public Guid LlmProviderId { get; private set; }

    public string ModelId { get; private set; } = null!;

    public string? Name { get; private set; }

    public decimal? InputCost { get; private set; }

    public decimal? OutputCost { get; private set; }

    public decimal? CachedInputCost { get; private set; }

    public string Currency { get; private set; } = "USD";

    public int? ContextLength { get; private set; }

    public string? Capabilities { get; private set; }

    public MiningStatus Status { get; private set; }

    public DateTimeOffset LastSyncedAt { get; private set; }

    public DateTimeOffset CreatedAt { get; private set; }

    public DateTimeOffset UpdatedAt { get; private set; }

    public void Sync(
        string? name,
        decimal? inputCost,
        decimal? outputCost,
        decimal? cachedInputCost,
        string? currency,
        int? contextLength,
        string? capabilities,
        DateTimeOffset now)
    {
        Name = name;
        InputCost = inputCost;
        OutputCost = outputCost;
        CachedInputCost = cachedInputCost;
        Currency = currency ?? "USD";
        ContextLength = contextLength;
        Capabilities = capabilities;
        Status = MiningStatus.Active;
        LastSyncedAt = now;
        UpdatedAt = now;
    }

    /// <summary>Marks a model the catalogue no longer lists.</summary>
    public void Disable(DateTimeOffset now)
    {
        Status = MiningStatus.Disabled;
        UpdatedAt = now;
    }
}

/// <summary>One observation of a provider's prepaid balance, in USD.</summary>
public sealed class ProviderBalanceSnapshot
{
    private ProviderBalanceSnapshot()
    {
    }

    public ProviderBalanceSnapshot(
        Guid id,
        Guid llmProviderId,
        decimal balance,
        decimal reservedBalance,
        decimal totalBalance,
        DateTimeOffset checkedAt)
    {
        Id = id;
        LlmProviderId = llmProviderId;
        Balance = balance;
        ReservedBalance = reservedBalance;
        TotalBalance = totalBalance;
        CheckedAt = checkedAt;
    }

    public Guid Id { get; private set; }

    public Guid LlmProviderId { get; private set; }

    /// <summary>Spendable right now, gift credit included.</summary>
    public decimal Balance { get; private set; }

    /// <summary>Held against in-flight requests.</summary>
    public decimal ReservedBalance { get; private set; }

    public decimal TotalBalance { get; private set; }

    public DateTimeOffset CheckedAt { get; private set; }
}

/// <summary>
/// A stablecoin transfer towards a provider's deposit address. Broadcasting and being credited
/// are separate states: the blockchain confirming a transfer does not mean the provider has
/// credited the wallet yet.
/// </summary>
public sealed class ProviderDeposit
{
    private ProviderDeposit()
    {
    }

    public ProviderDeposit(
        Guid id,
        Guid llmProviderId,
        Guid coinId,
        string network,
        string address,
        decimal amount,
        string idempotencyKey,
        DateTimeOffset now)
    {
        Id = id;
        LlmProviderId = llmProviderId;
        CoinId = coinId;
        Network = network;
        Address = address;
        Amount = amount;
        IdempotencyKey = idempotencyKey;
        Status = ProviderDepositStatus.Pending;
        CreatedAt = now;
        UpdatedAt = now;
    }

    public Guid Id { get; private set; }

    public Guid LlmProviderId { get; private set; }

    public Guid CoinId { get; private set; }

    public string Network { get; private set; } = null!;

    public string Address { get; private set; } = null!;

    public decimal Amount { get; private set; }

    public string? TransactionHash { get; private set; }

    public int Confirmations { get; private set; }

    public ProviderDepositStatus Status { get; private set; }

    /// <summary>Provider balance observed before the transfer, for credit reconciliation.</summary>
    public decimal? ProviderCreditBefore { get; private set; }

    public decimal? ProviderCreditAfter { get; private set; }

    /// <summary>Unique per transfer intent, so a retry cannot send the same funds twice.</summary>
    public string IdempotencyKey { get; private set; } = null!;

    public string? Error { get; private set; }

    public DateTimeOffset CreatedAt { get; private set; }

    public DateTimeOffset UpdatedAt { get; private set; }

    public DateTimeOffset? ConfirmedAt { get; private set; }

    public void Broadcast(string transactionHash, decimal providerCreditBefore, DateTimeOffset now)
    {
        TransactionHash = transactionHash;
        ProviderCreditBefore = providerCreditBefore;
        Status = ProviderDepositStatus.Broadcast;
        UpdatedAt = now;
    }

    public void ObserveConfirmations(int confirmations, DateTimeOffset now)
    {
        Confirmations = confirmations;
        UpdatedAt = now;

        if (Status is ProviderDepositStatus.Pending or ProviderDepositStatus.Broadcast)
        {
            Status = ProviderDepositStatus.AwaitingConfirmations;
        }
    }

    /// <returns><c>true</c> when this call moved the deposit to confirmed.</returns>
    public bool Confirm(DateTimeOffset now)
    {
        if (Status == ProviderDepositStatus.Confirmed || Status == ProviderDepositStatus.Credited)
        {
            return false;
        }

        Status = ProviderDepositStatus.Confirmed;
        ConfirmedAt = now;
        UpdatedAt = now;

        return true;
    }

    /// <summary>Records that the provider's balance actually grew by at least this transfer.</summary>
    public void MarkCredited(decimal providerCreditAfter, DateTimeOffset now)
    {
        Status = ProviderDepositStatus.Credited;
        ProviderCreditAfter = providerCreditAfter;
        UpdatedAt = now;
    }

    public void Fail(string error, DateTimeOffset now)
    {
        Status = ProviderDepositStatus.Failed;
        Error = error;
        UpdatedAt = now;
    }
}
