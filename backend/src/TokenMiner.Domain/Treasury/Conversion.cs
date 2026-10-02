using TokenMiner.Domain.Treasury.Enums;

namespace TokenMiner.Domain.Treasury;

/// <summary>
/// An exchange or swap service able to turn a mined coin into a stablecoin. Credentials are
/// never stored here — only a reference to a secret.
/// </summary>
public sealed class ConversionProvider
{
    private ConversionProvider()
    {
    }

    public ConversionProvider(
        Guid id,
        string name,
        string baseUrl,
        int priority,
        string? credentialRef,
        string? supportedFeatures,
        DateTimeOffset now)
    {
        Id = id;
        Name = name;
        BaseUrl = baseUrl;
        Priority = priority;
        CredentialRef = credentialRef;
        SupportedFeatures = supportedFeatures;
        Status = TreasuryStatus.Active;
        CreatedAt = now;
        UpdatedAt = now;
    }

    public Guid Id { get; private set; }

    /// <summary>Matches the adapter's <c>Name</c>, e.g. <c>simulated</c>.</summary>
    public string Name { get; private set; } = null!;

    public string BaseUrl { get; private set; } = null!;

    public TreasuryStatus Status { get; private set; }

    public int Priority { get; private set; }

    /// <summary>Key into the secret store; never the credential itself.</summary>
    public string? CredentialRef { get; private set; }

    public string? SupportedFeatures { get; private set; }

    public DateTimeOffset CreatedAt { get; private set; }

    public DateTimeOffset UpdatedAt { get; private set; }

    public void Update(string baseUrl, int priority, string? credentialRef, string? supportedFeatures, TreasuryStatus status, DateTimeOffset now)
    {
        BaseUrl = baseUrl;
        Priority = priority;
        CredentialRef = credentialRef;
        SupportedFeatures = supportedFeatures;
        Status = status;
        UpdatedAt = now;
    }
}

/// <summary>How one coin becomes another: the exchange, the pair and when it may be used.</summary>
public sealed class ConversionRoute
{
    private ConversionRoute()
    {
    }

    public ConversionRoute(
        Guid id,
        Guid conversionProviderId,
        Guid sourceCoinId,
        Guid destinationCoinId,
        string? sourceNetwork,
        string? destinationNetwork,
        int priority,
        decimal minimumAmount,
        DateTimeOffset now)
    {
        Id = id;
        ConversionProviderId = conversionProviderId;
        SourceCoinId = sourceCoinId;
        DestinationCoinId = destinationCoinId;
        SourceNetwork = sourceNetwork;
        DestinationNetwork = destinationNetwork;
        Priority = priority;
        MinimumAmount = minimumAmount;
        Enabled = true;
        CreatedAt = now;
        UpdatedAt = now;
    }

    public Guid Id { get; private set; }

    public Guid ConversionProviderId { get; private set; }

    public Guid SourceCoinId { get; private set; }

    public Guid DestinationCoinId { get; private set; }

    public string? SourceNetwork { get; private set; }

    public string? DestinationNetwork { get; private set; }

    public bool Enabled { get; private set; }

    public int Priority { get; private set; }

    public decimal MinimumAmount { get; private set; }

    public DateTimeOffset CreatedAt { get; private set; }

    public DateTimeOffset UpdatedAt { get; private set; }

    public void Update(bool enabled, int priority, decimal minimumAmount, DateTimeOffset now)
    {
        Enabled = enabled;
        Priority = priority;
        MinimumAmount = minimumAmount;
        UpdatedAt = now;
    }
}

/// <summary>
/// One conversion attempt. Financial in nature, so it carries an idempotency key and an
/// immutable record of the amounts and rate involved.
/// </summary>
public sealed class ConversionTransaction
{
    private ConversionTransaction()
    {
    }

    public ConversionTransaction(
        Guid id,
        Guid conversionProviderId,
        Guid sourceCoinId,
        decimal sourceAmount,
        Guid destinationCoinId,
        string idempotencyKey,
        DateTimeOffset now)
    {
        Id = id;
        ConversionProviderId = conversionProviderId;
        SourceCoinId = sourceCoinId;
        SourceAmount = sourceAmount;
        DestinationCoinId = destinationCoinId;
        IdempotencyKey = idempotencyKey;
        Status = ConversionTransactionStatus.Pending;
        CreatedAt = now;
        UpdatedAt = now;
    }

    public Guid Id { get; private set; }

    public Guid ConversionProviderId { get; private set; }

    public Guid SourceCoinId { get; private set; }

    public decimal SourceAmount { get; private set; }

    public Guid DestinationCoinId { get; private set; }

    public decimal? DestinationAmount { get; private set; }

    public string? SourceTransactionId { get; private set; }

    public string? DestinationTransactionId { get; private set; }

    public decimal? ExchangeRate { get; private set; }

    public decimal? Fees { get; private set; }

    public ConversionTransactionStatus Status { get; private set; }

    /// <summary>Unique per conversion intent, so a retry can never sell the same funds twice.</summary>
    public string IdempotencyKey { get; private set; } = null!;

    public string? Error { get; private set; }

    public DateTimeOffset CreatedAt { get; private set; }

    public DateTimeOffset UpdatedAt { get; private set; }

    public DateTimeOffset? CompletedAt { get; private set; }

    public void MarkProcessing(string? sourceTransactionId, DateTimeOffset now)
    {
        Status = ConversionTransactionStatus.Processing;
        SourceTransactionId ??= sourceTransactionId;
        UpdatedAt = now;
    }

    public void Complete(
        decimal destinationAmount,
        decimal exchangeRate,
        decimal fees,
        string? destinationTransactionId,
        DateTimeOffset now)
    {
        Status = ConversionTransactionStatus.Completed;
        DestinationAmount = destinationAmount;
        ExchangeRate = exchangeRate;
        Fees = fees;
        DestinationTransactionId = destinationTransactionId;
        Error = null;
        CompletedAt = now;
        UpdatedAt = now;
    }

    /// <summary>Completes a transaction from a provider response, using the quote as fallback.</summary>
    public void CompleteFromProvider(
        decimal? destinationAmount,
        decimal? exchangeRate,
        decimal? fees,
        string? destinationTransactionId,
        DateTimeOffset now) =>
        Complete(
            destinationAmount ?? DestinationAmount ?? 0m,
            exchangeRate ?? ExchangeRate ?? 0m,
            fees ?? Fees ?? 0m,
            destinationTransactionId ?? DestinationTransactionId,
            now);

    public void Fail(string error, DateTimeOffset now)
    {
        Status = ConversionTransactionStatus.Failed;
        Error = error;
        UpdatedAt = now;
    }

    public void Cancel(string reason, DateTimeOffset now)
    {
        Status = ConversionTransactionStatus.Cancelled;
        Error = reason;
        UpdatedAt = now;
    }

    public void RecordQuote(decimal exchangeRate, decimal destinationAmount, decimal fees, DateTimeOffset now)
    {
        ExchangeRate = exchangeRate;
        DestinationAmount = destinationAmount;
        Fees = fees;
        UpdatedAt = now;
    }
}
