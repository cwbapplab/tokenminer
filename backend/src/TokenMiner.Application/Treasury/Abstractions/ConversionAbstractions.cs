using TokenMiner.Domain.Treasury;

namespace TokenMiner.Application.Treasury.Abstractions;

public sealed record ConversionQuoteRequest(
    string SourceCoinCode,
    string DestinationCoinCode,
    decimal SourceAmount);

public sealed record ConversionQuote(
    decimal ExchangeRate,
    decimal DestinationAmount,
    decimal Fees,
    string? Detail);

public sealed record ConversionExecutionRequest(
    Guid TransactionId,
    string IdempotencyKey,
    string SourceCoinCode,
    string DestinationCoinCode,
    decimal SourceAmount);

public enum ConversionOutcome
{
    Pending = 0,
    Completed = 1,
    Failed = 2,
}

public sealed record ConversionResult(
    ConversionOutcome Outcome,
    decimal? DestinationAmount,
    decimal? ExchangeRate,
    decimal? Fees,
    string? SourceTransactionId,
    string? DestinationTransactionId,
    string? Error)
{
    public static ConversionResult Completed(
        decimal destinationAmount,
        decimal exchangeRate,
        decimal fees,
        string? destinationTransactionId = null) =>
        new(ConversionOutcome.Completed, destinationAmount, exchangeRate, fees, null, destinationTransactionId, null);

    public static ConversionResult AwaitingManualSettlement(string detail) =>
        new(ConversionOutcome.Pending, null, null, null, null, null, detail);

    public static ConversionResult Failed(string error) =>
        new(ConversionOutcome.Failed, null, null, null, null, null, error);
}

/// <summary>
/// Converts a mined coin into another asset.
/// </summary>
/// <remarks>
/// The plan also lists deposit, withdrawal, balance and transfer-status operations. Those only
/// become meaningful once a live exchange adapter exists; the booking-only providers shipped
/// here (manual and simulated) do not exercise them, so they are intentionally absent rather
/// than declared and unimplemented.
/// </remarks>
public interface IConversionProvider
{
    /// <summary>Matches <c>conversion_providers.name</c>.</summary>
    string Name { get; }

    /// <returns>Null when the provider cannot price the pair at all.</returns>
    Task<ConversionQuote?> QuoteAsync(ConversionQuoteRequest request, CancellationToken cancellationToken);

    /// <summary>
    /// Executes a conversion. Must be idempotent on
    /// <see cref="ConversionExecutionRequest.IdempotencyKey"/>, because a retry after a timeout
    /// must never sell the same funds twice.
    /// </summary>
    Task<ConversionResult> ExecuteAsync(
        ConversionExecutionRequest request,
        CancellationToken cancellationToken);
}

public interface IConversionRepository
{
    void AddProvider(ConversionProvider provider);

    void AddRoute(ConversionRoute route);

    void AddTransaction(ConversionTransaction transaction);

    Task<ConversionProvider?> GetProviderByIdAsync(Guid id, CancellationToken cancellationToken);

    Task<ConversionProvider?> GetProviderByNameAsync(string name, CancellationToken cancellationToken);

    Task<IReadOnlyList<ConversionProvider>> ListProvidersAsync(CancellationToken cancellationToken);

    Task<ConversionRoute?> GetRouteByIdAsync(Guid id, CancellationToken cancellationToken);

    Task<IReadOnlyList<ConversionRoute>> ListRoutesAsync(CancellationToken cancellationToken);

    Task<ConversionTransaction?> GetTransactionByIdAsync(Guid id, CancellationToken cancellationToken);

    Task<ConversionTransaction?> GetTransactionByIdempotencyKeyAsync(
        string idempotencyKey,
        CancellationToken cancellationToken);

    Task<IReadOnlyList<ConversionTransaction>> ListTransactionsInFlightAsync(
        int limit,
        CancellationToken cancellationToken);

    Task<IReadOnlyList<ConversionTransaction>> ListRecentTransactionsAsync(
        int limit,
        CancellationToken cancellationToken);

    /// <summary>
    /// Completed conversions that have no provider deposit yet, which is what the deposit
    /// pipeline is driven from. Idempotency is by the derived deposit key, not by this query.
    /// </summary>
    Task<IReadOnlyList<ConversionTransaction>> ListCompletedWithoutDepositAsync(
        CancellationToken cancellationToken);

    Task<int> CountInFlightAsync(CancellationToken cancellationToken);
}
