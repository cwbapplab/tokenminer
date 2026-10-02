namespace TokenMiner.Application.Treasury.Abstractions;

/// <summary>Everything a payout provider needs in order to talk to one pool.</summary>
public sealed record PoolPayoutContext(
    Guid PoolId,
    string SystemPoolId,
    string Provider,
    string CoinCode,
    string? PayoutAddress,
    string BaseUrl);

/// <summary>Balance a pool is holding for us.</summary>
public sealed record PoolBalance(decimal Total, decimal Confirmed, decimal Unconfirmed, decimal? Threshold);

/// <summary>A payout as reported by the pool.</summary>
public sealed record PoolPayoutRecord(
    string? TransactionHash,
    decimal Amount,
    DateTimeOffset? RequestedAt,
    DateTimeOffset? ReceivedAt,
    int Confirmations,
    /// <summary>
    /// True when the pool itself reports the payout as final. Pools that expose a status word
    /// rather than a confirmation count settle through this flag.
    /// </summary>
    bool IsSettled,
    string? Status);

/// <summary>Outcome of asking the pool to pay out.</summary>
public sealed record PoolWithdrawalResult(bool Requested, string? TransactionHash, string? Detail);

/// <summary>
/// Reads balances and payout history from a pool, and — only when the pool does not pay out on
/// its own — asks it to withdraw.
/// </summary>
public interface IPoolPayoutProvider
{
    /// <summary>Matches <c>pools.provider</c>, e.g. <c>kryptex</c>.</summary>
    string Name { get; }

    Task<PoolBalance> GetBalanceAsync(PoolPayoutContext context, CancellationToken cancellationToken);

    Task<IReadOnlyList<PoolPayoutRecord>> GetPayoutsAsync(
        PoolPayoutContext context,
        CancellationToken cancellationToken);

    /// <summary>Only meaningful for pools in <c>managed_request</c> payout mode.</summary>
    Task<PoolWithdrawalResult> RequestWithdrawalAsync(
        PoolPayoutContext context,
        decimal amount,
        CancellationToken cancellationToken);
}

/// <summary>Source of a coin's USD price, used to snapshot share valuations.</summary>
public interface ICoinPriceProvider
{
    /// <summary>Recorded against each history row, e.g. <c>kryptex</c>.</summary>
    string Source { get; }

    /// <returns>Null when no price is available; the caller keeps the previous value.</returns>
    Task<decimal?> GetUsdPriceAsync(string coinCode, CancellationToken cancellationToken);
}
