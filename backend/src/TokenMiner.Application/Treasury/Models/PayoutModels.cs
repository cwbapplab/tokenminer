using TokenMiner.Domain.Treasury;
using TokenMiner.Domain.Treasury.Enums;

namespace TokenMiner.Application.Treasury.Abstractions;

public interface IPoolPayoutRepository
{
    void Add(PoolPayout payout);

    /// <summary>Idempotency check so a re-observed payout is not stored twice.</summary>
    Task<bool> ExistsByTransactionHashAsync(
        Guid poolId,
        string transactionHash,
        CancellationToken cancellationToken);

    Task<IReadOnlyList<PoolPayout>> ListUnsettledAsync(CancellationToken cancellationToken);

    /// <summary>Confirmed payouts that still have value to attribute to shares.</summary>
    Task<IReadOnlyList<PoolPayout>> ListRedeemableAsync(CancellationToken cancellationToken);

    /// <summary>Every settled payout, which is what conversions are created from.</summary>
    Task<IReadOnlyList<PoolPayout>> ListConfirmedAsync(CancellationToken cancellationToken);

    Task<IReadOnlyList<PoolPayout>> ListRecentAsync(int limit, CancellationToken cancellationToken);
}

public sealed record PoolPayoutDto(
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

public static class PoolPayoutMappings
{
    public static PoolPayoutDto ToDto(this PoolPayout payout) => new(
        payout.Id,
        payout.PoolId,
        payout.CoinId,
        payout.WalletAddress,
        payout.Amount,
        payout.RedeemedAmount,
        payout.TransactionHash,
        payout.RequestedAt,
        payout.ReceivedAt,
        payout.ConfirmationsCount,
        payout.Status.ToDbValue(),
        payout.CreatedAt,
        payout.UpdatedAt);
}

public sealed record PoolMonitorResult(int PayoutsRecorded, int WithdrawalsRequested);

public sealed record PayoutReconciliationResult(int PayoutsConfirmed, int SharesRedeemed);
