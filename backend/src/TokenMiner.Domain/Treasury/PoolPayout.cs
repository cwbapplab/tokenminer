using TokenMiner.Domain.Treasury.Enums;

namespace TokenMiner.Domain.Treasury;

/// <summary>
/// Funds actually paid out by a pool. Distinct from individual shares: shares are earned work,
/// a payout is money that moved, and reconciliation links the two.
/// </summary>
public sealed class PoolPayout
{
    private PoolPayout()
    {
    }

    public PoolPayout(
        Guid id,
        Guid poolId,
        Guid coinId,
        string? walletAddress,
        decimal amount,
        string? transactionHash,
        DateTimeOffset? requestedAt,
        DateTimeOffset? receivedAt,
        int confirmationsCount,
        string? rawData,
        DateTimeOffset now)
    {
        Id = id;
        PoolId = poolId;
        CoinId = coinId;
        WalletAddress = walletAddress;
        Amount = amount;
        TransactionHash = transactionHash;
        RequestedAt = requestedAt;
        ReceivedAt = receivedAt;
        ConfirmationsCount = confirmationsCount;
        RawData = rawData;

        // A payout observed with a transaction hash is already on chain; one without is a request
        // we have made and are waiting for the pool to act on.
        Status = transactionHash is null
            ? PoolPayoutStatus.Pending
            : PoolPayoutStatus.AwaitingConfirmations;

        CreatedAt = now;
        UpdatedAt = now;
    }

    public Guid Id { get; private set; }

    public Guid PoolId { get; private set; }

    public Guid CoinId { get; private set; }

    public string? WalletAddress { get; private set; }

    public decimal Amount { get; private set; }

    public string? TransactionHash { get; private set; }

    public DateTimeOffset? RequestedAt { get; private set; }

    public DateTimeOffset? ReceivedAt { get; private set; }

    public int ConfirmationsCount { get; private set; }

    public PoolPayoutStatus Status { get; private set; }

    public string? RawData { get; private set; }

    /// <summary>
    /// How much of this payout has already been attributed to shares. Without it, reconciliation
    /// would redeem the same funds again on every run.
    /// </summary>
    public decimal RedeemedAmount { get; private set; }

    public DateTimeOffset CreatedAt { get; private set; }

    public DateTimeOffset UpdatedAt { get; private set; }

    /// <summary>Amount still available to attribute to earned shares.</summary>
    public decimal RemainingAmount => Amount - RedeemedAmount;

    public bool IsConfirmed => Status == PoolPayoutStatus.Confirmed;

    public void MarkProcessing(DateTimeOffset now)
    {
        Status = PoolPayoutStatus.Processing;
        UpdatedAt = now;
    }

    /// <summary>Records the blockchain hash and the confirmations observed so far.</summary>
    public void ObserveOnChain(string? transactionHash, int confirmationsCount, DateTimeOffset now)
    {
        TransactionHash ??= transactionHash;
        ConfirmationsCount = confirmationsCount;
        UpdatedAt = now;

        if (Status is PoolPayoutStatus.Pending or PoolPayoutStatus.Processing)
        {
            Status = PoolPayoutStatus.AwaitingConfirmations;
        }
    }

    /// <returns><c>true</c> when this call moved the payout to confirmed.</returns>
    public bool Confirm(DateTimeOffset receivedAt, DateTimeOffset now)
    {
        if (Status == PoolPayoutStatus.Confirmed)
        {
            return false;
        }

        Status = PoolPayoutStatus.Confirmed;
        ReceivedAt ??= receivedAt;
        UpdatedAt = now;

        return true;
    }

    public void Fail(string reason, DateTimeOffset now)
    {
        Status = PoolPayoutStatus.Failed;
        RawData = reason;
        UpdatedAt = now;
    }

    /// <summary>
    /// Attributes part of the payout to shares. Never exceeds the payout amount, so a payout can
    /// only ever redeem as much value as it actually carried.
    /// </summary>
    public void RecordRedeemed(decimal amount, DateTimeOffset now)
    {
        if (amount <= 0)
        {
            return;
        }

        RedeemedAmount = Math.Min(Amount, RedeemedAmount + amount);
        UpdatedAt = now;
    }
}
