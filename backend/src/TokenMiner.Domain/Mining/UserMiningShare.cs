using TokenMiner.Domain.Mining.Enums;

namespace TokenMiner.Domain.Mining;

/// <summary>
/// An accepted share reported by the stratum proxy. Records the proof-of-work detail the pool
/// used so a share can be independently re-validated later, plus the reward accounting state.
/// </summary>
public sealed class UserMiningShare
{
    private UserMiningShare()
    {
    }

    public UserMiningShare(
        Guid id,
        Guid userId,
        Guid userHardwareId,
        Guid userHardwareMinerId,
        Guid poolId,
        Guid coinId,
        string shareIdentifier,
        string workerIdentifier,
        string? jobId,
        string? nonce,
        string? extranonce,
        decimal? difficulty,
        string? target,
        string? resultHash,
        DateTimeOffset shareTimestamp,
        decimal? coinValue,
        decimal? approxUsdValueAtTime,
        string? poolResponse,
        DateTimeOffset now)
    {
        Id = id;
        UserId = userId;
        UserHardwareId = userHardwareId;
        UserHardwareMinerId = userHardwareMinerId;
        PoolId = poolId;
        CoinId = coinId;
        ShareIdentifier = shareIdentifier;
        WorkerIdentifier = workerIdentifier;
        JobId = jobId;
        Nonce = nonce;
        Extranonce = extranonce;
        Difficulty = difficulty;
        Target = target;
        ResultHash = resultHash;
        ShareTimestamp = shareTimestamp;
        CoinValue = coinValue;
        ApproxUsdValueAtTime = approxUsdValueAtTime;
        PoolResponse = poolResponse;
        RewardStatus = ShareRewardStatus.Pending;
        CreatedAt = now;
    }

    public Guid Id { get; private set; }

    public Guid UserId { get; private set; }

    public Guid UserHardwareId { get; private set; }

    public Guid UserHardwareMinerId { get; private set; }

    public Guid PoolId { get; private set; }

    public Guid CoinId { get; private set; }

    /// <summary>Pool-side identity of the share; unique per pool and the idempotency key.</summary>
    public string ShareIdentifier { get; private set; } = null!;

    public string WorkerIdentifier { get; private set; } = null!;

    public string? JobId { get; private set; }

    public string? Nonce { get; private set; }

    public string? Extranonce { get; private set; }

    public decimal? Difficulty { get; private set; }

    public string? Target { get; private set; }

    public string? ResultHash { get; private set; }

    public DateTimeOffset ShareTimestamp { get; private set; }

    /// <summary>Mined amount, when the pool or proxy reports it. Null until known.</summary>
    public decimal? CoinValue { get; private set; }

    /// <summary>Snapshot of the coin's USD price at acceptance time, never recalculated.</summary>
    public decimal? ApproxUsdValueAtTime { get; private set; }

    public string? PoolResponse { get; private set; }

    public ShareRewardStatus RewardStatus { get; private set; }

    public string? Reason { get; private set; }

    public DateTimeOffset CreatedAt { get; private set; }

    public DateTimeOffset? ProcessedAt { get; private set; }

    /// <returns><c>true</c> when the share was actually claimed for processing.</returns>
    public bool MarkProcessing()
    {
        if (RewardStatus != ShareRewardStatus.Pending)
        {
            return false;
        }

        RewardStatus = ShareRewardStatus.Processing;
        return true;
    }

    public void Accept(DateTimeOffset now)
    {
        RewardStatus = ShareRewardStatus.Accepted;
        Reason = null;
        ProcessedAt = now;
    }

    /// <summary>Terminal failure: the share can never be redeemed.</summary>
    public void Reject(string reason, DateTimeOffset now)
    {
        RewardStatus = ShareRewardStatus.Rejected;
        Reason = reason;
        ProcessedAt = now;
    }

    /// <summary>Retryable failure: the share goes back in the queue with the reason recorded.</summary>
    public void RevertToPending(string reason)
    {
        RewardStatus = ShareRewardStatus.Pending;
        Reason = reason;
    }

    /// <summary>Marks the share as paid out once a confirmed pool payout covers it.</summary>
    public void Redeem(DateTimeOffset now)
    {
        RewardStatus = ShareRewardStatus.Redeemed;
        Reason = null;
        ProcessedAt = now;
    }

    /// <summary>Fills in the mined amount when it becomes known after acceptance.</summary>
    public void RecordCoinValue(decimal coinValue)
    {
        CoinValue = coinValue;
    }
}
