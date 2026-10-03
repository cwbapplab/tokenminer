using TokenMiner.Domain.Mining;
using TokenMiner.Domain.Mining.Enums;

namespace TokenMiner.Application.Mining.Abstractions;

public interface IShareRepository
{
    void Add(UserMiningShare share);

    Task<UserMiningShare?> GetByIdAsync(Guid id, CancellationToken cancellationToken);

    /// <summary>Idempotency lookup on the pool-side share identity.</summary>
    Task<UserMiningShare?> GetByPoolAndIdentifierAsync(
        Guid poolId,
        string shareIdentifier,
        CancellationToken cancellationToken);

    /// <summary>Oldest pending shares, for the reward processor.</summary>
    Task<IReadOnlyList<UserMiningShare>> ListPendingAsync(int limit, CancellationToken cancellationToken);

    /// <summary>Shares that count towards earnings, i.e. accepted or already redeemed.</summary>
    Task<IReadOnlyList<UserMiningShare>> ListRewardableAsync(CancellationToken cancellationToken);

    /// <summary>Accepted shares for one pool, oldest first, for payout attribution.</summary>
    Task<IReadOnlyList<UserMiningShare>> ListAcceptedByPoolAsync(
        Guid poolId,
        CancellationToken cancellationToken);

    Task<int> CountByStatusAsync(ShareRewardStatus status, CancellationToken cancellationToken);

    /// <summary>Terminal reward counts (accepted/redeemed and rejected) per user.</summary>
    Task<IReadOnlyDictionary<Guid, ShareStatusCounts>> CountByUsersAsync(
        IEnumerable<Guid> userIds,
        CancellationToken cancellationToken);

    /// <summary>Terminal reward counts (accepted/redeemed and rejected) per device.</summary>
    Task<IReadOnlyDictionary<Guid, ShareStatusCounts>> CountByHardwareIdsAsync(
        IEnumerable<Guid> hardwareIds,
        CancellationToken cancellationToken);
}

/// <summary>
/// The two share states the operator sees: <see cref="Accepted"/> folds in redeemed shares
/// (an accepted share that has already been paid out), <see cref="Rejected"/> is terminal.
/// </summary>
public readonly record struct ShareStatusCounts(int Accepted, int Rejected);

public interface IMiningStatisticRepository
{
    void Add(MiningStatistic statistic);

    Task<IReadOnlyList<MiningStatistic>> ListAsync(CancellationToken cancellationToken);

    Task<IReadOnlyList<MiningStatistic>> ListByUserAsync(Guid userId, CancellationToken cancellationToken);
}

/// <summary>
/// Replay protection for signed service requests. A nonce may only be accepted once.
/// </summary>
public interface IServiceNonceStore
{
    /// <returns><c>false</c> when the nonce has already been used by that service.</returns>
    Task<bool> TryRegisterAsync(
        string serviceId,
        string nonce,
        DateTimeOffset expiresAt,
        CancellationToken cancellationToken);
}
