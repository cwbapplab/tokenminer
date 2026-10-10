using Microsoft.EntityFrameworkCore;
using TokenMiner.Application.Mining.Abstractions;
using TokenMiner.Domain.Mining;
using TokenMiner.Domain.Mining.Enums;

namespace TokenMiner.Infrastructure.Persistence.Repositories;

internal sealed class ShareRepository(AppDbContext dbContext) : IShareRepository
{
    public void Add(UserMiningShare share) => dbContext.UserMiningShares.Add(share);

    public Task<UserMiningShare?> GetByIdAsync(Guid id, CancellationToken cancellationToken) =>
        dbContext.UserMiningShares.FirstOrDefaultAsync(share => share.Id == id, cancellationToken);

    public Task<UserMiningShare?> GetByPoolAndIdentifierAsync(
        Guid poolId,
        string shareIdentifier,
        CancellationToken cancellationToken) =>
        dbContext.UserMiningShares.FirstOrDefaultAsync(
            share => share.PoolId == poolId && share.ShareIdentifier == shareIdentifier,
            cancellationToken);

    public async Task<IReadOnlyList<UserMiningShare>> ListPendingAsync(
        int limit,
        CancellationToken cancellationToken) =>
        await dbContext.UserMiningShares
            .Where(share => share.RewardStatus == ShareRewardStatus.Pending)
            .OrderBy(share => share.CreatedAt)
            .Take(limit)
            .ToListAsync(cancellationToken);

    public async Task<IReadOnlyList<UserMiningShare>> ListRewardableAsync(CancellationToken cancellationToken) =>
        await dbContext.UserMiningShares
            .Where(share => share.RewardStatus == ShareRewardStatus.Accepted
                || share.RewardStatus == ShareRewardStatus.Redeemed)
            .ToListAsync(cancellationToken);

    public Task<int> CountByStatusAsync(ShareRewardStatus status, CancellationToken cancellationToken) =>
        dbContext.UserMiningShares.CountAsync(share => share.RewardStatus == status, cancellationToken);

    public async Task<IReadOnlyDictionary<Guid, ShareStatusCounts>> CountByUsersAsync(
        IEnumerable<Guid> userIds,
        CancellationToken cancellationToken)
    {
        var ids = userIds.Distinct().ToList();

        var rows = await dbContext.UserMiningShares
            .Where(share => ids.Contains(share.UserId)
                && (share.RewardStatus == ShareRewardStatus.Accepted
                    || share.RewardStatus == ShareRewardStatus.Redeemed
                    || share.RewardStatus == ShareRewardStatus.Rejected))
            .GroupBy(share => new { share.UserId, share.RewardStatus })
            .Select(group => new { group.Key.UserId, group.Key.RewardStatus, Count = group.Count() })
            .ToListAsync(cancellationToken);

        return Collapse(rows.Select(row => (row.UserId, row.RewardStatus, row.Count)));
    }

    public async Task<IReadOnlyDictionary<Guid, ShareStatusCounts>> CountByHardwareIdsAsync(
        IEnumerable<Guid> hardwareIds,
        CancellationToken cancellationToken)
    {
        var ids = hardwareIds.Distinct().ToList();

        var rows = await dbContext.UserMiningShares
            .Where(share => ids.Contains(share.UserHardwareId)
                && (share.RewardStatus == ShareRewardStatus.Accepted
                    || share.RewardStatus == ShareRewardStatus.Redeemed
                    || share.RewardStatus == ShareRewardStatus.Rejected))
            .GroupBy(share => new { share.UserHardwareId, share.RewardStatus })
            .Select(group => new { group.Key.UserHardwareId, group.Key.RewardStatus, Count = group.Count() })
            .ToListAsync(cancellationToken);

        return Collapse(rows.Select(row => (row.UserHardwareId, row.RewardStatus, row.Count)));
    }

    private static IReadOnlyDictionary<Guid, ShareStatusCounts> Collapse(
        IEnumerable<(Guid Key, ShareRewardStatus Status, int Count)> rows) =>
        rows.GroupBy(row => row.Key).ToDictionary(
            group => group.Key,
            group => new ShareStatusCounts(
                group.Where(row => row.Status is ShareRewardStatus.Accepted or ShareRewardStatus.Redeemed)
                    .Sum(row => row.Count),
                group.Where(row => row.Status == ShareRewardStatus.Rejected)
                    .Sum(row => row.Count)));

    public async Task<IReadOnlyList<UserMiningShare>> ListAcceptedByPoolAsync(
        Guid poolId,
        CancellationToken cancellationToken) =>
        await dbContext.UserMiningShares
            .Where(share => share.PoolId == poolId && share.RewardStatus == ShareRewardStatus.Accepted)
            .OrderBy(share => share.CreatedAt)
            .ToListAsync(cancellationToken);

    public async Task<IReadOnlyDictionary<Guid, DateTimeOffset>> GetLastShareAtByHardwareAsync(
        IEnumerable<Guid> hardwareIds,
        CancellationToken cancellationToken)
    {
        var ids = hardwareIds.Distinct().ToList();

        var rows = await dbContext.UserMiningShares
            .Where(share => ids.Contains(share.UserHardwareId))
            .GroupBy(share => share.UserHardwareId)
            .Select(group => new { UserHardwareId = group.Key, LastShareAt = group.Max(share => share.CreatedAt) })
            .ToListAsync(cancellationToken);

        return rows.ToDictionary(row => row.UserHardwareId, row => row.LastShareAt);
    }
}

internal sealed class MiningStatisticRepository(AppDbContext dbContext) : IMiningStatisticRepository
{
    public void Add(MiningStatistic statistic) => dbContext.MiningStatistics.Add(statistic);

    public async Task<IReadOnlyList<MiningStatistic>> ListAsync(CancellationToken cancellationToken) =>
        await dbContext.MiningStatistics.ToListAsync(cancellationToken);

    public async Task<IReadOnlyList<MiningStatistic>> ListByUserAsync(
        Guid userId,
        CancellationToken cancellationToken) =>
        await dbContext.MiningStatistics
            .Where(statistic => statistic.UserId == userId)
            .ToListAsync(cancellationToken);
}
