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

    public async Task<IReadOnlyList<UserMiningShare>> ListAcceptedByPoolAsync(
        Guid poolId,
        CancellationToken cancellationToken) =>
        await dbContext.UserMiningShares
            .Where(share => share.PoolId == poolId && share.RewardStatus == ShareRewardStatus.Accepted)
            .OrderBy(share => share.CreatedAt)
            .ToListAsync(cancellationToken);
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
