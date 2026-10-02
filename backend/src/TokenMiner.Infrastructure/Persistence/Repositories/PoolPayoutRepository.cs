using Microsoft.EntityFrameworkCore;
using TokenMiner.Application.Treasury.Abstractions;
using TokenMiner.Domain.Treasury;
using TokenMiner.Domain.Treasury.Enums;

namespace TokenMiner.Infrastructure.Persistence.Repositories;

internal sealed class PoolPayoutRepository(AppDbContext dbContext) : IPoolPayoutRepository
{
    public void Add(PoolPayout payout) => dbContext.PoolPayouts.Add(payout);

    public Task<bool> ExistsByTransactionHashAsync(
        Guid poolId,
        string transactionHash,
        CancellationToken cancellationToken) =>
        dbContext.PoolPayouts.AnyAsync(
            payout => payout.PoolId == poolId && payout.TransactionHash == transactionHash,
            cancellationToken);

    public async Task<IReadOnlyList<PoolPayout>> ListUnsettledAsync(CancellationToken cancellationToken) =>
        await dbContext.PoolPayouts
            .Where(payout => payout.Status == PoolPayoutStatus.Pending
                || payout.Status == PoolPayoutStatus.Processing
                || payout.Status == PoolPayoutStatus.AwaitingConfirmations)
            .ToListAsync(cancellationToken);

    public async Task<IReadOnlyList<PoolPayout>> ListRedeemableAsync(CancellationToken cancellationToken) =>
        await dbContext.PoolPayouts
            .Where(payout => payout.Status == PoolPayoutStatus.Confirmed
                && payout.RedeemedAmount < payout.Amount)
            .OrderBy(payout => payout.ReceivedAt)
            .ToListAsync(cancellationToken);

    public async Task<IReadOnlyList<PoolPayout>> ListConfirmedAsync(CancellationToken cancellationToken) =>
        await dbContext.PoolPayouts
            .Where(payout => payout.Status == PoolPayoutStatus.Confirmed)
            .OrderBy(payout => payout.ReceivedAt)
            .ToListAsync(cancellationToken);

    public async Task<IReadOnlyList<PoolPayout>> ListRecentAsync(
        int limit,
        CancellationToken cancellationToken) =>
        await dbContext.PoolPayouts
            .OrderByDescending(payout => payout.CreatedAt)
            .Take(Math.Clamp(limit, 1, 500))
            .ToListAsync(cancellationToken);
}
