using MediatR;
using TokenMiner.Application.Common.Abstractions;
using TokenMiner.Application.Mining.Abstractions;
using TokenMiner.Domain.Mining;
using TokenMiner.Domain.Mining.Enums;

namespace TokenMiner.Application.Mining.Statistics;

public sealed record RecalculateMiningStatisticsCommand : IRequest<int>;

/// <summary>
/// Rebuilds the materialised totals analytics reads. Every period is recomputed from the
/// rewardable shares, so a late share or a corrected valuation simply lands in the next run.
/// </summary>
internal sealed class RecalculateMiningStatisticsCommandHandler(
    IShareRepository shares,
    IMiningStatisticRepository statistics,
    IUnitOfWork unitOfWork,
    TimeProvider timeProvider) : IRequestHandler<RecalculateMiningStatisticsCommand, int>
{
    public async Task<int> Handle(
        RecalculateMiningStatisticsCommand request,
        CancellationToken cancellationToken)
    {
        var now = timeProvider.GetUtcNow();

        var rewardable = await shares.ListRewardableAsync(cancellationToken);
        var existing = await statistics.ListAsync(cancellationToken);

        var existingByKey = existing.ToDictionary(
            statistic => (statistic.UserHardwareId, statistic.CoinId, statistic.Period));

        var touched = 0;

        foreach (var group in rewardable.GroupBy(share => new
                 {
                     share.UserId,
                     share.UserHardwareId,
                     share.CoinId,
                 }))
        {
            var groupShares = group.ToList();

            foreach (var period in StatisticPeriodDbValues.All)
            {
                var windowStart = WindowStart(period, now);

                var amount = 0m;
                var usdValue = 0m;

                foreach (var share in groupShares)
                {
                    if (windowStart is not null && share.CreatedAt < windowStart)
                    {
                        continue;
                    }

                    var coinValue = share.CoinValue ?? 0m;
                    amount += coinValue;
                    usdValue += coinValue * (share.ApproxUsdValueAtTime ?? 0m);
                }

                var key = (group.Key.UserHardwareId, group.Key.CoinId, period);

                if (existingByKey.TryGetValue(key, out var statistic))
                {
                    statistic.Update(amount, usdValue, now);
                }
                else
                {
                    statistics.Add(new MiningStatistic(
                        Guid.NewGuid(),
                        group.Key.UserId,
                        group.Key.UserHardwareId,
                        group.Key.CoinId,
                        period,
                        amount,
                        usdValue,
                        now));
                }

                touched++;
            }
        }

        if (touched > 0)
        {
            await unitOfWork.SaveChangesAsync(cancellationToken);
        }

        return touched;
    }

    private static DateTimeOffset? WindowStart(StatisticPeriod period, DateTimeOffset now) => period switch
    {
        StatisticPeriod.Day1 => now.AddDays(-1),
        StatisticPeriod.Day30 => now.AddDays(-30),
        StatisticPeriod.Day60 => now.AddDays(-60),
        _ => null,
    };
}
