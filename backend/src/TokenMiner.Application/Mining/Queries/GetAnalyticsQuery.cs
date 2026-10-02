using MediatR;
using TokenMiner.Application.Mining.Abstractions;
using TokenMiner.Application.Mining.Models;
using TokenMiner.Domain.Mining;
using TokenMiner.Domain.Mining.Enums;

namespace TokenMiner.Application.Mining.Queries;

public sealed record GetAnalyticsQuery(Guid UserId, Guid? UserHardwareId) : IRequest<MiningAnalyticsDto>;

/// <summary>
/// Earnings for the authenticated user, read from the materialised statistics rather than
/// aggregating raw shares. An optional device filter is always applied within the caller's
/// own rows, so it can never expose another user's data.
/// </summary>
internal sealed class GetAnalyticsQueryHandler(IMiningStatisticRepository statistics)
    : IRequestHandler<GetAnalyticsQuery, MiningAnalyticsDto>
{
    public async Task<MiningAnalyticsDto> Handle(
        GetAnalyticsQuery request,
        CancellationToken cancellationToken)
    {
        IReadOnlyList<MiningStatistic> rows = await statistics.ListByUserAsync(
            request.UserId,
            cancellationToken);

        if (request.UserHardwareId is not null)
        {
            rows = rows.Where(row => row.UserHardwareId == request.UserHardwareId).ToList();
        }

        var hardware = rows
            .GroupBy(row => row.UserHardwareId)
            .Select(group => new HardwareAnalyticsDto(
                group.Key,
                Sum(group, StatisticPeriod.Day30),
                Sum(group, StatisticPeriod.Day60),
                Sum(group, StatisticPeriod.AllTime)))
            .OrderBy(item => item.UserHardwareId)
            .ToList();

        return new MiningAnalyticsDto(
            hardware,
            new AnalyticsTotalsDto(
                hardware.Sum(item => item.Last30Usd),
                hardware.Sum(item => item.Last60Usd),
                hardware.Sum(item => item.AllTimeUsd)));
    }

    private static decimal Sum(IEnumerable<MiningStatistic> rows, StatisticPeriod period) =>
        rows.Where(row => row.Period == period).Sum(row => row.UsdValue);
}
