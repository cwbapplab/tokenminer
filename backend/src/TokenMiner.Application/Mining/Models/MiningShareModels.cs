using TokenMiner.Domain.Mining;
using TokenMiner.Domain.Mining.Enums;

namespace TokenMiner.Application.Mining.Models;

public sealed record RecordedShareDto(
    Guid Id,
    string ShareIdentifier,
    string RewardStatus,
    decimal? CoinValue,
    decimal? ApproxUsdValueAtTime,
    DateTimeOffset CreatedAt,
    bool AlreadyRecorded);

public sealed record HardwareAnalyticsDto(
    Guid UserHardwareId,
    decimal Last30Usd,
    decimal Last60Usd,
    decimal AllTimeUsd);

public sealed record AnalyticsTotalsDto(decimal Last30Usd, decimal Last60Usd, decimal AllTimeUsd);

public sealed record MiningAnalyticsDto(
    IReadOnlyList<HardwareAnalyticsDto> Hardware,
    AnalyticsTotalsDto Totals);

public static class MiningShareMappings
{
    public static RecordedShareDto ToDto(this UserMiningShare share, bool alreadyRecorded = false) => new(
        share.Id,
        share.ShareIdentifier,
        share.RewardStatus.ToDbValue(),
        share.CoinValue,
        share.ApproxUsdValueAtTime,
        share.CreatedAt,
        alreadyRecorded);
}
