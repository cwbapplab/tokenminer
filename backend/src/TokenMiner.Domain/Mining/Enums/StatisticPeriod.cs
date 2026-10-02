namespace TokenMiner.Domain.Mining.Enums;

/// <summary>Rolling windows the analytics rollup is materialised for.</summary>
public enum StatisticPeriod
{
    Day1 = 0,
    Day30 = 1,
    Day60 = 2,
    AllTime = 3,
}

public static class StatisticPeriodDbValues
{
    public const string Day1 = "d1";
    public const string Day30 = "d30";
    public const string Day60 = "d60";
    public const string AllTime = "all";

    public static string ToDbValue(this StatisticPeriod period) => period switch
    {
        StatisticPeriod.Day1 => Day1,
        StatisticPeriod.Day30 => Day30,
        StatisticPeriod.Day60 => Day60,
        StatisticPeriod.AllTime => AllTime,
        _ => throw new ArgumentOutOfRangeException(nameof(period), period, "Unknown statistic period."),
    };

    public static StatisticPeriod FromDbValue(string value) => value switch
    {
        Day1 => StatisticPeriod.Day1,
        Day30 => StatisticPeriod.Day30,
        Day60 => StatisticPeriod.Day60,
        AllTime => StatisticPeriod.AllTime,
        _ => throw new ArgumentOutOfRangeException(nameof(value), value, "Unknown statistic period value."),
    };

    /// <summary>Every period the rollup materialises.</summary>
    public static readonly StatisticPeriod[] All =
        [StatisticPeriod.Day1, StatisticPeriod.Day30, StatisticPeriod.Day60, StatisticPeriod.AllTime];
}
