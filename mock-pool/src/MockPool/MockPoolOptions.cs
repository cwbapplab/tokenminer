using System.Globalization;
using System.Net;

namespace MockPool;

/// <summary>Configuration for the mock pool, read from the environment.</summary>
public sealed class MockPoolOptions
{
    public IPEndPoint PearlEndPoint { get; init; } = IPEndPoint.Parse("0.0.0.0:3335");

    public IPEndPoint QuantusEndPoint { get; init; } = IPEndPoint.Parse("0.0.0.0:3336");

    public double Difficulty { get; init; } = 2.0;

    /// <summary>How often a fresh job (the next block) is pushed. Zero disables periodic jobs.</summary>
    public TimeSpan JobInterval { get; init; } = TimeSpan.FromSeconds(5);

    /// <summary>Pearl share difficulty; the pushed target is <c>2^256 / difficulty</c>.</summary>
    public double PearlDifficulty { get; init; } = 1_000_000_000;

    /// <summary>Vardiff target: aim for one accepted share per this many seconds. Zero disables vardiff.</summary>
    public TimeSpan TargetShareInterval { get; init; } = TimeSpan.FromSeconds(10);

    public LogVerbosity Verbosity { get; init; } = LogVerbosity.Info;

    /// <summary>
    /// Builds options from <c>MOCK_POOL_*</c> variables. Pass <paramref name="environment"/> in tests
    /// to avoid touching the process environment.
    /// </summary>
    public static MockPoolOptions FromEnvironment(IReadOnlyDictionary<string, string?>? environment = null)
    {
        string? Read(string name) => environment is not null
            ? environment.TryGetValue(name, out var value) ? value : null
            : Environment.GetEnvironmentVariable(name);

        return new MockPoolOptions
        {
            PearlEndPoint = ParseEndpoint(Read("MOCK_POOL_PRL_ADDR"), "0.0.0.0:3335", "MOCK_POOL_PRL_ADDR"),
            QuantusEndPoint = ParseEndpoint(Read("MOCK_POOL_QTC_ADDR"), "0.0.0.0:3336", "MOCK_POOL_QTC_ADDR"),
            Difficulty = ParseDifficulty(Read("MOCK_POOL_DIFFICULTY"), 2.0, "MOCK_POOL_DIFFICULTY"),
            JobInterval = ParseJobInterval(Read("MOCK_POOL_JOB_INTERVAL_SECONDS"), TimeSpan.FromSeconds(5)),
            PearlDifficulty = ParseDifficulty(Read("MOCK_POOL_PEARL_DIFFICULTY"), 1_000_000_000, "MOCK_POOL_PEARL_DIFFICULTY"),
            TargetShareInterval = ParseJobInterval(Read("MOCK_POOL_TARGET_SHARE_SECONDS"), TimeSpan.FromSeconds(10)),
            Verbosity = ParseVerbosity(Read("MOCK_POOL_LOG_LEVEL")),
        };
    }

    private static IPEndPoint ParseEndpoint(string? value, string fallback, string name)
    {
        var text = string.IsNullOrWhiteSpace(value) ? fallback : value.Trim();
        return IPEndPoint.TryParse(text, out var endpoint)
            ? endpoint
            : throw new InvalidOperationException($"{name}='{text}' is not a valid host:port endpoint.");
    }

    private static double ParseDifficulty(string? value, double fallback, string name)
    {
        if (string.IsNullOrWhiteSpace(value))
        {
            return fallback;
        }

        return double.TryParse(value.Trim(), NumberStyles.Float, CultureInfo.InvariantCulture, out var parsed)
            && parsed > 0
            && double.IsFinite(parsed)
                ? parsed
                : throw new InvalidOperationException($"{name}='{value}' is not a positive number.");
    }

    private static TimeSpan ParseJobInterval(string? value, TimeSpan fallback)
    {
        if (string.IsNullOrWhiteSpace(value))
        {
            return fallback;
        }

        return double.TryParse(value.Trim(), NumberStyles.Float, CultureInfo.InvariantCulture, out var seconds)
            && seconds >= 0
            && double.IsFinite(seconds)
                ? TimeSpan.FromSeconds(seconds)
                : throw new InvalidOperationException(
                    $"MOCK_POOL_JOB_INTERVAL_SECONDS='{value}' is not a non-negative number of seconds.");
    }

    private static LogVerbosity ParseVerbosity(string? value)
    {
        if (string.IsNullOrWhiteSpace(value))
        {
            return LogVerbosity.Info;
        }

        return Enum.TryParse<LogVerbosity>(value.Trim(), ignoreCase: true, out var level) && Enum.IsDefined(level)
            ? level
            : throw new InvalidOperationException(
                $"MOCK_POOL_LOG_LEVEL='{value}' is not one of debug, info, warning, error.");
    }
}
