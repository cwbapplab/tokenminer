using System.Net;
using MockPool;
using Xunit;

namespace MockPool.Tests;

public sealed class MockPoolOptionsTests
{
    [Fact]
    public void Defaults_match_the_documented_values()
    {
        var options = MockPoolOptions.FromEnvironment(new Dictionary<string, string?>());

        Assert.Equal(new IPEndPoint(IPAddress.Parse("0.0.0.0"), 3335), options.PearlEndPoint);
        Assert.Equal(new IPEndPoint(IPAddress.Parse("0.0.0.0"), 3336), options.QuantusEndPoint);
        Assert.Equal(2.0, options.Difficulty);
        Assert.Equal(TimeSpan.FromSeconds(5), options.JobInterval);
        Assert.Equal(1_000_000_000, options.PearlDifficulty);
        Assert.Equal(LogVerbosity.Info, options.Verbosity);
    }

    [Fact]
    public void Environment_overrides_are_applied()
    {
        var options = MockPoolOptions.FromEnvironment(new Dictionary<string, string?>
        {
            ["MOCK_POOL_PRL_ADDR"] = "127.0.0.1:4001",
            ["MOCK_POOL_QTC_ADDR"] = "127.0.0.1:4002",
            ["MOCK_POOL_DIFFICULTY"] = "8.5",
            ["MOCK_POOL_JOB_INTERVAL_SECONDS"] = "5",
            ["MOCK_POOL_PEARL_DIFFICULTY"] = "250000000",
            ["MOCK_POOL_LOG_LEVEL"] = "debug",
        });

        Assert.Equal(new IPEndPoint(IPAddress.Loopback, 4001), options.PearlEndPoint);
        Assert.Equal(new IPEndPoint(IPAddress.Loopback, 4002), options.QuantusEndPoint);
        Assert.Equal(8.5, options.Difficulty);
        Assert.Equal(TimeSpan.FromSeconds(5), options.JobInterval);
        Assert.Equal(250_000_000, options.PearlDifficulty);
        Assert.Equal(LogVerbosity.Debug, options.Verbosity);
    }

    [Fact]
    public void Zero_job_interval_disables_periodic_jobs()
    {
        var options = MockPoolOptions.FromEnvironment(new Dictionary<string, string?>
        {
            ["MOCK_POOL_JOB_INTERVAL_SECONDS"] = "0",
        });

        Assert.Equal(TimeSpan.Zero, options.JobInterval);
    }

    [Theory]
    [InlineData("MOCK_POOL_DIFFICULTY", "-1")]
    [InlineData("MOCK_POOL_DIFFICULTY", "abc")]
    [InlineData("MOCK_POOL_JOB_INTERVAL_SECONDS", "abc")]
    [InlineData("MOCK_POOL_JOB_INTERVAL_SECONDS", "-5")]
    [InlineData("MOCK_POOL_PRL_ADDR", "not-an-endpoint")]
    [InlineData("MOCK_POOL_LOG_LEVEL", "verbose")]
    public void Invalid_values_are_rejected(string key, string value) =>
        Assert.Throws<InvalidOperationException>(() =>
            MockPoolOptions.FromEnvironment(new Dictionary<string, string?> { [key] = value }));
}
