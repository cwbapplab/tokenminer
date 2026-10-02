using FluentAssertions;
using TokenMiner.Application.Mining.Services;
using Xunit;

namespace TokenMiner.UnitTests.Mining;

public sealed class MiningAlgoConfigurationTests
{
    [Fact]
    public void TryParse_ReadsCommandAndSupportedCoins()
    {
        const string json = """
            {
              "command": "rgminer.exe --algo {algo} --wallet {wallet}",
              "supportedCoins": ["QTC", "prl"],
              "extra": { "ignored": true }
            }
            """;

        MiningAlgoConfiguration.TryParse(json, out var configuration).Should().BeTrue();

        configuration.Command.Should().Be("rgminer.exe --algo {algo} --wallet {wallet}");
        configuration.SupportedCoins.Should().BeEquivalentTo(["qtc", "prl"]);
    }

    [Fact]
    public void TryParse_TreatsAMissingCoinListAsAllCoins()
    {
        MiningAlgoConfiguration.TryParse("""{ "command": "miner" }""", out var configuration);

        configuration.SupportedCoins.Should().BeEmpty();
        configuration.SupportsCoin("anything").Should().BeTrue();
    }

    [Theory]
    [InlineData(null)]
    [InlineData("")]
    [InlineData("   ")]
    [InlineData("not json")]
    [InlineData("[1,2,3]")]
    [InlineData("""{ "supportedCoins": ["prl"] }""")]
    [InlineData("""{ "command": "" }""")]
    public void TryParse_RejectsUnusableConfigurations(string? json)
    {
        MiningAlgoConfiguration.TryParse(json, out _).Should().BeFalse();
    }

    [Fact]
    public void SupportsCoin_IsCaseInsensitiveAndScopedWhenAListIsGiven()
    {
        MiningAlgoConfiguration.TryParse("""{ "command": "miner", "supportedCoins": ["QTC"] }""", out var configuration);

        configuration.SupportsCoin("qtc").Should().BeTrue();
        configuration.SupportsCoin("QTC").Should().BeTrue();
        configuration.SupportsCoin("prl").Should().BeFalse();
    }

    [Fact]
    public void Render_SubstitutesEveryPlaceholder()
    {
        const string template = "miner --algo {algo} --stratum {poolHost} --wallet {wallet} --worker {workerId}";

        var rendered = MiningCommandRenderer.Render(template, new Dictionary<string, string>
        {
            ["algo"] = "qtc",
            ["poolHost"] = "qtc-br.kryptex.network:7049",
            ["wallet"] = "krxYR9NJVQ",
            ["workerId"] = "user-hardware",
        });

        rendered.Should().Be("miner --algo qtc --stratum qtc-br.kryptex.network:7049 --wallet krxYR9NJVQ --worker user-hardware");
    }

    [Fact]
    public void Render_LeavesUnknownPlaceholdersUntouched()
    {
        var rendered = MiningCommandRenderer.Render("miner {known} {unknown}", new Dictionary<string, string>
        {
            ["known"] = "value",
        });

        rendered.Should().Be("miner value {unknown}");
    }
}
