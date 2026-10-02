using FluentAssertions;
using TokenMiner.Domain.Mining;
using TokenMiner.Domain.Mining.Enums;
using Xunit;

namespace TokenMiner.UnitTests.Mining;

public sealed class CoinTests
{
    private static readonly DateTimeOffset Now = new(2026, 1, 1, 12, 0, 0, TimeSpan.Zero);

    private static Coin CreateCoin() => new(Guid.NewGuid(), "prl", "Pearl", "pearl-mainnet", 8, Now);

    [Fact]
    public void NewCoin_IsActiveAndUnpriced()
    {
        var coin = CreateCoin();

        coin.Status.Should().Be(MiningStatus.Active);
        coin.LastKnownUsdValue.Should().BeNull();
        coin.LastValueDate.Should().BeNull();
        coin.CreatedAt.Should().Be(Now);
    }

    [Fact]
    public void UpdateDetails_ReplacesEditableFieldsAndBumpsUpdatedAt()
    {
        var coin = CreateCoin();
        var later = Now.AddMinutes(5);

        coin.UpdateDetails("Pearl v2", "pearl-testnet", 6, MiningStatus.Disabled, later);

        coin.Name.Should().Be("Pearl v2");
        coin.Network.Should().Be("pearl-testnet");
        coin.Decimals.Should().Be(6);
        coin.Status.Should().Be(MiningStatus.Disabled);
        coin.UpdatedAt.Should().Be(later);
    }

    [Fact]
    public void RecordPrice_UpdatesOnlyTheCachedValuation()
    {
        var coin = CreateCoin();
        var observedAt = Now.AddHours(-1);
        var later = Now.AddMinutes(1);

        coin.RecordPrice(0.42m, observedAt, later);

        coin.LastKnownUsdValue.Should().Be(0.42m);
        coin.LastValueDate.Should().Be(observedAt);
        coin.UpdatedAt.Should().Be(later);
    }
}

public sealed class MiningAlgoTests
{
    private static readonly DateTimeOffset Now = new(2026, 1, 1, 12, 0, 0, TimeSpan.Zero);

    [Fact]
    public void NewAlgorithm_IsActiveWithItsPriority()
    {
        var algorithm = new MiningAlgo(Guid.NewGuid(), "pearl-gpu", "Pearl GPU", 100, "{\"args\":[]}", Now);

        algorithm.Status.Should().Be(MiningStatus.Active);
        algorithm.Priority.Should().Be(100);
        algorithm.Configuration.Should().Be("{\"args\":[]}");
    }

    [Fact]
    public void Update_ReplacesSelectionMetadata()
    {
        var algorithm = new MiningAlgo(Guid.NewGuid(), "pearl-gpu", "Pearl GPU", 100, null, Now);
        var later = Now.AddMinutes(1);

        algorithm.Update("Pearl GPU v2", 250, "{\"bin\":\"rgminer\"}", MiningStatus.Disabled, later);

        algorithm.Name.Should().Be("Pearl GPU v2");
        algorithm.Priority.Should().Be(250);
        algorithm.Configuration.Should().Be("{\"bin\":\"rgminer\"}");
        algorithm.Status.Should().Be(MiningStatus.Disabled);
        algorithm.UpdatedAt.Should().Be(later);
    }
}
