using FluentAssertions;
using TokenMiner.Domain.Mining;
using TokenMiner.Domain.Mining.Enums;
using Xunit;

namespace TokenMiner.UnitTests.Mining;

public sealed class MiningShareTests
{
    private static readonly DateTimeOffset Now = new(2026, 1, 1, 12, 0, 0, TimeSpan.Zero);

    private static UserMiningShare CreateShare(decimal? coinValue = null, decimal? price = null) => new(
        Guid.NewGuid(),
        Guid.NewGuid(),
        Guid.NewGuid(),
        Guid.NewGuid(),
        Guid.NewGuid(),
        Guid.NewGuid(),
        "share-1",
        "user-hardware",
        "job-1",
        "nonce-1",
        null,
        1.5m,
        "target",
        "hash",
        Now,
        coinValue,
        price,
        null,
        Now);

    [Fact]
    public void NewShare_IsPendingWithItsProofOfWorkDetail()
    {
        var share = CreateShare();

        share.RewardStatus.Should().Be(ShareRewardStatus.Pending);
        share.ShareIdentifier.Should().Be("share-1");
        share.WorkerIdentifier.Should().Be("user-hardware");
        share.Difficulty.Should().Be(1.5m);
        share.ProcessedAt.Should().BeNull();
        share.Reason.Should().BeNull();
    }

    [Fact]
    public void MarkProcessing_OnlyClaimsAPendingShare()
    {
        var share = CreateShare();

        share.MarkProcessing().Should().BeTrue();
        share.MarkProcessing().Should().BeFalse();
        share.RewardStatus.Should().Be(ShareRewardStatus.Processing);
    }

    [Fact]
    public void Accept_CompletesTheReward()
    {
        var share = CreateShare();
        var acceptedAt = Now.AddMinutes(1);

        share.MarkProcessing();
        share.Accept(acceptedAt);

        share.RewardStatus.Should().Be(ShareRewardStatus.Accepted);
        share.ProcessedAt.Should().Be(acceptedAt);
        share.Reason.Should().BeNull();
    }

    [Fact]
    public void Reject_IsTerminalAndRecordsWhy()
    {
        var share = CreateShare();
        var rejectedAt = Now.AddMinutes(2);

        share.MarkProcessing();
        share.Reject("pool_coin_relationship_invalid", rejectedAt);

        share.RewardStatus.Should().Be(ShareRewardStatus.Rejected);
        share.Reason.Should().Be("pool_coin_relationship_invalid");
        share.ProcessedAt.Should().Be(rejectedAt);
    }

    [Fact]
    public void RevertToPending_ReturnsTheShareToTheQueueWithAReason()
    {
        var share = CreateShare();

        share.MarkProcessing();
        share.RevertToPending("coin_price_unavailable");

        share.RewardStatus.Should().Be(ShareRewardStatus.Pending);
        share.Reason.Should().Be("coin_price_unavailable");
        share.ProcessedAt.Should().BeNull();

        // Being pending again, it can be claimed once more.
        share.MarkProcessing().Should().BeTrue();
    }

    [Fact]
    public void Redeem_RequiresAnAcceptedShareAndClearsTheReason()
    {
        var share = CreateShare();
        var redeemedAt = Now.AddDays(1);

        share.MarkProcessing();
        share.Accept(Now.AddMinutes(1));
        share.Redeem(redeemedAt);

        share.RewardStatus.Should().Be(ShareRewardStatus.Redeemed);
        share.ProcessedAt.Should().Be(redeemedAt);
    }

    [Fact]
    public void RecordCoinValue_FillsInALateKnownAmount()
    {
        var share = CreateShare();

        share.CoinValue.Should().BeNull();

        share.RecordCoinValue(12.5m);

        share.CoinValue.Should().Be(12.5m);
    }

    [Fact]
    public void ValuationSnapshotIsKeptAsSupplied()
    {
        var share = CreateShare(coinValue: 4m, price: 2.5m);

        share.CoinValue.Should().Be(4m);
        share.ApproxUsdValueAtTime.Should().Be(2.5m);
    }
}

public sealed class ShareRewardEnumTests
{
    [Theory]
    [InlineData(ShareRewardStatus.Pending, "pending")]
    [InlineData(ShareRewardStatus.Processing, "processing")]
    [InlineData(ShareRewardStatus.Accepted, "accepted")]
    [InlineData(ShareRewardStatus.Redeemed, "redeemed")]
    [InlineData(ShareRewardStatus.Rejected, "rejected")]
    public void RewardStatus_RoundTrips(ShareRewardStatus status, string expected)
    {
        status.ToDbValue().Should().Be(expected);
        ShareRewardStatusDbValues.FromDbValue(expected).Should().Be(status);
    }

    [Theory]
    [InlineData(StatisticPeriod.Day1, "d1")]
    [InlineData(StatisticPeriod.Day30, "d30")]
    [InlineData(StatisticPeriod.Day60, "d60")]
    [InlineData(StatisticPeriod.AllTime, "all")]
    public void StatisticPeriod_RoundTrips(StatisticPeriod period, string expected)
    {
        period.ToDbValue().Should().Be(expected);
        StatisticPeriodDbValues.FromDbValue(expected).Should().Be(period);
    }
}
