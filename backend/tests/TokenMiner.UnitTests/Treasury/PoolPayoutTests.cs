using FluentAssertions;
using TokenMiner.Domain.Treasury;
using TokenMiner.Domain.Treasury.Enums;
using Xunit;

namespace TokenMiner.UnitTests.Treasury;

public sealed class PoolPayoutTests
{
    private static readonly DateTimeOffset Now = new(2026, 1, 1, 12, 0, 0, TimeSpan.Zero);

    private static PoolPayout CreatePayout(
        decimal amount = 10m,
        string? transactionHash = "tx-1") => new(
        Guid.NewGuid(),
        Guid.NewGuid(),
        Guid.NewGuid(),
        "wallet",
        amount,
        transactionHash,
        null,
        null,
        0,
        "{}",
        Now);

    [Fact]
    public void APayoutWithATransactionIsAwaitingConfirmations()
    {
        var payout = CreatePayout();

        payout.Status.Should().Be(PoolPayoutStatus.AwaitingConfirmations);
        payout.IsConfirmed.Should().BeFalse();
    }

    [Fact]
    public void APayoutWithoutATransactionIsPending()
    {
        var payout = CreatePayout(transactionHash: null);

        payout.Status.Should().Be(PoolPayoutStatus.Pending);
    }

    [Fact]
    public void ObserveOnChainRecordsTheHashAndConfirmations()
    {
        var payout = CreatePayout(transactionHash: null);

        payout.ObserveOnChain("tx-9", 3, Now.AddMinutes(1));

        payout.TransactionHash.Should().Be("tx-9");
        payout.ConfirmationsCount.Should().Be(3);
        payout.Status.Should().Be(PoolPayoutStatus.AwaitingConfirmations);
    }

    [Fact]
    public void ConfirmIsIdempotent()
    {
        var payout = CreatePayout();
        var confirmedAt = Now.AddMinutes(5);

        payout.Confirm(confirmedAt, confirmedAt).Should().BeTrue();
        payout.Confirm(confirmedAt.AddMinutes(1), confirmedAt.AddMinutes(1)).Should().BeFalse();

        payout.Status.Should().Be(PoolPayoutStatus.Confirmed);
        payout.IsConfirmed.Should().BeTrue();
        payout.ReceivedAt.Should().Be(confirmedAt);
    }

    [Fact]
    public void FailRecordsTheReason()
    {
        var payout = CreatePayout();

        payout.Fail("pool rejected the request", Now);

        payout.Status.Should().Be(PoolPayoutStatus.Failed);
        payout.RawData.Should().Be("pool rejected the request");
    }

    [Fact]
    public void RecordRedeemedAccumulatesAndNeverExceedsThePayout()
    {
        var payout = CreatePayout(amount: 10m);

        payout.RemainingAmount.Should().Be(10m);

        payout.RecordRedeemed(4m, Now);
        payout.RedeemedAmount.Should().Be(4m);
        payout.RemainingAmount.Should().Be(6m);

        // Over-attribution is clamped, so a payout can never redeem more than it carried.
        payout.RecordRedeemed(100m, Now);
        payout.RedeemedAmount.Should().Be(10m);
        payout.RemainingAmount.Should().Be(0m);
    }

    [Fact]
    public void RecordRedeemedIgnoresNonPositiveAmounts()
    {
        var payout = CreatePayout(amount: 10m);

        payout.RecordRedeemed(0m, Now);
        payout.RecordRedeemed(-5m, Now);

        payout.RedeemedAmount.Should().Be(0m);
    }
}

public sealed class PoolPayoutStatusTests
{
    [Theory]
    [InlineData(PoolPayoutStatus.Pending, "pending")]
    [InlineData(PoolPayoutStatus.Processing, "processing")]
    [InlineData(PoolPayoutStatus.AwaitingConfirmations, "awaiting_confirmations")]
    [InlineData(PoolPayoutStatus.Confirmed, "confirmed")]
    [InlineData(PoolPayoutStatus.Failed, "failed")]
    public void StatusRoundTrips(PoolPayoutStatus status, string expected)
    {
        status.ToDbValue().Should().Be(expected);
        PoolPayoutStatusDbValues.FromDbValue(expected).Should().Be(status);
    }

    [Fact]
    public void UnsettledCoversEverythingExceptConfirmedAndFailed()
    {
        PoolPayoutStatusDbValues.Unsettled
            .Select(status => status.ToDbValue())
            .Should()
            .BeEquivalentTo(["pending", "processing", "awaiting_confirmations"]);
    }
}
