using FluentAssertions;
using TokenMiner.Domain.Treasury;
using TokenMiner.Domain.Treasury.Enums;
using Xunit;

namespace TokenMiner.UnitTests.Treasury;

public sealed class ConversionTransactionTests
{
    private static readonly DateTimeOffset Now = new(2026, 1, 1, 12, 0, 0, TimeSpan.Zero);

    private static ConversionTransaction CreateTransaction() => new(
        Guid.NewGuid(),
        Guid.NewGuid(),
        Guid.NewGuid(),
        10m,
        Guid.NewGuid(),
        "payout:abc",
        Now);

    [Fact]
    public void ANewTransactionIsPendingWithItsIdempotencyKey()
    {
        var transaction = CreateTransaction();

        transaction.Status.Should().Be(ConversionTransactionStatus.Pending);
        transaction.IdempotencyKey.Should().Be("payout:abc");
        transaction.SourceAmount.Should().Be(10m);
        transaction.CompletedAt.Should().BeNull();
    }

    [Fact]
    public void CompleteRecordsTheAmountsAndTheRate()
    {
        var transaction = CreateTransaction();
        var completedAt = Now.AddMinutes(1);

        transaction.Complete(9.8m, 0.98m, 0.2m, "destination-1", completedAt);

        transaction.Status.Should().Be(ConversionTransactionStatus.Completed);
        transaction.DestinationAmount.Should().Be(9.8m);
        transaction.ExchangeRate.Should().Be(0.98m);
        transaction.Fees.Should().Be(0.2m);
        transaction.DestinationTransactionId.Should().Be("destination-1");
        transaction.CompletedAt.Should().Be(completedAt);
        transaction.Error.Should().BeNull();
    }

    [Fact]
    public void CompleteFromProviderFallsBackToWhatIsAlreadyKnown()
    {
        var transaction = CreateTransaction();
        transaction.RecordQuote(0.5m, 5m, 0.1m, Now);

        // The provider returns nothing but a destination hash.
        transaction.CompleteFromProvider(null, null, null, "destination-9", Now);

        transaction.Status.Should().Be(ConversionTransactionStatus.Completed);
        transaction.DestinationAmount.Should().Be(5m);
        transaction.ExchangeRate.Should().Be(0.5m);
        transaction.DestinationTransactionId.Should().Be("destination-9");
    }

    [Fact]
    public void MarkProcessingKeepsTheFirstSourceTransactionId()
    {
        var transaction = CreateTransaction();

        transaction.MarkProcessing("source-1", Now);
        transaction.MarkProcessing("source-2", Now.AddMinutes(1));

        transaction.Status.Should().Be(ConversionTransactionStatus.Processing);
        transaction.SourceTransactionId.Should().Be("source-1");
    }

    [Fact]
    public void FailAndCancelRecordTheirReason()
    {
        var failed = CreateTransaction();
        failed.Fail("exchange rejected the order", Now);

        failed.Status.Should().Be(ConversionTransactionStatus.Failed);
        failed.Error.Should().Be("exchange rejected the order");

        var cancelled = CreateTransaction();
        cancelled.Cancel("no longer needed", Now);

        cancelled.Status.Should().Be(ConversionTransactionStatus.Cancelled);
        cancelled.Error.Should().Be("no longer needed");
    }
}

public sealed class ConversionEnumTests
{
    [Theory]
    [InlineData(ConversionTransactionStatus.Pending, "pending")]
    [InlineData(ConversionTransactionStatus.Processing, "processing")]
    [InlineData(ConversionTransactionStatus.Completed, "completed")]
    [InlineData(ConversionTransactionStatus.Failed, "failed")]
    [InlineData(ConversionTransactionStatus.Cancelled, "cancelled")]
    public void ConversionStatusRoundTrips(ConversionTransactionStatus status, string expected)
    {
        status.ToDbValue().Should().Be(expected);
        ConversionTransactionStatusDbValues.FromDbValue(expected).Should().Be(status);
    }

    [Fact]
    public void InFlightCoversPendingAndProcessing()
    {
        ConversionTransactionStatusDbValues.InFlight
            .Select(status => status.ToDbValue())
            .Should()
            .BeEquivalentTo(["pending", "processing"]);
    }

    [Theory]
    [InlineData("active", true)]
    [InlineData("disabled", true)]
    [InlineData("enabled", false)]
    [InlineData(null, false)]
    public void TreasuryStatusTryParseOnlyAcceptsKnownValues(string? value, bool expected)
    {
        TreasuryStatusDbValues.TryParseDbValue(value, out _).Should().Be(expected);
    }
}
