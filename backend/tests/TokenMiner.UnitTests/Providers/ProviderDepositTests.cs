using FluentAssertions;
using TokenMiner.Domain.Providers;
using TokenMiner.Domain.Providers.Enums;
using Xunit;

namespace TokenMiner.UnitTests.Providers;

public sealed class ProviderDepositTests
{
    private static readonly DateTimeOffset Now = new(2026, 1, 1, 12, 0, 0, TimeSpan.Zero);

    private static ProviderDeposit CreateDeposit(decimal amount = 5m) => new(
        Guid.NewGuid(),
        Guid.NewGuid(),
        Guid.NewGuid(),
        "bep20",
        "0xdeposit",
        amount,
        "conversion:abc",
        Now);

    [Fact]
    public void ANewDepositStartsPending()
    {
        var deposit = CreateDeposit();

        deposit.Status.Should().Be(ProviderDepositStatus.Pending);
        deposit.IdempotencyKey.Should().Be("conversion:abc");
        deposit.TransactionHash.Should().BeNull();
        deposit.Confirmations.Should().Be(0);
    }

    [Fact]
    public void BroadcastRecordsTheHashAndTheBalanceBefore()
    {
        var deposit = CreateDeposit();

        deposit.Broadcast("tx-1", providerCreditBefore: 3m, Now);

        deposit.Status.Should().Be(ProviderDepositStatus.Broadcast);
        deposit.TransactionHash.Should().Be("tx-1");
        deposit.ProviderCreditBefore.Should().Be(3m);
    }

    [Fact]
    public void ObservingConfirmationsMovesTheDepositToAwaitingConfirmations()
    {
        var deposit = CreateDeposit();
        deposit.Broadcast("tx-1", 0m, Now);

        deposit.ObserveConfirmations(2, Now.AddMinutes(1));

        deposit.Status.Should().Be(ProviderDepositStatus.AwaitingConfirmations);
        deposit.Confirmations.Should().Be(2);
    }

    [Fact]
    public void ConfirmIsIdempotent()
    {
        var deposit = CreateDeposit();
        deposit.Broadcast("tx-1", 0m, Now);

        deposit.Confirm(Now.AddMinutes(2)).Should().BeTrue();
        deposit.Confirm(Now.AddMinutes(3)).Should().BeFalse();

        deposit.Status.Should().Be(ProviderDepositStatus.Confirmed);
        deposit.ConfirmedAt.Should().Be(Now.AddMinutes(2));
    }

    [Fact]
    public void ACreditedDepositCannotBeReconfirmed()
    {
        var deposit = CreateDeposit();
        deposit.Broadcast("tx-1", 0m, Now);
        deposit.Confirm(Now);

        deposit.MarkCredited(5m, Now.AddMinutes(1));

        deposit.Status.Should().Be(ProviderDepositStatus.Credited);
        deposit.ProviderCreditAfter.Should().Be(5m);
        deposit.Confirm(Now.AddMinutes(2)).Should().BeFalse();
        deposit.Status.Should().Be(ProviderDepositStatus.Credited);
    }

    [Fact]
    public void FailedDepositsRecordTheReason()
    {
        var deposit = CreateDeposit();

        deposit.Fail("broadcast rejected", Now);

        deposit.Status.Should().Be(ProviderDepositStatus.Failed);
        deposit.Error.Should().Be("broadcast rejected");
    }
}

public sealed class ProviderDepositStatusTests
{
    [Theory]
    [InlineData(ProviderDepositStatus.Pending, "pending")]
    [InlineData(ProviderDepositStatus.Broadcast, "broadcast")]
    [InlineData(ProviderDepositStatus.AwaitingConfirmations, "awaiting_confirmations")]
    [InlineData(ProviderDepositStatus.Confirmed, "confirmed")]
    [InlineData(ProviderDepositStatus.Credited, "credited")]
    [InlineData(ProviderDepositStatus.Failed, "failed")]
    public void StatusRoundTrips(ProviderDepositStatus status, string expected)
    {
        status.ToDbValue().Should().Be(expected);
        ProviderDepositStatusDbValues.FromDbValue(expected).Should().Be(status);
    }

    [Fact]
    public void CreditedIsNotInFlight()
    {
        ProviderDepositStatusDbValues.InFlight.Should().NotContain(ProviderDepositStatus.Credited);
        ProviderDepositStatusDbValues.InFlight.Should().NotContain(ProviderDepositStatus.Failed);
    }
}
