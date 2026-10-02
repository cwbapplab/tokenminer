using FluentAssertions;
using TokenMiner.Domain.Mining.Enums;
using Xunit;

namespace TokenMiner.UnitTests.Mining;

public sealed class MiningEnumTests
{
    [Theory]
    [InlineData(MiningStatus.Active, "active")]
    [InlineData(MiningStatus.Disabled, "disabled")]
    public void MiningStatus_RoundTripsThroughItsDatabaseValue(MiningStatus status, string expected)
    {
        status.ToDbValue().Should().Be(expected);
        MiningStatusDbValues.FromDbValue(expected).Should().Be(status);
    }

    [Theory]
    [InlineData(PayoutMode.ManagedRequest, "managed_request")]
    [InlineData(PayoutMode.NativeAutoPayout, "native_auto_payout")]
    [InlineData(PayoutMode.DirectToWallet, "direct_to_wallet")]
    public void PayoutMode_RoundTripsThroughItsDatabaseValue(PayoutMode mode, string expected)
    {
        mode.ToDbValue().Should().Be(expected);
        PayoutModeDbValues.FromDbValue(expected).Should().Be(mode);
    }

    [Theory]
    [InlineData("active", true)]
    [InlineData("disabled", true)]
    [InlineData("Active", false)]
    [InlineData("enabled", false)]
    [InlineData("", false)]
    [InlineData(null, false)]
    public void MiningStatus_TryParseOnlyAcceptsKnownValues(string? value, bool expected)
    {
        MiningStatusDbValues.TryParseDbValue(value, out _).Should().Be(expected);
    }

    [Theory]
    [InlineData("managed_request", true)]
    [InlineData("native_auto_payout", true)]
    [InlineData("direct_to_wallet", true)]
    [InlineData("managedRequest", false)]
    [InlineData("unknown", false)]
    [InlineData(null, false)]
    public void PayoutMode_TryParseOnlyAcceptsKnownValues(string? value, bool expected)
    {
        PayoutModeDbValues.TryParseDbValue(value, out _).Should().Be(expected);
    }

    [Fact]
    public void FromDbValue_ThrowsForUnknownValue()
    {
        var act = () => MiningStatusDbValues.FromDbValue("nope");

        act.Should().Throw<ArgumentOutOfRangeException>();
    }
}
