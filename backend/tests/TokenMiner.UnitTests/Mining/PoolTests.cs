using FluentAssertions;
using TokenMiner.Domain.Mining;
using TokenMiner.Domain.Mining.Enums;
using Xunit;

namespace TokenMiner.UnitTests.Mining;

public sealed class PoolTests
{
    private static readonly DateTimeOffset Now = new(2026, 1, 1, 12, 0, 0, TimeSpan.Zero);

    private static readonly Guid CoinId = Guid.NewGuid();

    [Fact]
    public void NewPool_IsActive()
    {
        var pool = new Pool(Guid.NewGuid(), "kryptex-qtc", "Kryptex QTC", "kryptex", Now);

        pool.Status.Should().Be(MiningStatus.Active);
        pool.SystemPoolId.Should().Be("kryptex-qtc");
        pool.Provider.Should().Be("kryptex");
        pool.CreatedAt.Should().Be(Now);
    }

    [Fact]
    public void Pool_UpdateDetails_BumpsUpdatedAt()
    {
        var pool = new Pool(Guid.NewGuid(), "kryptex-qtc", "Kryptex QTC", "kryptex", Now);
        var later = Now.AddMinutes(5);

        pool.UpdateDetails("Kryptex QTC pool", "kryptex", MiningStatus.Disabled, later);

        pool.Name.Should().Be("Kryptex QTC pool");
        pool.Status.Should().Be(MiningStatus.Disabled);
        pool.UpdatedAt.Should().Be(later);
    }

    [Fact]
    public void NewPoolDetails_CarriesPayoutConfiguration()
    {
        var details = new PoolDetails(
            Guid.NewGuid(),
            Guid.NewGuid(),
            "https://pool.kryptex.com",
            "/api/v1/index",
            "qtc-br.kryptex.network:7049",
            CoinId,
            PayoutMode.NativeAutoPayout,
            "krxYR9NJVQ",
            "bep20",
            Now);

        details.PayoutMode.Should().Be(PayoutMode.NativeAutoPayout);
        details.StratumEndpoint.Should().Be("qtc-br.kryptex.network:7049");
        details.PayoutAddress.Should().Be("krxYR9NJVQ");
        details.PayoutNetwork.Should().Be("bep20");
        details.CoinId.Should().Be(CoinId);
    }

    [Fact]
    public void PoolDetails_Update_ReplacesConfiguration()
    {
        var details = new PoolDetails(
            Guid.NewGuid(), Guid.NewGuid(), "https://old", null, null, CoinId,
            PayoutMode.ManagedRequest, null, null, Now);

        var later = Now.AddMinutes(1);
        var newCoin = Guid.NewGuid();

        details.Update("https://new", "/status", "pool.example.com:3333", newCoin, PayoutMode.DirectToWallet, "0xabc", "bsc", later);

        details.BaseUrl.Should().Be("https://new");
        details.StatusEndpoint.Should().Be("/status");
        details.StratumEndpoint.Should().Be("pool.example.com:3333");
        details.CoinId.Should().Be(newCoin);
        details.PayoutMode.Should().Be(PayoutMode.DirectToWallet);
        details.PayoutAddress.Should().Be("0xabc");
        details.PayoutNetwork.Should().Be("bsc");
        details.UpdatedAt.Should().Be(later);
        details.CreatedAt.Should().Be(Now);
    }
}

public sealed class UserHardwareTests
{
    private static readonly DateTimeOffset Now = new(2026, 1, 1, 12, 0, 0, TimeSpan.Zero);

    [Fact]
    public void NewHardware_StartsActiveAndSeenNow()
    {
        var hardwareId = Guid.NewGuid();
        var hardware = new UserHardware(Guid.NewGuid(), Guid.NewGuid(), hardwareId, "Desktop", Now);

        hardware.HardwareId.Should().Be(hardwareId);
        hardware.Status.Should().Be(MiningStatus.Active);
        hardware.LastSeenAt.Should().Be(Now);
        hardware.Name.Should().Be("Desktop");
    }

    [Fact]
    public void Touch_OnlyUpdatesLastSeen()
    {
        var hardware = new UserHardware(Guid.NewGuid(), Guid.NewGuid(), Guid.NewGuid(), null, Now);
        var later = Now.AddHours(2);

        hardware.Touch(later);

        hardware.LastSeenAt.Should().Be(later);
        hardware.CreatedAt.Should().Be(Now);
    }

    [Fact]
    public void Update_ReplacesNameAndStatus()
    {
        var hardware = new UserHardware(Guid.NewGuid(), Guid.NewGuid(), Guid.NewGuid(), null, Now);
        var later = Now.AddHours(1);

        hardware.Update("Rig 2", MiningStatus.Disabled, later);

        hardware.Name.Should().Be("Rig 2");
        hardware.Status.Should().Be(MiningStatus.Disabled);
        hardware.LastSeenAt.Should().Be(later);
    }
}
