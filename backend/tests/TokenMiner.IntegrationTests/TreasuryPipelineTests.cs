using System.Net.Http.Json;
using FluentAssertions;
using MediatR;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.DependencyInjection;
using TokenMiner.Application.Treasury.Abstractions;
using TokenMiner.Application.Treasury.Commands;
using TokenMiner.Contracts.Auth;
using TokenMiner.Contracts.Mining;
using TokenMiner.Domain.Mining;
using TokenMiner.Domain.Mining.Enums;
using TokenMiner.Domain.Treasury;
using TokenMiner.Domain.Treasury.Enums;
using TokenMiner.Infrastructure.Persistence;
using Xunit;

namespace TokenMiner.IntegrationTests;

[Collection(ApiCollection.Name)]
public sealed class TreasuryPipelineTests(ApiFactory factory)
{
    private sealed record Fixture(Guid UserId, Guid HardwareRowId, Guid SessionId, MiningTestData.Setup Setup);

    private async Task<T> WithDbAsync<T>(Func<AppDbContext, Task<T>> action)
    {
        await using var scope = factory.Services.CreateAsyncScope();
        return await action(scope.ServiceProvider.GetRequiredService<AppDbContext>());
    }

    private async Task<T> WithSenderAsync<T>(Func<ISender, Task<T>> action)
    {
        await using var scope = factory.Services.CreateAsyncScope();
        return await action(scope.ServiceProvider.GetRequiredService<ISender>());
    }

    private async Task<Fixture> StartMiningAsync()
    {
        var admin = await factory.CreateAdminClientAsync();
        var setup = await MiningTestData.SeedAsync(admin);

        var user = await factory.CreateUserClientAsync();
        var profile = await user.GetFromJsonAsync<UserProfileResponse>("/api/auth/me");

        var hardwareId = Guid.NewGuid();

        var start = await user.PostAsJsonAsync(
            "/api/mining/start",
            new StartMiningRequest(hardwareId, setup.Pool.Id));
        start.EnsureSuccessStatusCode();

        return await WithDbAsync(async dbContext =>
        {
            var hardware = await dbContext.UserHardware.SingleAsync(
                candidate => candidate.UserId == profile!.Id && candidate.HardwareId == hardwareId);

            var session = await dbContext.UserHardwareMiners.SingleAsync(
                candidate => candidate.UserHardwareId == hardware.Id);

            return new Fixture(profile!.Id, hardware.Id, session.Id, setup);
        });
    }

    /// <summary>Creates an accepted share directly, bypassing the signed ingest path.</summary>
    private async Task CreateAcceptedShareAsync(Fixture fixture, decimal coinValue, decimal price, int ageMinutes)
    {
        await WithDbAsync(async dbContext =>
        {
            var at = DateTimeOffset.UtcNow.AddMinutes(-ageMinutes);

            var share = new UserMiningShare(
                Guid.NewGuid(),
                fixture.UserId,
                fixture.HardwareRowId,
                fixture.SessionId,
                fixture.Setup.Pool.Id,
                fixture.Setup.Coin.Id,
                $"share-{Guid.NewGuid():N}",
                "worker",
                "job",
                "nonce",
                null,
                1m,
                "target",
                "hash",
                at,
                coinValue,
                price,
                null,
                at);

            share.MarkProcessing();
            share.Accept(at);

            dbContext.UserMiningShares.Add(share);
            await dbContext.SaveChangesAsync();

            return true;
        });
    }

    [Fact]
    public async Task MonitorRecordsReportedPayoutsAndNeverDuplicatesThem()
    {
        factory.PoolPayouts.Reset();

        var fixture = await StartMiningAsync();
        var transactionHash = $"tx-{Guid.NewGuid():N}";

        factory.PoolPayouts.Payouts.Add(new PoolPayoutRecord(
            transactionHash,
            5m,
            null,
            DateTimeOffset.UtcNow,
            1,
            IsSettled: true,
            "confirmed"));

        await WithSenderAsync(sender => sender.Send(new MonitorPoolsCommand()));

        var recorded = await WithDbAsync(dbContext => dbContext.PoolPayouts
            .Where(payout => payout.PoolId == fixture.Setup.Pool.Id && payout.TransactionHash == transactionHash)
            .ToListAsync());

        recorded.Should().HaveCount(1);
        recorded[0].Amount.Should().Be(5m);
        recorded[0].Status.Should().Be(PoolPayoutStatus.Confirmed);

        // A second pass sees the same payout and stores nothing new.
        await WithSenderAsync(sender => sender.Send(new MonitorPoolsCommand()));

        var afterSecondPass = await WithDbAsync(dbContext => dbContext.PoolPayouts
            .CountAsync(payout => payout.PoolId == fixture.Setup.Pool.Id && payout.TransactionHash == transactionHash));

        afterSecondPass.Should().Be(1);
    }

    [Fact]
    public async Task MonitorNeverWithdrawsFromAPoolThatPaysOutOnItsOwn()
    {
        factory.PoolPayouts.Reset();

        var fixture = await StartMiningAsync();

        // The seeded pool is in native_auto_payout mode, so a healthy balance changes nothing.
        factory.PoolPayouts.Balance = new PoolBalance(50m, 50m, 0m, 1m);
        factory.PoolPayouts.PlaceWithdrawals = true;

        await WithSenderAsync(sender => sender.Send(new MonitorPoolsCommand()));

        factory.PoolPayouts.WithdrawalPoolIds.Should().NotContain(fixture.Setup.Pool.Id);
    }

    [Fact]
    public async Task ReconcileSettlesPayoutsAndRedeemsEarnedSharesOnlyOnce()
    {
        factory.PoolPayouts.Reset();

        var fixture = await StartMiningAsync();

        // Three shares worth 2 coin each, against a payout of only 5.
        await CreateAcceptedShareAsync(fixture, coinValue: 2m, price: 1m, ageMinutes: 3);
        await CreateAcceptedShareAsync(fixture, coinValue: 2m, price: 1m, ageMinutes: 2);
        await CreateAcceptedShareAsync(fixture, coinValue: 2m, price: 1m, ageMinutes: 1);

        var transactionHash = $"tx-{Guid.NewGuid():N}";

        factory.PoolPayouts.Payouts.Add(new PoolPayoutRecord(
            transactionHash,
            5m,
            null,
            DateTimeOffset.UtcNow,
            1,
            IsSettled: true,
            "confirmed"));

        await WithSenderAsync(sender => sender.Send(new MonitorPoolsCommand()));
        await WithSenderAsync(sender => sender.Send(new ReconcilePayoutsCommand()));

        var payout = await WithDbAsync(dbContext => dbContext.PoolPayouts.SingleAsync(
            candidate => candidate.PoolId == fixture.Setup.Pool.Id
                && candidate.TransactionHash == transactionHash));

        // 2 + 2 fits in 5; the third share would overshoot and waits for the next payout.
        payout.RedeemedAmount.Should().Be(4m);
        payout.RemainingAmount.Should().Be(1m);

        var redeemed = await WithDbAsync(dbContext => dbContext.UserMiningShares.CountAsync(
            share => share.UserHardwareId == fixture.HardwareRowId && share.RewardStatus == ShareRewardStatus.Redeemed));

        redeemed.Should().Be(2);

        // Re-running changes nothing: shares are consumed exactly once.
        await WithSenderAsync(sender => sender.Send(new ReconcilePayoutsCommand()));

        var stillRedeemed = await WithDbAsync(dbContext => dbContext.UserMiningShares.CountAsync(
            share => share.UserHardwareId == fixture.HardwareRowId && share.RewardStatus == ShareRewardStatus.Redeemed));

        stillRedeemed.Should().Be(2);
    }

    [Fact]
    public async Task CoinPriceRefreshUpdatesTheCacheAndAppendsHistory()
    {
        factory.CoinPrices.Reset();

        var fixture = await StartMiningAsync();
        factory.CoinPrices.Price = 2.5m;

        await WithSenderAsync(sender => sender.Send(new RefreshCoinPricesCommand()));

        var coin = await WithDbAsync(dbContext => dbContext.Coins.SingleAsync(
            candidate => candidate.Id == fixture.Setup.Coin.Id));

        coin.LastKnownUsdValue.Should().Be(2.5m);
        coin.LastValueDate.Should().NotBeNull();

        var historyCount = await WithDbAsync(dbContext => dbContext.CoinPriceHistory.CountAsync(
            entry => entry.CoinId == fixture.Setup.Coin.Id));

        historyCount.Should().BeGreaterThanOrEqualTo(1);
    }
}
