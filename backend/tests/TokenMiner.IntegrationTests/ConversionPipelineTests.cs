using System.Net;
using System.Net.Http.Json;
using FluentAssertions;
using MediatR;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.DependencyInjection;
using TokenMiner.Application.Treasury.Commands;
using TokenMiner.Application.Treasury.Queries;
using TokenMiner.Contracts.Mining;
using TokenMiner.Contracts.Treasury;
using TokenMiner.Domain.Treasury;
using TokenMiner.Domain.Treasury.Enums;
using TokenMiner.Infrastructure.Persistence;
using Xunit;

namespace TokenMiner.IntegrationTests;

[Collection(ApiCollection.Name)]
public sealed class ConversionPipelineTests(ApiFactory factory)
{
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

    /// <summary>A settled payout waiting to be converted.</summary>
    private async Task<Guid> CreateConfirmedPayoutAsync(Guid poolId, Guid coinId, decimal amount) =>
        await WithDbAsync(async dbContext =>
        {
            var now = DateTimeOffset.UtcNow;

            var payout = new PoolPayout(
                Guid.NewGuid(),
                poolId,
                coinId,
                "wallet",
                amount,
                $"tx-{Guid.NewGuid():N}",
                null,
                now,
                1,
                "{}",
                now);

            payout.Confirm(now, now);

            dbContext.PoolPayouts.Add(payout);
            await dbContext.SaveChangesAsync();

            return payout.Id;
        });

    /// <summary>
    /// Returns an active conversion provider with the given name, creating it when absent. The
    /// name decides which adapter serves it: <c>simulated</c> is registered in-process, any
    /// other name has no adapter.
    /// </summary>
    private static async Task<Guid> EnsureProviderAsync(HttpClient admin, string name)
    {
        var existing = await admin.GetFromJsonAsync<List<ConversionProviderResponse>>(
            "/api/admin/conversion-providers");

        var match = existing!.FirstOrDefault(provider => provider.Name == name);
        if (match is not null)
        {
            return match.Id;
        }

        var response = await admin.PostAsJsonAsync(
            "/api/admin/conversion-providers",
            new CreateConversionProviderRequest(name, "https://example.invalid", 10, null, null));

        response.StatusCode.Should().Be(HttpStatusCode.Created);

        var created = await response.Content.ReadFromJsonAsync<ConversionProviderResponse>();

        return created!.Id;
    }

    /// <summary>Creates a destination coin and a route from the mined coin through a provider.</summary>
    private static async Task<Guid> ConfigureRouteAsync(HttpClient admin, Guid providerId, Guid sourceCoinId)
    {
        var destinationResponse = await admin.PostAsJsonAsync(
            "/api/admin/coins",
            new CreateCoinRequest($"usdt-{MiningTestData.Suffix()}", "Tether", "bep20", 6));

        var destination = (await destinationResponse.Content.ReadFromJsonAsync<CoinResponse>())!;

        var routeResponse = await admin.PostAsJsonAsync(
            "/api/admin/conversion-routes",
            new CreateConversionRouteRequest(
                providerId,
                sourceCoinId,
                destination.Id,
                null,
                "bep20",
                10,
                0m));

        routeResponse.StatusCode.Should().Be(HttpStatusCode.Created);

        return destination.Id;
    }

    [Fact]
    public async Task PipelineConvertsASettledPayoutExactlyOnce()
    {
        var admin = await factory.CreateAdminClientAsync();
        var setup = await MiningTestData.SeedAsync(admin);

        var providerId = await EnsureProviderAsync(admin, "simulated");
        await ConfigureRouteAsync(admin, providerId, setup.Coin.Id);

        var payoutId = await CreateConfirmedPayoutAsync(setup.Pool.Id, setup.Coin.Id, amount: 4m);

        await WithSenderAsync(sender => sender.Send(new RunConversionPipelineCommand(50)));

        var conversions = await WithSenderAsync(sender => sender.Send(new ListConversionsQuery(200)));
        var mine = conversions.Where(item => item.IdempotencyKey == $"payout:{payoutId}").ToList();

        mine.Should().HaveCount(1);
        mine[0].Status.Should().Be("completed");
        mine[0].SourceAmount.Should().Be(4m);

        // The simulated adapter converts at a 1:1 rate by default.
        mine[0].DestinationAmount.Should().Be(4m);
        mine[0].DestinationTransactionId.Should().NotBeNullOrWhiteSpace();

        // A second pass sees the payout already converted and adds nothing.
        await WithSenderAsync(sender => sender.Send(new RunConversionPipelineCommand(50)));

        var afterSecondPass = await WithSenderAsync(sender => sender.Send(new ListConversionsQuery(200)));

        afterSecondPass.Count(item => item.IdempotencyKey == $"payout:{payoutId}").Should().Be(1);
    }

    [Fact]
    public async Task PayoutWithoutARouteStaysUnconverted()
    {
        var admin = await factory.CreateAdminClientAsync();
        var setup = await MiningTestData.SeedAsync(admin);

        // No route is configured for this coin.
        var payoutId = await CreateConfirmedPayoutAsync(setup.Pool.Id, setup.Coin.Id, amount: 3m);

        await WithSenderAsync(sender => sender.Send(new RunConversionPipelineCommand(50)));

        var conversions = await WithSenderAsync(sender => sender.Send(new ListConversionsQuery(200)));

        conversions.Should().NotContain(item => item.IdempotencyKey == $"payout:{payoutId}");
    }

    [Fact]
    public async Task AProviderWithoutAnAdapterLeavesTheConversionForManualSettlement()
    {
        var admin = await factory.CreateAdminClientAsync();
        var setup = await MiningTestData.SeedAsync(admin);

        // A name with no registered adapter, i.e. a swap an operator performs by hand.
        var providerId = await EnsureProviderAsync(admin, $"manual-{MiningTestData.Suffix()}");
        await ConfigureRouteAsync(admin, providerId, setup.Coin.Id);

        var payoutId = await CreateConfirmedPayoutAsync(setup.Pool.Id, setup.Coin.Id, amount: 6m);

        await WithSenderAsync(sender => sender.Send(new RunConversionPipelineCommand(50)));

        var queued = await WithDbAsync(dbContext => dbContext.ConversionTransactions.SingleAsync(
            transaction => transaction.IdempotencyKey == $"payout:{payoutId}"));

        queued.Status.Should().Be(ConversionTransactionStatus.Pending);

        var settlement = await admin.PostAsJsonAsync(
            $"/api/admin/conversions/{queued.Id}/settlement",
            new SettleConversionRequest("completed", 5.9m, 0.98m, 0.1m, "off-system-1", null));

        settlement.StatusCode.Should().Be(HttpStatusCode.NoContent);

        var settled = await WithDbAsync(dbContext => dbContext.ConversionTransactions.SingleAsync(
            transaction => transaction.Id == queued.Id));

        settled.Status.Should().Be(ConversionTransactionStatus.Completed);
        settled.DestinationAmount.Should().Be(5.9m);
        settled.DestinationTransactionId.Should().Be("off-system-1");
    }
}
