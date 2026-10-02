using System.Net.Http.Json;
using FluentAssertions;
using MediatR;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.DependencyInjection;
using TokenMiner.Application.Providers.Commands;
using TokenMiner.Application.Treasury.Commands;
using TokenMiner.Contracts.Mining;
using TokenMiner.Contracts.Providers;
using TokenMiner.Contracts.Treasury;
using TokenMiner.Domain.Providers.Enums;
using TokenMiner.Domain.Treasury;
using TokenMiner.Infrastructure.Persistence;
using Xunit;

namespace TokenMiner.IntegrationTests;

[Collection(ApiCollection.Name)]
public sealed class ProviderDepositPipelineTests(ApiFactory factory)
{
    private const string StablecoinNetwork = "bep20";

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

    private static async Task<CoinResponse> CreateCoinAsync(HttpClient admin, string network)
    {
        var response = await admin.PostAsJsonAsync(
            "/api/admin/coins",
            new CreateCoinRequest($"coin-{MiningTestData.Suffix()}", "Test Coin", network, 6));

        return (await response.Content.ReadFromJsonAsync<CoinResponse>())!;
    }

    private async Task<Guid> ConfigureProviderAsync(HttpClient admin, Guid stablecoinId)
    {
        // The gateway the deposit pipeline tops up.
        var providerResponse = await admin.PostAsJsonAsync(
            "/api/admin/llm-providers",
            new CreateLlmProviderRequest($"router-one-{MiningTestData.Suffix()}", "https://api.router.one/v1", "router-one-key"));

        providerResponse.StatusCode.Should().Be(System.Net.HttpStatusCode.Created);
        var provider = (await providerResponse.Content.ReadFromJsonAsync<LlmProviderResponse>())!;

        // The deposit rail is identified by coin *and* network.
        var accountResponse = await admin.PutAsJsonAsync(
            $"/api/admin/llm-providers/{provider.Id}/deposit-accounts",
            new UpsertDepositAccountRequest(stablecoinId, StablecoinNetwork, "0xdeposit-address", "active"));

        accountResponse.StatusCode.Should().Be(System.Net.HttpStatusCode.OK);

        return provider.Id;
    }

    private static async Task<Guid> ConfigureConversionAsync(
        HttpClient admin,
        Guid sourceCoinId,
        Guid stablecoinId)
    {
        var providerResponse = await admin.PostAsJsonAsync(
            "/api/admin/conversion-providers",
            new CreateConversionProviderRequest("simulated", "https://example.invalid", 10, null, null));

        var provider = (await providerResponse.Content.ReadFromJsonAsync<ConversionProviderResponse>())!;

        var routeResponse = await admin.PostAsJsonAsync(
            "/api/admin/conversion-routes",
            new CreateConversionRouteRequest(provider.Id, sourceCoinId, stablecoinId, null, StablecoinNetwork, 10, 0m));

        routeResponse.StatusCode.Should().Be(System.Net.HttpStatusCode.Created);

        return provider.Id;
    }

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

    [Fact]
    public async Task ADepositIsBroadcastConfirmedAndOnlyCreditedOnceTheBalanceGrows()
    {
        factory.LlmProvider.Reset();

        var admin = await factory.CreateAdminClientAsync();
        var setup = await MiningTestData.SeedAsync(admin);

        var stablecoin = await CreateCoinAsync(admin, StablecoinNetwork);

        await ConfigureProviderAsync(admin, stablecoin.Id);
        await ConfigureConversionAsync(admin, setup.Coin.Id, stablecoin.Id);

        var payoutId = await CreateConfirmedPayoutAsync(setup.Pool.Id, setup.Coin.Id, amount: 4m);

        // Get a completed conversion first: deposits are driven from those.
        await WithSenderAsync(sender => sender.Send(new RunConversionPipelineCommand(50)));

        // The balance before any transfer, which is what credit reconciliation compares against.
        factory.LlmProvider.Balance = new Application.Providers.Abstractions.LlmBalance(0m, 0m, 0m);

        var queued = await WithSenderAsync(sender => sender.Send(new RunProviderDepositPipelineCommand()));

        queued.DepositsQueued.Should().Be(1);
        queued.DepositsAdvanced.Should().Be(0);

        var broadcast = await WithDbAsync(dbContext => dbContext.ProviderDeposits
            .SingleAsync(deposit => deposit.IdempotencyKey.StartsWith("conversion:")));

        broadcast.Status.Should().Be(ProviderDepositStatus.Broadcast);
        broadcast.TransactionHash.Should().NotBeNullOrWhiteSpace();
        broadcast.Address.Should().Be("0xdeposit-address");
        broadcast.Amount.Should().Be(4m);

        // The chain reports enough confirmations, but the provider has not credited yet.
        await WithSenderAsync(sender => sender.Send(new RunProviderDepositPipelineCommand()));

        var confirmed = await WithDbAsync(dbContext => dbContext.ProviderDeposits.SingleAsync(
            deposit => deposit.Id == broadcast.Id));

        confirmed.Status.Should().Be(ProviderDepositStatus.Confirmed);

        // Only a balance increase proves the credit.
        factory.LlmProvider.Balance = new Application.Providers.Abstractions.LlmBalance(4m, 0m, 4m);

        await WithSenderAsync(sender => sender.Send(new RunProviderDepositPipelineCommand()));

        var credited = await WithDbAsync(dbContext => dbContext.ProviderDeposits.SingleAsync(
            deposit => deposit.Id == broadcast.Id));

        credited.Status.Should().Be(ProviderDepositStatus.Credited);
        credited.ProviderCreditAfter.Should().Be(4m);

        // A further pass changes nothing: one conversion funds one transfer.
        await WithSenderAsync(sender => sender.Send(new RunProviderDepositPipelineCommand()));

        var depositCount = await WithDbAsync(dbContext => dbContext.ProviderDeposits.CountAsync(
            deposit => deposit.IdempotencyKey.StartsWith("conversion:")));

        depositCount.Should().Be(1);
    }

    [Fact]
    public async Task TheModelCatalogueIsMirroredAndStaleEntriesAreDisabled()
    {
        factory.LlmProvider.Reset();

        var admin = await factory.CreateAdminClientAsync();
        var provider = await admin.PostAsJsonAsync(
            "/api/admin/llm-providers",
            new CreateLlmProviderRequest($"router-one-{MiningTestData.Suffix()}", "https://api.router.one/v1", null));
        var created = (await provider.Content.ReadFromJsonAsync<LlmProviderResponse>())!;

        factory.LlmProvider.Models.Add(new Application.Providers.Abstractions.LlmCatalogModel(
            "vendor/model-a",
            "Model A",
            1m,
            2m,
            null,
            "USD",
            128000,
            ["chat"]));

        await WithSenderAsync(sender => sender.Send(new SyncModelCatalogCommand()));

        var models = await admin.GetFromJsonAsync<List<LlmProviderModelResponse>>(
            $"/api/admin/llm-providers/{created.Id}/models");

        models.Should().Contain(model => model.ModelId == "vendor/model-a");

        // A catalogue without that model disables it rather than deleting it.
        factory.LlmProvider.Models.Clear();
        factory.LlmProvider.Models.Add(new Application.Providers.Abstractions.LlmCatalogModel(
            "vendor/model-b", "Model B", null, null, null, "USD", null, []));

        await WithSenderAsync(sender => sender.Send(new SyncModelCatalogCommand()));

        var afterSync = await admin.GetFromJsonAsync<List<LlmProviderModelResponse>>(
            $"/api/admin/llm-providers/{created.Id}/models");

        afterSync.Should().Contain(model => model.ModelId == "vendor/model-b" && model.Status == "active");
        afterSync.Should().Contain(model => model.ModelId == "vendor/model-a" && model.Status == "disabled");
    }
}
