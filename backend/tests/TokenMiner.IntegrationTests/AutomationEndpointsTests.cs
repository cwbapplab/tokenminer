using System.Net;
using System.Net.Http.Json;
using FluentAssertions;
using MediatR;
using Microsoft.Extensions.DependencyInjection;
using TokenMiner.Application.Configuration.Abstractions;
using TokenMiner.Application.Configuration.Commands;
using TokenMiner.Application.Providers.Abstractions;
using TokenMiner.Application.Providers.Commands;
using TokenMiner.Contracts.Providers;
using TokenMiner.Contracts.System;
using TokenMiner.Domain.Configuration;
using Xunit;

namespace TokenMiner.IntegrationTests;

[Collection(ApiCollection.Name)]
public sealed class AutomationEndpointsTests(ApiFactory factory)
{
    private async Task<T> WithStoreAsync<T>(Func<ISystemConfigurationStore, Task<T>> action)
    {
        await using var scope = factory.Services.CreateAsyncScope();
        return await action(scope.ServiceProvider.GetRequiredService<ISystemConfigurationStore>());
    }

    private async Task<T> WithSenderAsync<T>(Func<ISender, Task<T>> action)
    {
        await using var scope = factory.Services.CreateAsyncScope();
        return await action(scope.ServiceProvider.GetRequiredService<ISender>());
    }

    /// <summary>Values are JSON, so a bare <c>true</c> is a valid setting.</summary>
    private Task SetConfigAsync(string key, string jsonValue) =>
        WithStoreAsync(async store =>
        {
            await store.SetAsync(key, jsonValue, DateTimeOffset.UtcNow, CancellationToken.None);
            return true;
        });

    /// <summary>There must be an active provider for the top-up decision to have anything to read.</summary>
    private static async Task EnsureProviderAsync(HttpClient admin)
    {
        var existing = await admin.GetFromJsonAsync<List<LlmProviderResponse>>("/api/admin/llm-providers");

        if (existing!.Count > 0)
        {
            return;
        }

        await admin.PostAsJsonAsync(
            "/api/admin/llm-providers",
            new CreateLlmProviderRequest($"router-one-{MiningTestData.Suffix()}", "https://api.router.one/v1", null));
    }

    private async Task GiveProviderBalanceAsync(decimal balance)
    {
        factory.LlmProvider.Balance = new LlmBalance(balance, 0m, balance);
        await WithSenderAsync(sender => sender.Send(new RefreshProviderBalanceCommand()));
    }

    [Fact]
    public async Task StatusRequiresTheAdminRole()
    {
        var user = await factory.CreateUserClientAsync();

        var response = await user.GetAsync("/api/admin/system/status");

        response.StatusCode.Should().Be(HttpStatusCode.Forbidden);
    }

    [Fact]
    public async Task StatusReportsWhatIsInFlight()
    {
        var admin = await factory.CreateAdminClientAsync();

        var status = await admin.GetFromJsonAsync<SystemStatusResponse>("/api/admin/system/status");

        status.Should().NotBeNull();
        status!.CapturedAt.Should().BeCloseTo(DateTimeOffset.UtcNow, TimeSpan.FromMinutes(1));
        status.RunningSessions.Should().BeGreaterThanOrEqualTo(0);
        status.PendingShares.Should().BeGreaterThanOrEqualTo(0);
        status.ProviderDepositsInFlight.Should().BeGreaterThanOrEqualTo(0);
    }

    [Fact]
    public async Task ConfigurationsRoundTripThroughTheAdminApi()
    {
        var admin = await factory.CreateAdminClientAsync();

        var put = await admin.PutAsJsonAsync(
            $"/api/admin/system/configurations/{SystemConfigurationKeys.ProviderBalanceMinimum}",
            new SetSystemConfigurationRequest("7.5"));

        put.StatusCode.Should().Be(HttpStatusCode.OK);

        var stored = await WithStoreAsync(store => store.GetDecimalAsync(
            SystemConfigurationKeys.ProviderBalanceMinimum,
            0m,
            CancellationToken.None));

        stored.Should().Be(7.5m);

        var listed = await admin.GetFromJsonAsync<List<SystemConfigurationResponse>>(
            "/api/admin/system/configurations");

        listed.Should().Contain(setting => setting.Key == SystemConfigurationKeys.ProviderBalanceMinimum);
    }

    [Fact]
    public async Task AMalformedSettingFallsBackToTheDefault()
    {
        var admin = await factory.CreateAdminClientAsync();

        await admin.PutAsJsonAsync(
            $"/api/admin/system/configurations/{SystemConfigurationKeys.PayoutConfirmationThreshold}",
            new SetSystemConfigurationRequest("not-a-number"));

        // A bad value must not break the jobs that read it.
        var value = await WithStoreAsync(store => store.GetIntAsync(
            SystemConfigurationKeys.PayoutConfirmationThreshold,
            fallback: 6,
            CancellationToken.None));

        value.Should().Be(6);
    }

    [Fact]
    public async Task AutoTopUpDoesNothingWhileTheProviderIsWellFunded()
    {
        factory.LlmProvider.Reset();

        var admin = await factory.CreateAdminClientAsync();
        await EnsureProviderAsync(admin);

        await SetConfigAsync(SystemConfigurationKeys.AutoTopUpEnabled, "true");
        await GiveProviderBalanceAsync(50m);

        var status = await WithSenderAsync(sender => sender.Send(new RunAutoTopUpCommand()));

        status.Outcome.Should().Be(TopUpOutcome.Skipped);
        status.Balance.Should().Be(50m);

        // Thresholds fall back to the configured policy when no setting overrides them.
        status.Target.Should().Be(20m);
        status.Minimum.Should().Be(5m);
    }

    [Fact]
    public async Task AutoTopUpDrivesThePipelineWhenTheProviderIsShortOfCredit()
    {
        factory.LlmProvider.Reset();
        factory.PoolPayouts.Reset();

        var admin = await factory.CreateAdminClientAsync();
        await EnsureProviderAsync(admin);

        await SetConfigAsync(SystemConfigurationKeys.AutoTopUpEnabled, "true");
        await GiveProviderBalanceAsync(0m);

        var status = await WithSenderAsync(sender => sender.Send(new RunAutoTopUpCommand()));

        // Below the minimum, so every stage is advanced rather than skipped.
        status.Outcome.Should().Be(TopUpOutcome.Driven);
        status.PayoutsRecorded.Should().BeGreaterThanOrEqualTo(0);
    }

    [Fact]
    public async Task AutoTopUpCanBeSwitchedOff()
    {
        factory.LlmProvider.Reset();

        var admin = await factory.CreateAdminClientAsync();
        await EnsureProviderAsync(admin);

        await SetConfigAsync(SystemConfigurationKeys.AutoTopUpEnabled, "false");
        await GiveProviderBalanceAsync(0m);

        var status = await WithSenderAsync(sender => sender.Send(new RunAutoTopUpCommand()));

        status.Outcome.Should().Be(TopUpOutcome.Disabled);

        // Restore the default so later tests are unaffected.
        await SetConfigAsync(SystemConfigurationKeys.AutoTopUpEnabled, "true");
    }
}
