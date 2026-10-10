using System.Net;
using System.Net.Http.Json;
using FluentAssertions;
using MediatR;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.DependencyInjection;
using TokenMiner.Application.Mining.Sessions;
using TokenMiner.Application.Mining.Shares;
using TokenMiner.Contracts.Admin;
using TokenMiner.Contracts.Auth;
using TokenMiner.Contracts.Mining;
using TokenMiner.Domain.Mining;
using TokenMiner.Domain.Mining.Enums;
using TokenMiner.Infrastructure.Persistence;
using Xunit;

namespace TokenMiner.IntegrationTests;

[Collection(ApiCollection.Name)]
public sealed class MiningSessionEndpointsTests(ApiFactory factory)
{
    private async Task<T> WithSenderAsync<T>(Func<ISender, Task<T>> action)
    {
        await using var scope = factory.Services.CreateAsyncScope();
        var sender = scope.ServiceProvider.GetRequiredService<ISender>();
        return await action(sender);
    }

    private async Task<UserHardwareMiner> ReadSessionAsync(Guid sessionId)
    {
        await using var scope = factory.Services.CreateAsyncScope();
        var dbContext = scope.ServiceProvider.GetRequiredService<AppDbContext>();
        return await dbContext.UserHardwareMiners.SingleAsync(session => session.Id == sessionId);
    }

    private async Task<int> CountLogsAsync(Guid sessionId, MiningEventType eventType)
    {
        await using var scope = factory.Services.CreateAsyncScope();
        var dbContext = scope.ServiceProvider.GetRequiredService<AppDbContext>();
        return await dbContext.MiningLogs
            .CountAsync(log => log.UserHardwareMinerId == sessionId && log.EventType == eventType);
    }

    /// <summary>Reports an accepted share for the device, the way the proxy does.</summary>
    private async Task RecordShareAsync(Guid hardwareId, Guid poolId, Guid coinId)
    {
        await WithSenderAsync(sender => sender.Send(new RecordShareCommand(
            hardwareId.ToString("N"),
            poolId,
            coinId,
            $"share-{Guid.NewGuid():N}",
            JobId: "job-1",
            Nonce: "nonce-1",
            Extranonce: null,
            Difficulty: 1.5m,
            Target: "target",
            ResultHash: "hash",
            ShareTimestamp: DateTimeOffset.UtcNow,
            CoinValue: null,
            PoolResponse: null)));
    }

    // --- Start ---------------------------------------------------------------------------

    [Fact]
    public async Task Start_CreatesASessionAndReturnsTheLaunchCommand()
    {
        var client = await factory.CreateUserClientAsync();
        var admin = await factory.CreateAdminClientAsync();
        var setup = await MiningTestData.SeedAsync(admin);
        var hardwareId = Guid.NewGuid();

        var response = await client.PostAsJsonAsync(
            "/api/mining/start",
            new StartMiningRequest(hardwareId, setup.Pool.Id));

        response.StatusCode.Should().Be(HttpStatusCode.OK);

        var session = await response.Content.ReadFromJsonAsync<MiningSessionResponse>();
        // Status reflects share activity: a brand-new session has produced no shares, so it is idle.
        session!.Status.Should().Be("idle");
        session.PoolId.Should().Be(setup.Pool.Id);
        session.CoinCode.Should().Be(setup.Coin.Code);
        session.AlgorithmCode.Should().Be(setup.Algo.Code);

        // The worker identity is the device's hardware id in "N" form — 32 chars, the pool's cap.
        session.WorkerId.Should().Be(hardwareId.ToString("N"));

        // The client is told the proxy endpoint explicitly; it must never use a pool endpoint.
        session.StratumEndpoint.Should().Be("localhost:3333");

        // The client reads the structured config directly instead of parsing a command line.
        session.MinerConfig.GetProperty("algo").GetString().Should().Be(setup.Algo.Code);
        session.MinerConfig.GetProperty("endpoint").GetString().Should().Be("localhost:3333");
        session.MinerConfig.GetProperty("wallet").GetString().Should().Be("wallet-123");
        session.MinerConfig.GetProperty("workerId").GetString().Should().Be(session.WorkerId);
        session.MinerConfig.GetProperty("coin").GetString().Should().Be(setup.Coin.Code);

        // The legacy command field is empty for a structured configuration.
        session.MinerCommand.Should().BeEmpty();

        var stored = await ReadSessionAsync(session.SessionId);
        stored.StoppedAt.Should().BeNull();
        stored.WorkerIdentifier.Should().Be(session.WorkerId);
        (await CountLogsAsync(session.SessionId, MiningEventType.MiningStarted)).Should().Be(1);
    }

    [Fact]
    public async Task Start_WhileASessionIsActive_ReturnsConflict()
    {
        var client = await factory.CreateUserClientAsync();
        var admin = await factory.CreateAdminClientAsync();
        var setup = await MiningTestData.SeedAsync(admin);
        var hardwareId = Guid.NewGuid();

        var first = await client.PostAsJsonAsync("/api/mining/start", new StartMiningRequest(hardwareId, setup.Pool.Id));
        first.StatusCode.Should().Be(HttpStatusCode.OK);

        var second = await client.PostAsJsonAsync("/api/mining/start", new StartMiningRequest(hardwareId, setup.Pool.Id));

        second.StatusCode.Should().Be(HttpStatusCode.Conflict);
    }

    [Fact]
    public async Task Start_WithoutAnAlgorithmForTheCoin_ReturnsConflict()
    {
        var client = await factory.CreateUserClientAsync();
        var admin = await factory.CreateAdminClientAsync();

        var coinResponse = await admin.PostAsJsonAsync(
            "/api/admin/coins",
            new CreateCoinRequest($"coin-{MiningTestData.Suffix()}", "Unsupported Coin", "testnet", 8));
        var coin = (await coinResponse.Content.ReadFromJsonAsync<CoinResponse>())!;

        var poolResponse = await admin.PostAsJsonAsync(
            "/api/admin/pools",
            new CreatePoolRequest(
                $"pool-{MiningTestData.Suffix()}",
                "Unsupported Pool",
                new PoolDetailsRequest("https://pool.example.com", null, null, coin.Id, "native_auto_payout", "wallet", "bep20"),
                [coin.Id]));
        var pool = (await poolResponse.Content.ReadFromJsonAsync<PoolResponse>())!;

        var response = await client.PostAsJsonAsync(
            "/api/mining/start",
            new StartMiningRequest(Guid.NewGuid(), pool.Id));

        response.StatusCode.Should().Be(HttpStatusCode.Conflict);
    }

    [Fact]
    public async Task Start_RequiresAuthentication()
    {
        var anonymous = factory.CreateClient();

        var response = await anonymous.PostAsJsonAsync(
            "/api/mining/start",
            new StartMiningRequest(Guid.NewGuid(), null));

        response.StatusCode.Should().Be(HttpStatusCode.Unauthorized);
    }

    // --- Current session -----------------------------------------------------------------

    [Fact]
    public async Task GetSession_WithNoActiveSession_ReturnsNoContent()
    {
        var client = await factory.CreateUserClientAsync();

        var response = await client.GetAsync($"/api/mining/session?hardwareId={Guid.NewGuid()}");

        response.StatusCode.Should().Be(HttpStatusCode.NoContent);
    }

    [Fact]
    public async Task GetSession_AfterStart_ReturnsTheSameSession()
    {
        var client = await factory.CreateUserClientAsync();
        var admin = await factory.CreateAdminClientAsync();
        var setup = await MiningTestData.SeedAsync(admin);
        var hardwareId = Guid.NewGuid();

        var start = await client.PostAsJsonAsync("/api/mining/start", new StartMiningRequest(hardwareId, setup.Pool.Id));
        var started = (await start.Content.ReadFromJsonAsync<MiningSessionResponse>())!;

        var response = await client.GetAsync($"/api/mining/session?hardwareId={hardwareId}");
        response.StatusCode.Should().Be(HttpStatusCode.OK);

        var current = (await response.Content.ReadFromJsonAsync<MiningSessionResponse>())!;
        current.SessionId.Should().Be(started.SessionId);
        current.WorkerId.Should().Be(hardwareId.ToString("N"));
        current.StratumEndpoint.Should().Be("localhost:3333");
    }

    [Fact]
    public async Task GetSession_ForATeardownSession_ReturnsNoContent()
    {
        var client = await factory.CreateUserClientAsync();
        var admin = await factory.CreateAdminClientAsync();
        var setup = await MiningTestData.SeedAsync(admin);
        var hardwareId = Guid.NewGuid();

        (await client.PostAsJsonAsync("/api/mining/start", new StartMiningRequest(hardwareId, setup.Pool.Id)))
            .StatusCode.Should().Be(HttpStatusCode.OK);
        (await client.PostAsJsonAsync("/api/mining/stop", new StopMiningRequest(hardwareId, "user-triggered")))
            .StatusCode.Should().Be(HttpStatusCode.NoContent);

        var response = await client.GetAsync($"/api/mining/session?hardwareId={hardwareId}");

        response.StatusCode.Should().Be(HttpStatusCode.NoContent);
    }

    [Fact]
    public async Task GetSession_RequiresAuthentication()
    {
        var anonymous = factory.CreateClient();

        var response = await anonymous.GetAsync($"/api/mining/session?hardwareId={Guid.NewGuid()}");

        response.StatusCode.Should().Be(HttpStatusCode.Unauthorized);
    }

    // --- Share-driven status -------------------------------------------------------------

    [Fact]
    public async Task Session_IsIdleUntilAShareArrives_ThenRunning()
    {
        var client = await factory.CreateUserClientAsync();
        var admin = await factory.CreateAdminClientAsync();
        var setup = await MiningTestData.SeedAsync(admin);
        var hardwareId = Guid.NewGuid();

        await client.PostAsJsonAsync("/api/mining/start", new StartMiningRequest(hardwareId, setup.Pool.Id));

        // No shares yet: the session is idle.
        var before = (await client.GetFromJsonAsync<MiningSessionResponse>(
            $"/api/mining/session?hardwareId={hardwareId}"))!;
        before.Status.Should().Be("idle");

        await RecordShareAsync(hardwareId, setup.Pool.Id, setup.Coin.Id);

        // An accepted share inside the window makes it running.
        var after = (await client.GetFromJsonAsync<MiningSessionResponse>(
            $"/api/mining/session?hardwareId={hardwareId}"))!;
        after.Status.Should().Be("running");
    }

    [Fact]
    public async Task AdminUserDetail_TracksShareActivity()
    {
        var admin = await factory.CreateAdminClientAsync();
        var userClient = await factory.CreateUserClientAsync();
        var setup = await MiningTestData.SeedAsync(admin);
        var hardwareId = Guid.NewGuid();

        var profile = (await userClient.GetFromJsonAsync<UserProfileResponse>("/api/auth/me"))!;

        await userClient.PostAsJsonAsync("/api/mining/start", new StartMiningRequest(hardwareId, setup.Pool.Id));

        var idle = (await admin.GetFromJsonAsync<AdminUserDetailResponse>($"/api/admin/users/{profile.Id}"))!;
        idle.Hardware.Single().ActiveSession!.Status.Should().Be("idle");

        await RecordShareAsync(hardwareId, setup.Pool.Id, setup.Coin.Id);

        var running = (await admin.GetFromJsonAsync<AdminUserDetailResponse>($"/api/admin/users/{profile.Id}"))!;
        running.Hardware.Single().ActiveSession!.Status.Should().Be("running");
    }

    // --- Stop ----------------------------------------------------------------------------

    [Fact]
    public async Task Stop_IsIdempotentAndLogsASingleStopEvent()
    {
        var client = await factory.CreateUserClientAsync();
        var admin = await factory.CreateAdminClientAsync();
        var setup = await MiningTestData.SeedAsync(admin);
        var hardwareId = Guid.NewGuid();

        var start = await client.PostAsJsonAsync("/api/mining/start", new StartMiningRequest(hardwareId, setup.Pool.Id));
        var session = (await start.Content.ReadFromJsonAsync<MiningSessionResponse>())!;

        var first = await client.PostAsJsonAsync(
            "/api/mining/stop",
            new StopMiningRequest(hardwareId, "user-triggered"));
        first.StatusCode.Should().Be(HttpStatusCode.NoContent);

        var second = await client.PostAsJsonAsync(
            "/api/mining/stop",
            new StopMiningRequest(hardwareId, "user-triggered"));
        second.StatusCode.Should().Be(HttpStatusCode.NoContent);

        var stored = await ReadSessionAsync(session.SessionId);
        stored.StoppedAt.Should().NotBeNull();
        stored.StopReason.Should().Be("user-triggered");
        (await CountLogsAsync(session.SessionId, MiningEventType.MiningStopped)).Should().Be(1);
    }

    [Fact]
    public async Task Stop_WithAnUnknownReason_ReturnsBadRequest()
    {
        var client = await factory.CreateUserClientAsync();

        var response = await client.PostAsJsonAsync(
            "/api/mining/stop",
            new StopMiningRequest(Guid.NewGuid(), "because"));

        response.StatusCode.Should().Be(HttpStatusCode.BadRequest);
    }

    [Fact]
    public async Task Start_OnADeviceOwnedByAnotherAccount_IsRefused()
    {
        var owner = await factory.CreateUserClientAsync();
        var intruder = await factory.CreateUserClientAsync();
        var admin = await factory.CreateAdminClientAsync();
        var setup = await MiningTestData.SeedAsync(admin);
        var hardwareId = Guid.NewGuid();

        (await owner.PostAsJsonAsync("/api/mining/start", new StartMiningRequest(hardwareId, setup.Pool.Id)))
            .StatusCode.Should().Be(HttpStatusCode.OK);

        // The hardware id is the device's identity, so another account cannot take it over.
        var response = await intruder.PostAsJsonAsync(
            "/api/mining/start",
            new StartMiningRequest(hardwareId, setup.Pool.Id));

        response.StatusCode.Should().Be(HttpStatusCode.Conflict);
    }
}
