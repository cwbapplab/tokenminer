using System.Net;
using System.Net.Http.Json;
using System.Net.WebSockets;
using System.Text.Json;
using FluentAssertions;
using MediatR;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.DependencyInjection;
using TokenMiner.Application.Mining.Sessions;
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
    private static readonly JsonSerializerOptions WebJson = new(JsonSerializerDefaults.Web);

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

    private async Task<MiningHeartbeatAck> SendHeartbeatAsync(string accessToken, Guid hardwareId)
    {
        using var socket = await factory.Server
            .CreateWebSocketClient()
            .ConnectAsync(new Uri($"ws://localhost/ws/mining?access_token={accessToken}"), CancellationToken.None);

        var payload = JsonSerializer.SerializeToUtf8Bytes(
            new MiningHeartbeatMessage("heartbeat", hardwareId),
            WebJson);

        await socket.SendAsync(payload, WebSocketMessageType.Text, endOfMessage: true, CancellationToken.None);

        var buffer = new byte[4096];
        var received = await socket.ReceiveAsync(buffer, CancellationToken.None);

        return JsonSerializer.Deserialize<MiningHeartbeatAck>(buffer.AsSpan(0, received.Count), WebJson)!;
    }

    // --- Start ---------------------------------------------------------------------------

    [Fact]
    public async Task Start_CreatesASessionAndReturnsTheLaunchCommand()
    {
        var client = await factory.CreateUserClientAsync();
        var admin = await factory.CreateAdminClientAsync();
        var setup = await MiningTestData.SeedAsync(admin);
        var hardwareId = Guid.NewGuid();

        var profile = await client.GetFromJsonAsync<UserProfileResponse>("/api/auth/me");

        var response = await client.PostAsJsonAsync(
            "/api/mining/start",
            new StartMiningRequest(hardwareId, setup.Pool.Id));

        response.StatusCode.Should().Be(HttpStatusCode.OK);

        var session = await response.Content.ReadFromJsonAsync<MiningSessionResponse>();
        session!.Status.Should().Be("running");
        session.PoolId.Should().Be(setup.Pool.Id);
        session.CoinCode.Should().Be(setup.Coin.Code);
        session.AlgorithmCode.Should().Be(setup.Algo.Code);

        // Worker identity is derived server-side from the authenticated user and the device.
        session.WorkerId.Should().StartWith($"{profile!.Id}-");

        session.MinerCommand.Should().Contain($"--algo {setup.Algo.Code}");
        session.MinerCommand.Should().Contain("--stratum pool.example.com");
        session.MinerCommand.Should().Contain("--wallet wallet-123");
        session.MinerCommand.Should().EndWith(session.WorkerId);

        var stored = await ReadSessionAsync(session.SessionId);
        stored.Status.Should().Be(MiningSessionStatus.Running);
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
        stored.Status.Should().Be(MiningSessionStatus.Stopped);
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

    // --- Liveness ------------------------------------------------------------------------

    [Fact]
    public async Task Heartbeat_OnAnActiveSession_IsAcknowledged()
    {
        var (client, accessToken) = await factory.CreateUserClientWithTokenAsync();
        var admin = await factory.CreateAdminClientAsync();
        var setup = await MiningTestData.SeedAsync(admin);
        var hardwareId = Guid.NewGuid();

        var start = await client.PostAsJsonAsync("/api/mining/start", new StartMiningRequest(hardwareId, setup.Pool.Id));
        var session = (await start.Content.ReadFromJsonAsync<MiningSessionResponse>())!;

        var ack = await SendHeartbeatAsync(accessToken, hardwareId);

        ack.SessionFound.Should().BeTrue();
        ack.Status.Should().Be("running");
        ack.Resumed.Should().BeFalse();

        var stored = await ReadSessionAsync(session.SessionId);
        stored.Status.Should().Be(MiningSessionStatus.Running);
    }

    [Fact]
    public async Task Heartbeat_WithoutASession_IsReportedAsNotFound()
    {
        var (_, accessToken) = await factory.CreateUserClientWithTokenAsync();

        var ack = await SendHeartbeatAsync(accessToken, Guid.NewGuid());

        ack.SessionFound.Should().BeFalse();
        ack.Status.Should().BeNull();
    }

    [Fact]
    public async Task IdleSession_IsPausedOnceAndResumesOnTheNextHeartbeat()
    {
        var (client, accessToken) = await factory.CreateUserClientWithTokenAsync();
        var admin = await factory.CreateAdminClientAsync();
        var setup = await MiningTestData.SeedAsync(admin);
        var hardwareId = Guid.NewGuid();

        var start = await client.PostAsJsonAsync("/api/mining/start", new StartMiningRequest(hardwareId, setup.Pool.Id));
        var session = (await start.Content.ReadFromJsonAsync<MiningSessionResponse>())!;

        // Simulate the client going quiet, then run the watchdog once.
        await using (var scope = factory.Services.CreateAsyncScope())
        {
            var dbContext = scope.ServiceProvider.GetRequiredService<AppDbContext>();
            var row = await dbContext.UserHardwareMiners.SingleAsync(candidate => candidate.Id == session.SessionId);
            row.Touch(DateTimeOffset.UtcNow.AddMinutes(-5));
            await dbContext.SaveChangesAsync();
        }

        var pausedCount = await WithSenderAsync(sender => sender.Send(new PauseStaleSessionsCommand(10)));
        pausedCount.Should().Be(1);

        var paused = await ReadSessionAsync(session.SessionId);
        paused.Status.Should().Be(MiningSessionStatus.Paused);
        paused.PauseReason.Should().Be("connection-failure");
        (await CountLogsAsync(session.SessionId, MiningEventType.MiningPaused)).Should().Be(1);

        // A second pass changes nothing, so one outage produces one pause event.
        (await WithSenderAsync(sender => sender.Send(new PauseStaleSessionsCommand(10)))).Should().Be(0);
        (await CountLogsAsync(session.SessionId, MiningEventType.MiningPaused)).Should().Be(1);

        // The client reconnecting resumes the session.
        var ack = await SendHeartbeatAsync(accessToken, hardwareId);
        ack.SessionFound.Should().BeTrue();
        ack.Resumed.Should().BeTrue();
        ack.Status.Should().Be("running");

        var resumed = await ReadSessionAsync(session.SessionId);
        resumed.Status.Should().Be(MiningSessionStatus.Running);
        resumed.PauseReason.Should().BeNull();
        (await CountLogsAsync(session.SessionId, MiningEventType.MiningResumed)).Should().Be(1);
    }

    [Fact]
    public async Task UserCannotStartMiningForAnotherUsersDevice()
    {
        var owner = await factory.CreateUserClientAsync();
        var intruder = await factory.CreateUserClientAsync();
        var admin = await factory.CreateAdminClientAsync();
        var setup = await MiningTestData.SeedAsync(admin);
        var hardwareId = Guid.NewGuid();

        await owner.PostAsJsonAsync("/api/mining/start", new StartMiningRequest(hardwareId, setup.Pool.Id));

        // Same hardware GUID, different account: a separate device with its own session.
        var response = await intruder.PostAsJsonAsync(
            "/api/mining/start",
            new StartMiningRequest(hardwareId, setup.Pool.Id));

        response.StatusCode.Should().Be(HttpStatusCode.OK);

        var session = await response.Content.ReadFromJsonAsync<MiningSessionResponse>();
        var ownerProfile = await owner.GetFromJsonAsync<UserProfileResponse>("/api/auth/me");

        session!.WorkerId.Should().NotStartWith($"{ownerProfile!.Id}-");
    }
}
