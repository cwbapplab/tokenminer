using System.Globalization;
using System.Net;
using System.Net.Http.Headers;
using System.Net.Http.Json;
using System.Text;
using System.Text.Json;
using FluentAssertions;
using MediatR;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.DependencyInjection;
using TokenMiner.Application.Mining.Services;
using TokenMiner.Application.Mining.Shares;
using TokenMiner.Application.Mining.Statistics;
using TokenMiner.Contracts.Auth;
using TokenMiner.Contracts.Mining;
using TokenMiner.Infrastructure.Persistence;
using Xunit;

namespace TokenMiner.IntegrationTests;

[Collection(ApiCollection.Name)]
public sealed class MiningShareEndpointsTests(ApiFactory factory)
{
    private static readonly JsonSerializerOptions WebJson = new(JsonSerializerDefaults.Web);

    private const string SharesPath = "/internal/mining/shares";

    // --- Helpers -------------------------------------------------------------------------

    private async Task WithScopeAsync(Func<AppDbContext, Task> action)
    {
        await using var scope = factory.Services.CreateAsyncScope();
        await action(scope.ServiceProvider.GetRequiredService<AppDbContext>());
    }

    private async Task<Guid> ReadHardwareRowIdAsync(Guid userId, Guid hardwareId)
    {
        await using var scope = factory.Services.CreateAsyncScope();
        var dbContext = scope.ServiceProvider.GetRequiredService<AppDbContext>();

        return await dbContext.UserHardware
            .Where(hardware => hardware.UserId == userId && hardware.HardwareId == hardwareId)
            .Select(hardware => hardware.Id)
            .SingleAsync();
    }

    private async Task SetCoinPriceAsync(Guid coinId, decimal price)
    {
        await WithScopeAsync(async dbContext =>
        {
            var coin = await dbContext.Coins.SingleAsync(candidate => candidate.Id == coinId);
            coin.RecordPrice(price, DateTimeOffset.UtcNow, DateTimeOffset.UtcNow);
            await dbContext.SaveChangesAsync();
        });
    }

    private async Task<int> CountSharesAsync(string shareIdentifier)
    {
        await using var scope = factory.Services.CreateAsyncScope();
        var dbContext = scope.ServiceProvider.GetRequiredService<AppDbContext>();

        return await dbContext.UserMiningShares.CountAsync(share => share.ShareIdentifier == shareIdentifier);
    }

    private static object SharePayload(
        Guid userId,
        Guid hardwareRowId,
        Guid poolId,
        Guid coinId,
        string shareIdentifier,
        decimal? coinValue = null) => new
    {
        userId,
        userHardwareId = hardwareRowId,
        poolId,
        coinId,
        shareIdentifier,
        jobId = "job-1",
        nonce = "nonce-1",
        extranonce = (string?)null,
        difficulty = 1.5m,
        target = "target",
        hash = "hash",
        timestamp = DateTimeOffset.UtcNow,
        coinValue,
        poolResponse = new { accepted = true },
    };

    private static async Task<HttpResponseMessage> PostShareAsync(
        HttpClient client,
        object payload,
        string? nonce = null,
        long? timestamp = null,
        string? secret = null)
    {
        var body = Encoding.UTF8.GetBytes(JsonSerializer.Serialize(payload, WebJson));
        var bodyHash = ServiceRequestSignature.ComputeBodyHash(body);

        var timestampValue = (timestamp ?? DateTimeOffset.UtcNow.ToUnixTimeSeconds())
            .ToString(CultureInfo.InvariantCulture);
        var nonceValue = nonce ?? Guid.NewGuid().ToString("N");

        var signature = ServiceRequestSignature.Compute(
            secret ?? MiningTestData.ServiceSharedSecret,
            "POST",
            SharesPath,
            timestampValue,
            nonceValue,
            bodyHash);

        var content = new ByteArrayContent(body);
        content.Headers.ContentType = new MediaTypeHeaderValue("application/json");

        using var request = new HttpRequestMessage(HttpMethod.Post, SharesPath) { Content = content };
        request.Headers.Add("X-Service-Id", "stratum-proxy");
        request.Headers.Add("X-Timestamp", timestampValue);
        request.Headers.Add("X-Nonce", nonceValue);
        request.Headers.Add("X-Signature", signature);

        return await client.SendAsync(request);
    }

    private sealed record Session(
        HttpClient UserClient,
        HttpClient ServiceClient,
        MiningTestData.Setup Setup,
        Guid UserId,
        Guid HardwareRowId);

    private async Task<Session> StartSessionAsync()
    {
        var userClient = await factory.CreateUserClientAsync();
        var admin = await factory.CreateAdminClientAsync();
        var setup = await MiningTestData.SeedAsync(admin);
        var hardwareId = Guid.NewGuid();

        var start = await userClient.PostAsJsonAsync(
            "/api/mining/start",
            new StartMiningRequest(hardwareId, setup.Pool.Id));
        start.EnsureSuccessStatusCode();

        var profile = await userClient.GetFromJsonAsync<UserProfileResponse>("/api/auth/me");
        var hardwareRowId = await ReadHardwareRowIdAsync(profile!.Id, hardwareId);

        return new Session(userClient, factory.CreateClient(), setup, profile.Id, hardwareRowId);
    }

    private async Task ProcessAndRollUpAsync()
    {
        await using var scope = factory.Services.CreateAsyncScope();
        var sender = scope.ServiceProvider.GetRequiredService<ISender>();

        await sender.Send(new ProcessPendingSharesCommand(500));
        await sender.Send(new RecalculateMiningStatisticsCommand());
    }

    // --- Service authentication ----------------------------------------------------------

    [Fact]
    public async Task Share_WithoutSignatureHeaders_IsRejected()
    {
        var client = factory.CreateClient();

        var response = await client.PostAsJsonAsync(SharesPath, new { });

        response.StatusCode.Should().Be(HttpStatusCode.Unauthorized);
    }

    [Fact]
    public async Task Share_WithAForgedSignature_IsRejected()
    {
        var session = await StartSessionAsync();

        var response = await PostShareAsync(
            session.ServiceClient,
            SharePayload(session.UserId, session.HardwareRowId, session.Setup.Pool.Id, session.Setup.Coin.Id, "forged-1"),
            secret: "not-the-real-secret");

        response.StatusCode.Should().Be(HttpStatusCode.Unauthorized);
    }

    [Fact]
    public async Task Share_WithAStaleTimestamp_IsRejected()
    {
        var session = await StartSessionAsync();

        var response = await PostShareAsync(
            session.ServiceClient,
            SharePayload(session.UserId, session.HardwareRowId, session.Setup.Pool.Id, session.Setup.Coin.Id, "stale-1"),
            timestamp: DateTimeOffset.UtcNow.AddHours(-1).ToUnixTimeSeconds());

        response.StatusCode.Should().Be(HttpStatusCode.Unauthorized);
    }

    [Fact]
    public async Task Share_WithAReplayedNonce_IsRejected()
    {
        var session = await StartSessionAsync();
        var nonce = Guid.NewGuid().ToString("N");

        var first = await PostShareAsync(
            session.ServiceClient,
            SharePayload(session.UserId, session.HardwareRowId, session.Setup.Pool.Id, session.Setup.Coin.Id, "replay-1"),
            nonce);

        first.StatusCode.Should().Be(HttpStatusCode.OK);

        // A different share, but signed with a nonce already spent.
        var second = await PostShareAsync(
            session.ServiceClient,
            SharePayload(session.UserId, session.HardwareRowId, session.Setup.Pool.Id, session.Setup.Coin.Id, "replay-2"),
            nonce);

        second.StatusCode.Should().Be(HttpStatusCode.Unauthorized);
    }

    [Fact]
    public async Task Share_IsRecordedAsPendingThenDeduplicated()
    {
        var session = await StartSessionAsync();
        var shareIdentifier = $"share-{Guid.NewGuid():N}";

        var payload = SharePayload(
            session.UserId,
            session.HardwareRowId,
            session.Setup.Pool.Id,
            session.Setup.Coin.Id,
            shareIdentifier);

        var first = await PostShareAsync(session.ServiceClient, payload);
        first.StatusCode.Should().Be(HttpStatusCode.OK);

        var created = await first.Content.ReadFromJsonAsync<MiningShareResponse>();
        created!.AlreadyRecorded.Should().BeFalse();
        created.RewardStatus.Should().Be("pending");
        created.ShareIdentifier.Should().Be(shareIdentifier);

        // Replaying the same pool-side share is an idempotent success, not a duplicate row.
        var second = await PostShareAsync(session.ServiceClient, payload);
        second.StatusCode.Should().Be(HttpStatusCode.OK);

        var replayed = await second.Content.ReadFromJsonAsync<MiningShareResponse>();
        replayed!.AlreadyRecorded.Should().BeTrue();
        replayed.Id.Should().Be(created.Id);

        (await CountSharesAsync(shareIdentifier)).Should().Be(1);
    }

    [Fact]
    public async Task Share_ForAnUnknownDevice_IsRejected()
    {
        var session = await StartSessionAsync();

        var response = await PostShareAsync(
            session.ServiceClient,
            SharePayload(
                session.UserId,
                Guid.NewGuid(),
                session.Setup.Pool.Id,
                session.Setup.Coin.Id,
                $"unknown-{Guid.NewGuid():N}"));

        response.StatusCode.Should().Be(HttpStatusCode.BadRequest);
    }

    [Fact]
    public async Task Share_WhenThePoolDoesNotSupportTheCoin_IsRejected()
    {
        var session = await StartSessionAsync();
        var admin = await factory.CreateAdminClientAsync();

        var otherCoinResponse = await admin.PostAsJsonAsync(
            "/api/admin/coins",
            new CreateCoinRequest($"coin-{MiningTestData.Suffix()}", "Unrelated Coin", "testnet", 8));
        var otherCoin = (await otherCoinResponse.Content.ReadFromJsonAsync<CoinResponse>())!;

        var response = await PostShareAsync(
            session.ServiceClient,
            SharePayload(
                session.UserId,
                session.HardwareRowId,
                session.Setup.Pool.Id,
                otherCoin.Id,
                $"unrelated-{Guid.NewGuid():N}"));

        response.StatusCode.Should().Be(HttpStatusCode.BadRequest);
    }

    // --- Analytics -----------------------------------------------------------------------

    [Fact]
    public async Task Analytics_RequiresAuthentication()
    {
        var anonymous = factory.CreateClient();

        var response = await anonymous.GetAsync("/api/analytics");

        response.StatusCode.Should().Be(HttpStatusCode.Unauthorized);
    }

    [Fact]
    public async Task Analytics_StartsEmptyForANewAccount()
    {
        var session = await StartSessionAsync();

        var analytics = await session.UserClient.GetFromJsonAsync<MiningAnalyticsResponse>("/api/analytics");

        analytics!.Hardware.Should().BeEmpty();
        analytics.Totals.AllTimeUsd.Should().Be(0m);
    }

    [Fact]
    public async Task Analytics_ReflectsProcessedShares()
    {
        var session = await StartSessionAsync();

        // The share snapshot uses the coin's price at acceptance time.
        await SetCoinPriceAsync(session.Setup.Coin.Id, 2.5m);

        (await PostShareAsync(session.ServiceClient, SharePayload(
            session.UserId, session.HardwareRowId, session.Setup.Pool.Id, session.Setup.Coin.Id,
            $"a-{Guid.NewGuid():N}", coinValue: 4m))).StatusCode.Should().Be(HttpStatusCode.OK);

        (await PostShareAsync(session.ServiceClient, SharePayload(
            session.UserId, session.HardwareRowId, session.Setup.Pool.Id, session.Setup.Coin.Id,
            $"b-{Guid.NewGuid():N}", coinValue: 2m))).StatusCode.Should().Be(HttpStatusCode.OK);

        await ProcessAndRollUpAsync();

        var analytics = await session.UserClient.GetFromJsonAsync<MiningAnalyticsResponse>("/api/analytics");

        // 4 * 2.5 + 2 * 2.5 = 15
        analytics!.Totals.AllTimeUsd.Should().Be(15m);
        analytics.Totals.Last30Usd.Should().Be(15m);
        analytics.Totals.Last60Usd.Should().Be(15m);

        analytics.Hardware.Should().HaveCount(1);
        analytics.Hardware[0].UserHardwareId.Should().Be(session.HardwareRowId);
        analytics.Hardware[0].AllTimeUsd.Should().Be(15m);
    }

    [Fact]
    public async Task Analytics_CanBeFilteredToASingleDevice()
    {
        var userClient = await factory.CreateUserClientAsync();
        var admin = await factory.CreateAdminClientAsync();
        var setup = await MiningTestData.SeedAsync(admin);
        var serviceClient = factory.CreateClient();

        await SetCoinPriceAsync(setup.Coin.Id, 1m);

        var profile = await userClient.GetFromJsonAsync<UserProfileResponse>("/api/auth/me");

        var hardwareRowIds = new List<Guid>();

        foreach (var amount in new[] { 3m, 7m })
        {
            var hardwareId = Guid.NewGuid();

            var start = await userClient.PostAsJsonAsync(
                "/api/mining/start",
                new StartMiningRequest(hardwareId, setup.Pool.Id));
            start.EnsureSuccessStatusCode();

            var hardwareRowId = await ReadHardwareRowIdAsync(profile!.Id, hardwareId);
            hardwareRowIds.Add(hardwareRowId);

            var response = await PostShareAsync(serviceClient, SharePayload(
                profile.Id, hardwareRowId, setup.Pool.Id, setup.Coin.Id,
                $"h-{Guid.NewGuid():N}", coinValue: amount));

            response.StatusCode.Should().Be(HttpStatusCode.OK);
        }

        await ProcessAndRollUpAsync();

        var all = await userClient.GetFromJsonAsync<MiningAnalyticsResponse>("/api/analytics");
        all!.Totals.AllTimeUsd.Should().Be(10m);
        all.Hardware.Should().HaveCount(2);

        var filtered = await userClient.GetFromJsonAsync<MiningAnalyticsResponse>(
            $"/api/analytics?userHardwareId={hardwareRowIds[0]}");

        filtered!.Totals.AllTimeUsd.Should().Be(3m);
        filtered.Hardware.Should().HaveCount(1);
        filtered.Hardware[0].UserHardwareId.Should().Be(hardwareRowIds[0]);
    }

    [Fact]
    public async Task Analytics_NeverExposesAnotherUsersDevice()
    {
        var owner = await StartSessionAsync();
        await SetCoinPriceAsync(owner.Setup.Coin.Id, 1m);

        (await PostShareAsync(owner.ServiceClient, SharePayload(
            owner.UserId, owner.HardwareRowId, owner.Setup.Pool.Id, owner.Setup.Coin.Id,
            $"o-{Guid.NewGuid():N}", coinValue: 5m))).StatusCode.Should().Be(HttpStatusCode.OK);

        await ProcessAndRollUpAsync();

        // A different account asking for the first user's device sees nothing.
        var intruder = await factory.CreateUserClientAsync();
        var analytics = await intruder.GetFromJsonAsync<MiningAnalyticsResponse>(
            $"/api/analytics?userHardwareId={owner.HardwareRowId}");

        analytics!.Totals.AllTimeUsd.Should().Be(0m);
        analytics.Hardware.Should().BeEmpty();
    }
}
