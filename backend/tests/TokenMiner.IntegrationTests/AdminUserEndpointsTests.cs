using System.Net;
using System.Net.Http.Json;
using FluentAssertions;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.DependencyInjection;
using TokenMiner.Contracts.Admin;
using TokenMiner.Contracts.Auth;
using TokenMiner.Contracts.Mining;
using TokenMiner.Domain.Mining;
using TokenMiner.Domain.Mining.Enums;
using TokenMiner.Infrastructure.Persistence;
using Xunit;

namespace TokenMiner.IntegrationTests;

[Collection(ApiCollection.Name)]
public sealed class AdminUserEndpointsTests(ApiFactory factory)
{
    private static UserMiningShare AddShare(
        AppDbContext dbContext,
        Guid userId,
        Guid hardwareId,
        Guid sessionId,
        Guid poolId,
        Guid coinId,
        string identifier,
        ShareRewardStatus status)
    {
        var now = DateTimeOffset.UtcNow;

        var share = new UserMiningShare(
            Guid.NewGuid(),
            userId,
            hardwareId,
            sessionId,
            poolId,
            coinId,
            identifier,
            $"worker-{identifier}",
            "job-1",
            "nonce-1",
            null,
            1.5m,
            "target",
            "hash",
            now,
            1m,
            1m,
            null,
            now);

        if (status == ShareRewardStatus.Accepted)
        {
            share.Accept(now);
        }
        else if (status == ShareRewardStatus.Rejected)
        {
            share.Reject("test rejection", now);
        }

        dbContext.UserMiningShares.Add(share);

        return share;
    }

    // --- Authorisation -------------------------------------------------------------------

    [Fact]
    public async Task AdminUserEndpoints_RequireAuthentication()
    {
        var anonymous = factory.CreateClient();

        var response = await anonymous.GetAsync("/api/admin/users");

        response.StatusCode.Should().Be(HttpStatusCode.Unauthorized);
    }

    [Fact]
    public async Task AdminUserEndpoints_RejectNonAdminUsers()
    {
        var user = await factory.CreateUserClientAsync();

        var response = await user.GetAsync("/api/admin/users");

        response.StatusCode.Should().Be(HttpStatusCode.Forbidden);
    }

    // --- Listing and detail --------------------------------------------------------------

    [Fact]
    public async Task Users_ListAndDetailExposeDevicesAndTerminalShareCounts()
    {
        var admin = await factory.CreateAdminClientAsync();
        var userClient = await factory.CreateUserClientAsync();
        var setup = await MiningTestData.SeedAsync(admin);

        var profile = (await userClient.GetFromJsonAsync<UserProfileResponse>("/api/auth/me"))!;
        var hardwareId = Guid.NewGuid();

        var start = await userClient.PostAsJsonAsync(
            "/api/mining/start",
            new StartMiningRequest(hardwareId, setup.Pool.Id));
        start.EnsureSuccessStatusCode();

        var session = (await start.Content.ReadFromJsonAsync<MiningSessionResponse>())!;

        await using (var scope = factory.Services.CreateAsyncScope())
        {
            var dbContext = scope.ServiceProvider.GetRequiredService<AppDbContext>();

            var hardwareRowId = await dbContext.UserHardware
                .Where(hardware => hardware.UserId == profile.Id && hardware.HardwareId == hardwareId)
                .Select(hardware => hardware.Id)
                .SingleAsync();

            AddShare(dbContext, profile.Id, hardwareRowId, session.SessionId, setup.Pool.Id, setup.Coin.Id, "s-1", ShareRewardStatus.Accepted);
            AddShare(dbContext, profile.Id, hardwareRowId, session.SessionId, setup.Pool.Id, setup.Coin.Id, "s-2", ShareRewardStatus.Accepted).Redeem(DateTimeOffset.UtcNow);
            AddShare(dbContext, profile.Id, hardwareRowId, session.SessionId, setup.Pool.Id, setup.Coin.Id, "s-3", ShareRewardStatus.Rejected);

            // An in-flight share must not be counted under either terminal state.
            AddShare(dbContext, profile.Id, hardwareRowId, session.SessionId, setup.Pool.Id, setup.Coin.Id, "s-4", ShareRewardStatus.Pending);

            await dbContext.SaveChangesAsync();
        }

        var listed = await admin.GetFromJsonAsync<List<AdminUserSummaryResponse>>("/api/admin/users?limit=500");
        var row = listed!.Single(user => user.Id == profile.Id);

        row.HardwareCount.Should().Be(1);
        row.AcceptedShares.Should().Be(2);
        row.RejectedShares.Should().Be(1);

        var detail = await admin.GetFromJsonAsync<AdminUserDetailResponse>($"/api/admin/users/{profile.Id}");

        detail!.Email.Should().Be(profile.Email);
        detail.HardwareCount.Should().Be(1);
        detail.AcceptedShares.Should().Be(2);
        detail.RejectedShares.Should().Be(1);
        detail.Hardware.Should().HaveCount(1);

        var device = detail.Hardware[0];
        device.HardwareId.Should().Be(hardwareId);
        device.AcceptedShares.Should().Be(2);
        device.RejectedShares.Should().Be(1);
        device.ActiveSession.Should().NotBeNull();
        device.ActiveSession!.CoinCode.Should().Be(setup.Coin.Code);
        device.ActiveSession.PoolName.Should().Be(setup.Pool.Name);
        device.ActiveSession.Status.Should().Be("running");
    }

    [Fact]
    public async Task Users_ListWithNoLimitStillReturnsTheAccount()
    {
        var admin = await factory.CreateAdminClientAsync();
        var userClient = await factory.CreateUserClientAsync();

        var profile = (await userClient.GetFromJsonAsync<UserProfileResponse>("/api/auth/me"))!;

        var listed = await admin.GetFromJsonAsync<List<AdminUserSummaryResponse>>("/api/admin/users");

        listed!.Should().Contain(user => user.Id == profile.Id);
    }

    [Fact]
    public async Task User_DetailIsNotFoundForAnUnknownAccount()
    {
        var admin = await factory.CreateAdminClientAsync();

        var response = await admin.GetAsync($"/api/admin/users/{Guid.NewGuid()}");

        response.StatusCode.Should().Be(HttpStatusCode.NotFound);
    }
}
