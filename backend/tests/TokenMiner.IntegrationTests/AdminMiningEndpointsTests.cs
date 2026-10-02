using System.Net;
using System.Net.Http.Json;
using System.Text.Json;
using FluentAssertions;
using TokenMiner.Contracts.Mining;
using Xunit;

namespace TokenMiner.IntegrationTests;

[Collection(ApiCollection.Name)]
public sealed class AdminMiningEndpointsTests(ApiFactory factory)
{
    private static string UniqueSuffix() => Guid.NewGuid().ToString("N")[..8];

    private static async Task<CoinResponse> CreateCoinAsync(HttpClient admin)
    {
        var request = new CreateCoinRequest($"coin-{UniqueSuffix()}", "Test Coin", "testnet", 8);

        var response = await admin.PostAsJsonAsync("/api/admin/coins", request);
        response.StatusCode.Should().Be(HttpStatusCode.Created);

        return (await response.Content.ReadFromJsonAsync<CoinResponse>())!;
    }

    private static PoolDetailsRequest AutoPayoutDetails(Guid coinId) =>
        new("https://pool.example.com", "/api/v1/index", "pool.example.com:7049", coinId, "native_auto_payout", "wallet-123", "bep20");

    // --- Authorisation -------------------------------------------------------------------

    [Fact]
    public async Task AdminEndpoints_RequireAuthentication()
    {
        var anonymous = factory.CreateClient();

        var response = await anonymous.GetAsync("/api/admin/coins");

        response.StatusCode.Should().Be(HttpStatusCode.Unauthorized);
    }

    [Fact]
    public async Task AdminEndpoints_RejectNonAdminUsers()
    {
        var user = await factory.CreateUserClientAsync();

        var response = await user.GetAsync("/api/admin/coins");

        response.StatusCode.Should().Be(HttpStatusCode.Forbidden);
    }

    // --- Coins ---------------------------------------------------------------------------

    [Fact]
    public async Task Coins_CanBeCreatedReadListedAndUpdated()
    {
        var admin = await factory.CreateAdminClientAsync();

        var created = await CreateCoinAsync(admin);
        created.Code.Should().StartWith("coin-");
        created.Status.Should().Be("active");
        created.LastKnownUsdValue.Should().BeNull();

        var fetched = await admin.GetFromJsonAsync<CoinResponse>($"/api/admin/coins/{created.Id}");
        fetched!.Id.Should().Be(created.Id);

        var listed = await admin.GetFromJsonAsync<List<CoinResponse>>("/api/admin/coins");
        listed.Should().Contain(coin => coin.Id == created.Id);

        var update = await admin.PutAsJsonAsync(
            $"/api/admin/coins/{created.Id}",
            new UpdateCoinRequest("Renamed Coin", "othernet", 6, "disabled"));
        update.StatusCode.Should().Be(HttpStatusCode.OK);

        var updated = await update.Content.ReadFromJsonAsync<CoinResponse>();
        updated!.Name.Should().Be("Renamed Coin");
        updated.Network.Should().Be("othernet");
        updated.Decimals.Should().Be(6);
        updated.Status.Should().Be("disabled");
    }

    [Fact]
    public async Task Coins_RejectDuplicateCode()
    {
        var admin = await factory.CreateAdminClientAsync();
        var created = await CreateCoinAsync(admin);

        var duplicate = await admin.PostAsJsonAsync(
            "/api/admin/coins",
            new CreateCoinRequest(created.Code, "Another", "testnet", 8));

        duplicate.StatusCode.Should().Be(HttpStatusCode.Conflict);
    }

    [Fact]
    public async Task Coins_RejectUnknownStatus()
    {
        var admin = await factory.CreateAdminClientAsync();
        var created = await CreateCoinAsync(admin);

        var response = await admin.PutAsJsonAsync(
            $"/api/admin/coins/{created.Id}",
            new UpdateCoinRequest("Coin", "testnet", 8, "enabled"));

        response.StatusCode.Should().Be(HttpStatusCode.BadRequest);
    }

    // --- Pools ---------------------------------------------------------------------------

    [Fact]
    public async Task Pools_CanBeCreatedReadAndUpdated()
    {
        var admin = await factory.CreateAdminClientAsync();
        var firstCoin = await CreateCoinAsync(admin);
        var secondCoin = await CreateCoinAsync(admin);
        var systemPoolId = $"pool-{UniqueSuffix()}";

        var create = await admin.PostAsJsonAsync(
            "/api/admin/pools",
            new CreatePoolRequest(systemPoolId, "Kryptex QTC", AutoPayoutDetails(firstCoin.Id), [firstCoin.Id]));
        create.StatusCode.Should().Be(HttpStatusCode.Created);

        var pool = await create.Content.ReadFromJsonAsync<PoolResponse>();
        pool!.SystemPoolId.Should().Be(systemPoolId);
        pool.Status.Should().Be("active");
        pool.Details!.PayoutMode.Should().Be("native_auto_payout");
        pool.Details.PayoutAddress.Should().Be("wallet-123");
        pool.CoinIds.Should().BeEquivalentTo([firstCoin.Id]);

        var fetched = await admin.GetFromJsonAsync<PoolResponse>($"/api/admin/pools/{pool.Id}");
        fetched!.CoinIds.Should().BeEquivalentTo([firstCoin.Id]);

        var update = await admin.PutAsJsonAsync(
            $"/api/admin/pools/{pool.Id}",
            new UpdatePoolRequest("Kryptex QTC (retired)", "disabled", AutoPayoutDetails(secondCoin.Id), [secondCoin.Id]));
        update.StatusCode.Should().Be(HttpStatusCode.OK);

        var updated = await update.Content.ReadFromJsonAsync<PoolResponse>();
        updated!.Name.Should().Be("Kryptex QTC (retired)");
        updated.Status.Should().Be("disabled");
        updated.Details!.CoinId.Should().Be(secondCoin.Id);
        updated.CoinIds.Should().BeEquivalentTo([secondCoin.Id]);
    }

    [Fact]
    public async Task Pools_RejectUnknownCoin()
    {
        var admin = await factory.CreateAdminClientAsync();
        var unknownCoin = Guid.NewGuid();

        var response = await admin.PostAsJsonAsync(
            "/api/admin/pools",
            new CreatePoolRequest($"pool-{UniqueSuffix()}", "Bad Pool", AutoPayoutDetails(unknownCoin), [unknownCoin]));

        response.StatusCode.Should().Be(HttpStatusCode.BadRequest);
    }

    [Fact]
    public async Task Pools_RejectUnknownPayoutMode()
    {
        var admin = await factory.CreateAdminClientAsync();
        var coin = await CreateCoinAsync(admin);

        var details = new PoolDetailsRequest("https://pool.example.com", null, null, coin.Id, "auto_magic", null, null);

        var response = await admin.PostAsJsonAsync(
            "/api/admin/pools",
            new CreatePoolRequest($"pool-{UniqueSuffix()}", "Bad Pool", details, [coin.Id]));

        response.StatusCode.Should().Be(HttpStatusCode.BadRequest);
    }

    [Fact]
    public async Task Pools_RequireAPayoutAddressWhenNotManagedByUs()
    {
        var admin = await factory.CreateAdminClientAsync();
        var coin = await CreateCoinAsync(admin);

        var details = new PoolDetailsRequest("https://pool.example.com", null, null, coin.Id, "direct_to_wallet", null, null);

        var response = await admin.PostAsJsonAsync(
            "/api/admin/pools",
            new CreatePoolRequest($"pool-{UniqueSuffix()}", "No Wallet", details, [coin.Id]));

        response.StatusCode.Should().Be(HttpStatusCode.BadRequest);
    }

    [Fact]
    public async Task Pools_AllowManagedRequestWithoutAPayoutAddress()
    {
        var admin = await factory.CreateAdminClientAsync();
        var coin = await CreateCoinAsync(admin);

        var details = new PoolDetailsRequest("https://pool.example.com", null, null, coin.Id, "managed_request", null, null);

        var response = await admin.PostAsJsonAsync(
            "/api/admin/pools",
            new CreatePoolRequest($"pool-{UniqueSuffix()}", "Managed Pool", details, [coin.Id]));

        response.StatusCode.Should().Be(HttpStatusCode.Created);
    }

    [Fact]
    public async Task Pools_RejectDuplicateSystemPoolId()
    {
        var admin = await factory.CreateAdminClientAsync();
        var coin = await CreateCoinAsync(admin);
        var systemPoolId = $"pool-{UniqueSuffix()}";

        var first = await admin.PostAsJsonAsync(
            "/api/admin/pools",
            new CreatePoolRequest(systemPoolId, "Pool", AutoPayoutDetails(coin.Id), [coin.Id]));
        first.StatusCode.Should().Be(HttpStatusCode.Created);

        var duplicate = await admin.PostAsJsonAsync(
            "/api/admin/pools",
            new CreatePoolRequest(systemPoolId, "Pool again", AutoPayoutDetails(coin.Id), [coin.Id]));

        duplicate.StatusCode.Should().Be(HttpStatusCode.Conflict);
    }

    // --- Mining algorithms ---------------------------------------------------------------

    [Fact]
    public async Task MiningAlgos_CanBeCreatedAndListedByPriority()
    {
        var admin = await factory.CreateAdminClientAsync();

        var low = await admin.PostAsJsonAsync("/api/admin/mining-algos", new
        {
            code = $"algo-low-{UniqueSuffix()}",
            name = "Low priority",
            priority = 10,
            configuration = new { bin = "rgminer", args = new[] { "--algo", "pearl" } },
        });
        low.StatusCode.Should().Be(HttpStatusCode.Created);

        var high = await admin.PostAsJsonAsync("/api/admin/mining-algos", new
        {
            code = $"algo-high-{UniqueSuffix()}",
            name = "High priority",
            priority = 500,
            configuration = (object?)null,
        });
        high.StatusCode.Should().Be(HttpStatusCode.Created);

        var highAlgo = await high.Content.ReadFromJsonAsync<MiningAlgoResponse>();
        highAlgo!.Configuration.Should().BeNull();

        var lowAlgo = await low.Content.ReadFromJsonAsync<MiningAlgoResponse>();
        lowAlgo!.Status.Should().Be("active");
        lowAlgo.Configuration!.Value.GetProperty("bin").GetString().Should().Be("rgminer");
        lowAlgo.Configuration.Value.GetProperty("args").GetArrayLength().Should().Be(2);

        var listed = await admin.GetFromJsonAsync<List<MiningAlgoResponse>>("/api/admin/mining-algos");

        // Highest priority first.
        var ours = listed!.Where(algo => algo.Id == lowAlgo.Id || algo.Id == highAlgo.Id).ToList();
        ours.Should().HaveCount(2);
        ours[0].Id.Should().Be(highAlgo.Id);
    }

    [Fact]
    public async Task MiningAlgos_RejectDuplicateCode()
    {
        var admin = await factory.CreateAdminClientAsync();
        var code = $"algo-{UniqueSuffix()}";

        var first = await admin.PostAsJsonAsync(
            "/api/admin/mining-algos",
            new CreateMiningAlgoRequest(code, "First", 1, null));
        first.StatusCode.Should().Be(HttpStatusCode.Created);

        var duplicate = await admin.PostAsJsonAsync(
            "/api/admin/mining-algos",
            new CreateMiningAlgoRequest(code.ToUpperInvariant(), "Second", 2, null));

        duplicate.StatusCode.Should().Be(HttpStatusCode.Conflict);
    }

    [Fact]
    public async Task MiningAlgos_CanBeUpdated()
    {
        var admin = await factory.CreateAdminClientAsync();

        var created = await admin.PostAsJsonAsync(
            "/api/admin/mining-algos",
            new CreateMiningAlgoRequest($"algo-{UniqueSuffix()}", "Before", 5, null));
        var algorithm = await created.Content.ReadFromJsonAsync<MiningAlgoResponse>();

        var update = await admin.PutAsJsonAsync(
            $"/api/admin/mining-algos/{algorithm!.Id}",
            new UpdateMiningAlgoRequest("After", 99, "disabled", JsonDocument.Parse("{\"bin\":\"peakminer\"}").RootElement));
        update.StatusCode.Should().Be(HttpStatusCode.OK);

        var updated = await update.Content.ReadFromJsonAsync<MiningAlgoResponse>();
        updated!.Name.Should().Be("After");
        updated.Priority.Should().Be(99);
        updated.Status.Should().Be("disabled");
        updated.Configuration!.Value.GetProperty("bin").GetString().Should().Be("peakminer");
    }
}
