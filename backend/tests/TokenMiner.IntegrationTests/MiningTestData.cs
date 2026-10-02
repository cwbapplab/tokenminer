using System.Net.Http.Json;
using TokenMiner.Contracts.Mining;

namespace TokenMiner.IntegrationTests;

/// <summary>Shared fixtures for mining tests: a coin, a pool and an algorithm wired together.</summary>
internal static class MiningTestData
{
    /// <summary>Must match the secret the test host is configured with.</summary>
    public const string ServiceSharedSecret = "test-service-shared-secret";

    public sealed record Setup(CoinResponse Coin, PoolResponse Pool, MiningAlgoResponse Algo);

    public static string Suffix() => Guid.NewGuid().ToString("N")[..8];

    public static async Task<Setup> SeedAsync(HttpClient admin)
    {
        var coinResponse = await admin.PostAsJsonAsync(
            "/api/admin/coins",
            new CreateCoinRequest($"coin-{Suffix()}", "Session Coin", "testnet", 8));
        var coin = (await coinResponse.Content.ReadFromJsonAsync<CoinResponse>())!;

        // Only this algorithm advertises the coin, so selection is unambiguous.
        var algorithmResponse = await admin.PostAsJsonAsync("/api/admin/mining-algos", new
        {
            code = $"algo-{Suffix()}",
            name = "Session Algorithm",
            priority = 900,
            configuration = new
            {
                command = "rgminer.exe --algo {algo} --stratum {poolHost} --wallet {wallet} --worker-name {workerId}",
                supportedCoins = new[] { coin.Code },
            },
        });
        var algorithm = (await algorithmResponse.Content.ReadFromJsonAsync<MiningAlgoResponse>())!;

        var poolResponse = await admin.PostAsJsonAsync(
            "/api/admin/pools",
            new CreatePoolRequest(
                $"pool-{Suffix()}",
                "Session Pool",
                new PoolDetailsRequest(
                    "https://pool.example.com",
                    "/status",
                    "pool.example.com:7049",
                    coin.Id,
                    "native_auto_payout",
                    "wallet-123",
                    "bep20"),
                [coin.Id]));
        var pool = (await poolResponse.Content.ReadFromJsonAsync<PoolResponse>())!;

        return new Setup(coin, pool, algorithm);
    }
}
