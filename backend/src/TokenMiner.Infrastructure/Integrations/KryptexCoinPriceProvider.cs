using System.Net.Http.Json;
using System.Text.Json.Serialization;
using Microsoft.Extensions.Logging;
using TokenMiner.Application.Treasury;
using TokenMiner.Application.Treasury.Abstractions;

namespace TokenMiner.Infrastructure.Integrations;

/// <summary>
/// Coin prices from the pool's public price chart:
/// <c>/api/v1/coin/{coin}/price/chart?time_range=day</c>. The most recent point is used.
/// </summary>
internal sealed class KryptexCoinPriceProvider(
    HttpClient httpClient,
    TreasuryOptions options,
    ILogger<KryptexCoinPriceProvider> logger) : ICoinPriceProvider
{
    public string Source => "kryptex";

    public async Task<decimal?> GetUsdPriceAsync(string coinCode, CancellationToken cancellationToken)
    {
        var baseUrl = string.IsNullOrWhiteSpace(options.PriceSourceBaseUrl)
            ? "https://pool.kryptex.com"
            : options.PriceSourceBaseUrl.TrimEnd('/');

        var url = $"{baseUrl}/api/v1/coin/{Uri.EscapeDataString(coinCode)}/price/chart?time_range=day";

        var points = await httpClient.GetFromJsonAsync<List<KryptexPricePoint>>(url, cancellationToken);

        if (points is null || points.Count == 0)
        {
            logger.LogDebug("No price points returned for {Coin}.", coinCode);
            return null;
        }

        // The chart is chronological; the last point is the current price.
        var latest = points
            .Where(point => point.Price > 0)
            .OrderBy(point => point.Timestamp)
            .LastOrDefault();

        return latest?.Price;
    }

    private sealed record KryptexPricePoint
    {
        [JsonPropertyName("timestamp")]
        public long Timestamp { get; init; }

        [JsonPropertyName("price")]
        public decimal Price { get; init; }
    }
}
