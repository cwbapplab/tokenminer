using System.Net.Http.Json;
using System.Text.Json.Serialization;
using Microsoft.Extensions.Logging;
using TokenMiner.Application.Treasury.Abstractions;

namespace TokenMiner.Infrastructure.Integrations;

/// <summary>
/// Kryptex pool integration, built on the pool's public read-only API:
/// <c>/{coin}/api/v1/miner/balance/{address}</c> and <c>/{coin}/api/v1/miner/payouts/{address}</c>.
/// </summary>
/// <remarks>
/// Kryptex has no public endpoint for requesting a payout — that action only exists in the web
/// UI — so <see cref="RequestWithdrawalAsync"/> delegates to <see cref="IPoolWithdrawalClient"/>.
/// The pool's own coin slug is expected to match <c>coins.code</c>.
/// </remarks>
internal sealed class KryptexPoolPayoutProvider(
    HttpClient httpClient,
    IPoolWithdrawalClient withdrawalClient,
    ILogger<KryptexPoolPayoutProvider> logger) : IPoolPayoutProvider
{
    /// <summary>Statuses Kryptex uses for a payout that has finished.</summary>
    private static readonly string[] SettledStatuses =
        ["confirmed", "complete", "completed", "done", "paid", "sent"];

    public string Name => "kryptex";

    public async Task<PoolBalance> GetBalanceAsync(
        PoolPayoutContext context,
        CancellationToken cancellationToken)
    {
        var address = RequireAddress(context);
        var url = $"{BaseUrl(context)}/{Uri.EscapeDataString(context.CoinCode)}/api/v1/miner/balance/{Uri.EscapeDataString(address)}";

        var payload = await httpClient.GetFromJsonAsync<KryptexBalance>(url, cancellationToken)
            ?? throw new InvalidOperationException($"Kryptex returned no balance for {context.CoinCode}.");

        return new PoolBalance(
            payload.Total,
            payload.Confirmed,
            payload.Unconfirmed,
            payload.Threshold);
    }

    public async Task<IReadOnlyList<PoolPayoutRecord>> GetPayoutsAsync(
        PoolPayoutContext context,
        CancellationToken cancellationToken)
    {
        var address = RequireAddress(context);
        var url = $"{BaseUrl(context)}/{Uri.EscapeDataString(context.CoinCode)}/api/v1/miner/payouts/{Uri.EscapeDataString(address)}";

        var payload = await httpClient.GetFromJsonAsync<KryptexPayouts>(url, cancellationToken);

        if (payload is null)
        {
            return [];
        }

        var records = new List<PoolPayoutRecord>(payload.Results.Count);

        foreach (var payout in payload.Results)
        {
            var settled = IsSettled(payout.Status);

            if (!settled && payout.Status is not null)
            {
                logger.LogDebug(
                    "Kryptex reported payout status {Status}, which is not in the settled set.",
                    payout.Status);
            }

            records.Add(new PoolPayoutRecord(
                payout.TransactionId,
                payout.Received,
                RequestedAt: null,
                ReceivedAt: payout.Date > 0
                    ? DateTimeOffset.FromUnixTimeSeconds(payout.Date)
                    : null,
                Confirmations: settled ? 1 : 0,
                IsSettled: settled,
                payout.Status));
        }

        return records;
    }

    public Task<PoolWithdrawalResult> RequestWithdrawalAsync(
        PoolPayoutContext context,
        decimal amount,
        CancellationToken cancellationToken) =>
        withdrawalClient.RequestAsync(context, amount, cancellationToken);

    private static string BaseUrl(PoolPayoutContext context) =>
        string.IsNullOrWhiteSpace(context.BaseUrl)
            ? "https://pool.kryptex.com"
            : context.BaseUrl.TrimEnd('/');

    private static string RequireAddress(PoolPayoutContext context) =>
        string.IsNullOrWhiteSpace(context.PayoutAddress)
            ? throw new InvalidOperationException(
                $"Pool {context.SystemPoolId} has no payout address configured.")
            : context.PayoutAddress;

    private static bool IsSettled(string? status) =>
        status is not null && SettledStatuses.Contains(status.Trim().ToLowerInvariant());

    private sealed record KryptexBalance
    {
        [JsonPropertyName("total")]
        public decimal Total { get; init; }

        [JsonPropertyName("threshold")]
        public decimal? Threshold { get; init; }

        [JsonPropertyName("unconfirmed")]
        public decimal Unconfirmed { get; init; }

        [JsonPropertyName("confirmed")]
        public decimal Confirmed { get; init; }
    }

    private sealed record KryptexPayouts
    {
        [JsonPropertyName("results")]
        public List<KryptexPayout> Results { get; init; } = [];
    }

    private sealed record KryptexPayout
    {
        [JsonPropertyName("date")]
        public long Date { get; init; }

        [JsonPropertyName("received")]
        public decimal Received { get; init; }

        [JsonPropertyName("txid")]
        public string? TransactionId { get; init; }

        [JsonPropertyName("status")]
        public string? Status { get; init; }
    }
}
