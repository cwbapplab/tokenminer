using System.Net.Http.Headers;
using System.Net.Http.Json;
using System.Text.Json;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.Logging;
using TokenMiner.Application.Providers;
using TokenMiner.Application.Providers.Abstractions;

namespace TokenMiner.Infrastructure.Integrations;

/// <summary>
/// Reads the prepaid balance and model catalogue from an OpenAI-compatible LLM gateway.
/// </summary>
/// <remarks>
/// Confirmed against router.one: <c>GET /v1/balance</c> returns
/// <c>{ balance, reserved_balance, total_balance }</c> in USD, and <c>GET /v1/models</c> returns an
/// OpenAI-shaped <c>{ data: [...] }</c> catalogue. There is no documented endpoint for initiating
/// a top-up, which is why deposits are sent on chain and reconciled through the balance.
/// </remarks>
internal sealed class OpenAiCompatibleLlmProviderClient(
    HttpClient httpClient,
    ILogger<OpenAiCompatibleLlmProviderClient> logger) : ILlmProviderClient
{
    public string Name => "router-one";

    public async Task<LlmBalance> GetBalanceAsync(
        LlmProviderContext context,
        CancellationToken cancellationToken)
    {
        using var request = CreateRequest(context, $"{context.Endpoint.TrimEnd('/')}/balance");

        using var response = await httpClient.SendAsync(request, cancellationToken);
        response.EnsureSuccessStatusCode();

        using var payload = await response.Content.ReadAsJsonDocumentAsync(cancellationToken);
        var root = payload.RootElement;

        return new LlmBalance(
            ReadDecimal(root, "balance") ?? 0m,
            ReadDecimal(root, "reserved_balance") ?? 0m,
            ReadDecimal(root, "total_balance") ?? ReadDecimal(root, "balance") ?? 0m);
    }

    public async Task<IReadOnlyList<LlmCatalogModel>> GetModelsAsync(
        LlmProviderContext context,
        CancellationToken cancellationToken)
    {
        using var request = CreateRequest(context, $"{context.Endpoint.TrimEnd('/')}/models");

        using var response = await httpClient.SendAsync(request, cancellationToken);
        response.EnsureSuccessStatusCode();

        using var payload = await response.Content.ReadAsJsonDocumentAsync(cancellationToken);

        if (!payload.RootElement.TryGetProperty("data", out var data)
            || data.ValueKind != JsonValueKind.Array)
        {
            logger.LogWarning("The catalogue response from {Provider} had no data array.", context.Name);
            return [];
        }

        var models = new List<LlmCatalogModel>();

        foreach (var entry in data.EnumerateArray())
        {
            var modelId = ReadString(entry, "id");
            if (string.IsNullOrWhiteSpace(modelId))
            {
                continue;
            }

            models.Add(new LlmCatalogModel(
                modelId,
                ReadString(entry, "name") ?? ReadString(entry, "display_name"),
                ReadDecimal(entry, "input_cost") ?? ReadNestedDecimal(entry, "pricing", "input"),
                ReadDecimal(entry, "output_cost") ?? ReadNestedDecimal(entry, "pricing", "output"),
                ReadDecimal(entry, "cached_input_cost") ?? ReadNestedDecimal(entry, "pricing", "cached_input"),
                ReadString(entry, "currency") ?? "USD",
                ReadInt(entry, "context_length") ?? ReadInt(entry, "max_tokens"),
                ReadStringArray(entry, "capabilities")));
        }

        return models;
    }

    private static HttpRequestMessage CreateRequest(LlmProviderContext context, string url)
    {
        var request = new HttpRequestMessage(HttpMethod.Get, url);

        if (!string.IsNullOrWhiteSpace(context.ApiKey))
        {
            request.Headers.Authorization = new AuthenticationHeaderValue("Bearer", context.ApiKey);
        }

        return request;
    }

    private static string? ReadString(JsonElement element, string name) =>
        element.TryGetProperty(name, out var value) && value.ValueKind == JsonValueKind.String
            ? value.GetString()
            : null;

    private static decimal? ReadDecimal(JsonElement element, string name) =>
        element.TryGetProperty(name, out var value)
        && value.ValueKind == JsonValueKind.Number
        && value.TryGetDecimal(out var parsed)
            ? parsed
            : null;

    private static decimal? ReadNestedDecimal(JsonElement element, string parent, string name) =>
        element.TryGetProperty(parent, out var nested) && nested.ValueKind == JsonValueKind.Object
            ? ReadDecimal(nested, name)
            : null;

    private static int? ReadInt(JsonElement element, string name) =>
        element.TryGetProperty(name, out var value)
        && value.ValueKind == JsonValueKind.Number
        && value.TryGetInt32(out var parsed)
            ? parsed
            : null;

    private static IReadOnlyList<string> ReadStringArray(JsonElement element, string name)
    {
        if (!element.TryGetProperty(name, out var value) || value.ValueKind != JsonValueKind.Array)
        {
            return [];
        }

        return value.EnumerateArray()
            .Where(item => item.ValueKind == JsonValueKind.String)
            .Select(item => item.GetString()!)
            .ToList();
    }
}

internal static class HttpContentJsonExtensions
{
    /// <summary>Reads the body as a JSON document, keeping the stream inside a known lifetime.</summary>
    public static async Task<JsonDocument> ReadAsJsonDocumentAsync(
        this HttpContent content,
        CancellationToken cancellationToken)
    {
        await using var stream = await content.ReadAsStreamAsync(cancellationToken);

        return await JsonDocument.ParseAsync(stream, cancellationToken: cancellationToken);
    }
}

/// <summary>
/// Reads secrets from configuration under <c>Secrets:&lt;reference&gt;</c>.
/// </summary>
/// <remarks>
/// Adequate for development and for a single-tenant deployment where configuration is itself
/// protected. A vault or key-management adapter replaces this class without touching callers.
/// </remarks>
internal sealed class ConfigurationSecretResolver(IConfiguration configuration) : ISecretResolver
{
    public Task<string?> GetSecretAsync(string credentialRef, CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();

        return Task.FromResult(configuration[$"Secrets:{credentialRef}"]);
    }
}

/// <summary>
/// Deterministic stand-in for a blockchain. Broadcasting returns a stable hash derived from the
/// request's idempotency key, so a retry produces the same transfer rather than a second one.
/// </summary>
internal sealed class SimulatedBlockchainTransferProvider(ProviderOptions options)
    : IBlockchainTransferProvider
{
    public string Name => "simulated";

    public Task<BlockchainTransferResult> SendAsync(
        BlockchainTransferRequest request,
        CancellationToken cancellationToken) =>
        Task.FromResult(new BlockchainTransferResult(DeriveHash(request.IdempotencyKey), null));

    public Task<int> GetConfirmationsAsync(
        string network,
        string transactionHash,
        CancellationToken cancellationToken) =>
        Task.FromResult(Math.Max(0, options.SimulatedConfirmations));

    /// <summary>
    /// A stable digest, so the same transfer always reports the same hash. String.GetHashCode is
    /// randomised per process and would not be reproducible across restarts.
    /// </summary>
    private static string DeriveHash(string idempotencyKey) =>
        $"sim-{Convert.ToHexString(System.Security.Cryptography.SHA256.HashData(
            System.Text.Encoding.UTF8.GetBytes(idempotencyKey)))[..24].ToLowerInvariant()}";
}
