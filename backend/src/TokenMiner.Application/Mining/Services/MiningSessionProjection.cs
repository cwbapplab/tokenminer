using System.Text.Json;
using TokenMiner.Application.Common.Exceptions;
using TokenMiner.Application.Mining.Models;
using TokenMiner.Domain.Mining;
using TokenMiner.Domain.Mining.Enums;

namespace TokenMiner.Application.Mining.Services;

/// <summary>
/// Builds the session DTO that clients receive, shared by "start" and "get current" so the two
/// can never disagree about the launch command, the config, or the proxy endpoint.
/// </summary>
internal static class MiningSessionProjection
{
    /// <summary>
    /// The public proxy endpoint clients must connect to. Mining always goes through the proxy,
    /// so with none configured there is nowhere for a client to connect: refuse rather than
    /// hand out a pool endpoint.
    /// </summary>
    public static string RequirePublicStratumEndpoint(MiningOptions options)
    {
        if (string.IsNullOrWhiteSpace(options.PublicStratumEndpoint))
        {
            throw new ConflictException(
                "The public Stratum proxy endpoint is not configured (Mining:PublicStratumEndpoint).");
        }

        return options.PublicStratumEndpoint.Trim();
    }

    public static MiningSessionDto Build(
        UserHardwareMiner session,
        UserHardware device,
        Pool pool,
        PoolDetails poolDetails,
        Coin coin,
        MiningAlgo algorithm,
        MiningOptions options)
    {
        var publicStratumHost = RequirePublicStratumEndpoint(options);

        var upstreamStratumHost = string.IsNullOrWhiteSpace(poolDetails.StratumEndpoint)
            ? StripScheme(poolDetails.BaseUrl)
            : poolDetails.StratumEndpoint;

        var walletAddress = poolDetails.PayoutAddress ?? string.Empty;

        // Placeholders available to both the legacy command and the structured config. `endpoint`
        // and `pool.payoutAddress` are aliases the config templates use; the endpoint is always the
        // *proxy*, never the pool, because clients must not connect to pools.
        var values = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase)
        {
            ["coin"] = coin.Code,
            ["coinName"] = coin.Name,
            ["algo"] = algorithm.Code,
            ["poolUrl"] = poolDetails.BaseUrl,
            ["poolHost"] = StripScheme(poolDetails.BaseUrl),
            ["stratumHost"] = publicStratumHost,
            ["endpoint"] = publicStratumHost,
            ["poolEndpoint"] = publicStratumHost,
            ["upstreamStratumHost"] = upstreamStratumHost,
            ["wallet"] = walletAddress,
            ["pool.payoutAddress"] = walletAddress,
            ["workerId"] = session.WorkerIdentifier,
            ["worker"] = session.WorkerIdentifier,
            ["hardwareId"] = device.HardwareId.ToString(),
        };

        _ = MiningAlgoConfiguration.TryParse(algorithm.Configuration, out var configuration);

        var minerCommand = configuration is null
            ? string.Empty
            : MiningCommandRenderer.Render(configuration.Command, values);

        var minerConfig = configuration?.Template is { } template
            ? MiningCommandRenderer.RenderConfig(template, values)
            : EmptyObject();

        return new MiningSessionDto(
            session.Id,
            pool.Id,
            pool.Name,
            coin.Id,
            coin.Code,
            algorithm.Id,
            algorithm.Code,
            session.WorkerIdentifier,
            minerCommand,
            minerConfig,
            publicStratumHost,
            session.Status.ToDbValue(),
            session.StartedAt);
    }

    private static JsonElement EmptyObject()
    {
        using var document = JsonDocument.Parse("{}");

        return document.RootElement.Clone();
    }

    private static string StripScheme(string url)
    {
        var separator = url.IndexOf("://", StringComparison.Ordinal);

        return separator >= 0 ? url[(separator + 3)..] : url;
    }
}
