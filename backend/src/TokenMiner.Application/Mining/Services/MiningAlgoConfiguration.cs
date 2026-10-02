using System.Text.Json;

namespace TokenMiner.Application.Mining.Services;

/// <summary>
/// The subset of a mining algorithm's JSON configuration needed to render a launch command.
/// Unrecognised members are ignored so the stored blob can carry algorithm-specific extras.
/// </summary>
public sealed record MiningAlgoConfiguration(string Command, IReadOnlyList<string> SupportedCoins)
{
    public static bool TryParse(string? rawJson, out MiningAlgoConfiguration configuration)
    {
        configuration = null!;

        if (string.IsNullOrWhiteSpace(rawJson))
        {
            return false;
        }

        try
        {
            using var document = JsonDocument.Parse(rawJson);

            if (document.RootElement.ValueKind != JsonValueKind.Object)
            {
                return false;
            }

            if (!document.RootElement.TryGetProperty("command", out var commandElement)
                || commandElement.ValueKind != JsonValueKind.String)
            {
                return false;
            }

            var command = commandElement.GetString();
            if (string.IsNullOrWhiteSpace(command))
            {
                return false;
            }

            var supportedCoins = new List<string>();
            if (document.RootElement.TryGetProperty("supportedCoins", out var coinsElement)
                && coinsElement.ValueKind == JsonValueKind.Array)
            {
                foreach (var coinElement in coinsElement.EnumerateArray())
                {
                    var coinCode = coinElement.GetString();
                    if (!string.IsNullOrWhiteSpace(coinCode))
                    {
                        supportedCoins.Add(coinCode.Trim().ToLowerInvariant());
                    }
                }
            }

            configuration = new MiningAlgoConfiguration(command, supportedCoins);
            return true;
        }
        catch (JsonException)
        {
            return false;
        }
    }

    /// <summary>An empty coin list means the algorithm is eligible for every coin.</summary>
    public bool SupportsCoin(string coinCode) =>
        SupportedCoins.Count == 0 || SupportedCoins.Contains(coinCode.ToLowerInvariant());
}

/// <summary>Substitutes <c>{placeholder}</c> tokens in a stored command template.</summary>
public static class MiningCommandRenderer
{
    public static string Render(string template, IReadOnlyDictionary<string, string> values)
    {
        var rendered = template;

        foreach (var (key, value) in values)
        {
            rendered = rendered.Replace($"{{{key}}}", value, StringComparison.OrdinalIgnoreCase);
        }

        return rendered;
    }
}
