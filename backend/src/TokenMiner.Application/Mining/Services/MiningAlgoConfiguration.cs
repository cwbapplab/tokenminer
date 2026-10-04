using System.Text.Json;

namespace TokenMiner.Application.Mining.Services;

/// <summary>
/// A mining algorithm's JSON configuration.
///
/// It carries either a legacy <c>command</c> string template, or structured fields the client
/// reads directly (<c>algo</c>, <c>endpoint</c>, <c>wallet</c>, <c>workerId</c>, <c>coin</c>).
/// The whole object is kept as <see cref="Template"/> so every string value can be rendered with
/// the session's placeholders before it reaches a client.
/// </summary>
public sealed record MiningAlgoConfiguration(
    string Command,
    IReadOnlyList<string> SupportedCoins,
    JsonElement? Template)
{
    /// <summary>Keys that make a configuration structured (usable without a <c>command</c>).</summary>
    private static readonly string[] StructuredKeys = ["algo", "endpoint", "wallet", "workerId", "coin"];

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

            var command = document.RootElement.TryGetProperty("command", out var commandElement)
                && commandElement.ValueKind == JsonValueKind.String
                ? commandElement.GetString() ?? string.Empty
                : string.Empty;

            var isStructured = StructuredKeys.Any(key => document.RootElement.TryGetProperty(key, out _));

            // A configuration with neither a command nor any structured field cannot drive a miner.
            if (string.IsNullOrWhiteSpace(command) && !isStructured)
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

            configuration = new MiningAlgoConfiguration(command, supportedCoins, document.RootElement.Clone());
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

/// <summary>Substitutes <c>{placeholder}</c> tokens in a stored template.</summary>
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

    /// <summary>
    /// Renders every string value in a JSON template, recursing through objects and arrays.
    /// Non-string values are copied unchanged.
    /// </summary>
    public static JsonElement RenderConfig(JsonElement template, IReadOnlyDictionary<string, string> values)
    {
        using var buffer = new MemoryStream();

        using (var writer = new Utf8JsonWriter(buffer))
        {
            WriteRendered(template, values, writer);
        }

        using var document = JsonDocument.Parse(buffer.ToArray());

        return document.RootElement.Clone();
    }

    private static void WriteRendered(JsonElement element, IReadOnlyDictionary<string, string> values, Utf8JsonWriter writer)
    {
        switch (element.ValueKind)
        {
            case JsonValueKind.Object:
                writer.WriteStartObject();
                foreach (var property in element.EnumerateObject())
                {
                    writer.WritePropertyName(property.Name);
                    WriteRendered(property.Value, values, writer);
                }
                writer.WriteEndObject();
                break;

            case JsonValueKind.Array:
                writer.WriteStartArray();
                foreach (var item in element.EnumerateArray())
                {
                    WriteRendered(item, values, writer);
                }
                writer.WriteEndArray();
                break;

            case JsonValueKind.String:
                writer.WriteStringValue(Render(element.GetString() ?? string.Empty, values));
                break;

            default:
                element.WriteTo(writer);
                break;
        }
    }
}
