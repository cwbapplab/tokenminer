using System.Text.Json;

namespace MockPool.Stratum;

/// <summary>A parsed Stratum request. A request without an id is a notification.</summary>
public sealed class StratumRequest
{
    public required bool HasId { get; init; }

    public required JsonElement Id { get; init; }

    public required string Method { get; init; }

    /// <summary>Positional (array) params for classic V1, or named (object) params for Pearl.</summary>
    public required JsonElement Params { get; init; }

    public bool IsNotification => !HasId;
}

/// <summary>
/// Stratum JSON-RPC envelopes. One message per line, LF-terminated, with an optional trailing CR on
/// read. Classic V1 carries array params and <c>[code, message, traceback]</c> errors; Pearl carries
/// object params and <c>{code, msg}</c> errors.
/// </summary>
public static class StratumWire
{
    private static readonly JsonSerializerOptions Options = new();

    /// <summary>
    /// Parses one line. Returns <c>true</c> for a request or notification; sets <paramref name="error"/>
    /// only when the line is malformed. A well-formed line that is not a request (for example a stray
    /// response) yields <c>false</c> with no error, and the caller should ignore it.
    /// </summary>
    public static bool TryParse(string line, out StratumRequest? request, out string? error)
    {
        request = null;
        error = null;

        JsonDocument document;
        try
        {
            document = JsonDocument.Parse(line);
        }
        catch (JsonException)
        {
            error = "Parse error";
            return false;
        }

        using (document)
        {
            var root = document.RootElement;
            if (root.ValueKind != JsonValueKind.Object)
            {
                error = "Invalid request";
                return false;
            }

            if (!root.TryGetProperty("method", out var methodElement) || methodElement.ValueKind != JsonValueKind.String)
            {
                return false;
            }

            var hasId = root.TryGetProperty("id", out var idElement) && idElement.ValueKind != JsonValueKind.Null;
            var hasParams = root.TryGetProperty("params", out var paramsElement)
                && paramsElement.ValueKind is JsonValueKind.Array or JsonValueKind.Object;

            request = new StratumRequest
            {
                HasId = hasId,
                Id = hasId ? idElement.Clone() : default,
                Method = methodElement.GetString()!,
                Params = hasParams ? paramsElement.Clone() : default,
            };
            return true;
        }
    }

    /// <summary>Success envelope for both dialects: <c>{"id":..,"result":..,"error":null}</c>.</summary>
    public static string Response(JsonElement? id, object? result) =>
        JsonSerializer.Serialize(new { id, result, error = (object?)null }, Options);

    /// <summary>Classic V1 error: <c>{"id":..,"result":null,"error":[code,msg,null]}</c>.</summary>
    public static string Error(JsonElement? id, int code, string message) =>
        JsonSerializer.Serialize(new { id, result = (object?)null, error = new object?[] { code, message, null } }, Options);

    /// <summary>Pearl error: <c>{"id":..,"result":null,"error":{"code":code,"msg":msg}}</c>.</summary>
    public static string PearlError(JsonElement? id, int code, string message) =>
        JsonSerializer.Serialize(new { id, result = (object?)null, error = new { code, msg = message } }, Options);

    /// <summary>Notification; <paramref name="parameters"/> is an array (classic) or an object (Pearl).</summary>
    public static string Notification(string method, object? parameters) =>
        JsonSerializer.Serialize(new { id = (JsonElement?)null, method, @params = parameters }, Options);
}
