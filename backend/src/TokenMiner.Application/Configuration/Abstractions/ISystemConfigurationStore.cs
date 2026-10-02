using System.Text.Json;
using TokenMiner.Domain.Configuration;

namespace TokenMiner.Application.Configuration.Abstractions;

/// <summary>
/// Reads and writes operator-tunable settings, decoupling the runtime from configuration files.
/// </summary>
public interface ISystemConfigurationStore
{
    Task<string?> GetRawAsync(string key, CancellationToken cancellationToken);

    /// <summary>Deserialises the stored JSON, or returns null when the key is unset or unusable.</summary>
    Task<T?> GetAsync<T>(string key, CancellationToken cancellationToken);

    Task SetAsync(string key, string value, DateTimeOffset now, CancellationToken cancellationToken);

    Task<IReadOnlyList<SystemConfiguration>> ListAsync(CancellationToken cancellationToken);
}

public static class SystemConfigurationStoreExtensions
{
    private static readonly JsonSerializerOptions Options = new(JsonSerializerDefaults.Web);

    /// <summary>Reads a decimal setting, falling back when it is unset or malformed.</summary>
    public static async Task<decimal> GetDecimalAsync(
        this ISystemConfigurationStore store,
        string key,
        decimal fallback,
        CancellationToken cancellationToken) =>
        await store.GetAsync<decimal?>(key, cancellationToken) ?? fallback;

    public static async Task<int> GetIntAsync(
        this ISystemConfigurationStore store,
        string key,
        int fallback,
        CancellationToken cancellationToken) =>
        await store.GetAsync<int?>(key, cancellationToken) ?? fallback;

    public static async Task<bool> GetBoolAsync(
        this ISystemConfigurationStore store,
        string key,
        bool fallback,
        CancellationToken cancellationToken) =>
        await store.GetAsync<bool?>(key, cancellationToken) ?? fallback;

    /// <summary>Serialises a value for storage.</summary>
    public static string Serialize<T>(T value) => JsonSerializer.Serialize(value, Options);

    /// <summary>Deserialises a stored value, falling back to default when it is unusable.</summary>
    public static T? Deserialize<T>(string? value)
    {
        if (string.IsNullOrWhiteSpace(value))
        {
            return default;
        }

        try
        {
            return JsonSerializer.Deserialize<T>(value, Options);
        }
        catch (JsonException)
        {
            // A malformed value falls back to the caller's default rather than failing the job.
            return default;
        }
    }
}
