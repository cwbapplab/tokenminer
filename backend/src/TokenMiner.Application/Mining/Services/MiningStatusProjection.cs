namespace TokenMiner.Application.Mining.Services;

/// <summary>
/// Derives the mining status shown to operators and clients from the device's last accepted share.
/// A device is <c>running</c> when that share is inside the activity window, otherwise <c>idle</c>.
/// There is no third status, and nothing computes or stores it in the background — it is resolved
/// per request from the share rows.
/// </summary>
internal static class MiningStatusProjection
{
    public const string Running = "running";
    public const string Idle = "idle";

    public static string Resolve(DateTimeOffset? lastAcceptedShareAt, DateTimeOffset now, int windowSeconds)
        => lastAcceptedShareAt is { } last && last >= now.AddSeconds(-windowSeconds)
            ? Running
            : Idle;
}
