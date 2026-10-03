using System.Globalization;
using System.Security.Cryptography;

namespace MockPool.Stratum;

/// <summary>A mining job pushed to the client in a <c>mining.notify</c> notification.</summary>
public sealed class StratumJob
{
    public StratumJob(string jobId)
    {
        JobId = jobId;
        PrevHash = RandomNumberGenerator.GetHexString(64);
        Coinbase1 = RandomNumberGenerator.GetHexString(16);
        Coinbase2 = RandomNumberGenerator.GetHexString(32);
        MerkleBranches = [];
        Version = "20000000";
        NBits = "1d00ffff";
        NTime = DateTimeOffset.UtcNow.ToUnixTimeSeconds().ToString("x8", CultureInfo.InvariantCulture);
    }

    public string JobId { get; }

    public string PrevHash { get; }

    public string Coinbase1 { get; }

    public string Coinbase2 { get; }

    public IReadOnlyList<string> MerkleBranches { get; }

    public string Version { get; }

    public string NBits { get; }

    public string NTime { get; }

    public bool CleanJobs => true;

    public object?[] ToNotifyParams() =>
    [
        JobId, PrevHash, Coinbase1, Coinbase2, MerkleBranches, Version, NBits, NTime, CleanJobs,
    ];
}
