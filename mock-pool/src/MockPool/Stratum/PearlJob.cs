using System.Globalization;
using System.Numerics;
using System.Security.Cryptography;

namespace MockPool.Stratum;

/// <summary>
/// A Pearl job. <c>mining.notify</c> carries the first 76 bytes of the (108-byte) block header — the
/// miner appends the 32-byte ProofCommitment it constructs while mining — plus the height, the opaque
/// job id and the 64-hex share target. See the Pearl stratum spec §5.1.
/// </summary>
public sealed class PearlJob
{
    private const string Version = "00004020";
    private const string Bits = "ffff001d";

    public PearlJob(string jobId, string target, int height)
    {
        JobId = jobId;
        Target = target;
        Height = height;

        var timestamp = ((uint)DateTimeOffset.UtcNow.ToUnixTimeSeconds()).ToString("x8", CultureInfo.InvariantCulture);

        // 4 version + 32 prev_block_hash + 32 merkle_root + 4 timestamp + 4 bits = 76 bytes, LE fields.
        Header = Version
            + RandomNumberGenerator.GetHexString(64, lowercase: true)
            + RandomNumberGenerator.GetHexString(64, lowercase: true)
            + timestamp
            + Bits;
    }

    public string JobId { get; }

    public string Header { get; }

    public int Height { get; }

    public string Target { get; }

    public object ToNotifyParams() => new { header = Header, height = Height, job_id = JobId, target = Target };

    /// <summary>Big-endian 256-bit share target for a share difficulty: <c>2^256 / difficulty</c>.</summary>
    public static string TargetFor(double difficulty)
    {
        var divisor = new BigInteger(Math.Max(1.0, difficulty));
        var value = BigInteger.Pow(2, 256) / divisor;
        var bytes = value.ToByteArray(isUnsigned: true, isBigEndian: true);

        var padded = new byte[32];
        var length = Math.Min(32, bytes.Length);
        Array.Copy(bytes, bytes.Length - length, padded, 32 - length, length);
        return Convert.ToHexString(padded).ToLowerInvariant();
    }
}
