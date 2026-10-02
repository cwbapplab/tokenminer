using System.Security.Cryptography;
using System.Text;

namespace TokenMiner.Application.Mining.Services;

/// <summary>
/// Signing scheme for service-to-service calls. The signature covers the HTTP method, path,
/// timestamp, nonce and a hash of the body, so a captured request can be neither replayed
/// against another route nor replayed with a different payload.
/// </summary>
public static class ServiceRequestSignature
{
    /// <summary>Hex-encoded SHA-256 of the raw request body (hex of an empty body when absent).</summary>
    public static string ComputeBodyHash(ReadOnlySpan<byte> body) =>
        Convert.ToHexString(SHA256.HashData(body));

    public static string Compute(
        string sharedSecret,
        string method,
        string path,
        string timestamp,
        string nonce,
        string bodyHash)
    {
        var canonical = string.Join(
            '\n',
            method.ToUpperInvariant(),
            path,
            timestamp,
            nonce,
            bodyHash);

        using var hmac = new HMACSHA256(Encoding.UTF8.GetBytes(sharedSecret));

        return Convert.ToHexString(hmac.ComputeHash(Encoding.UTF8.GetBytes(canonical)));
    }

    /// <summary>Constant-time comparison of a supplied signature against the expected one.</summary>
    public static bool Verify(
        string sharedSecret,
        string method,
        string path,
        string timestamp,
        string nonce,
        string bodyHash,
        string signature)
    {
        var expected = Compute(sharedSecret, method, path, timestamp, nonce, bodyHash);

        return CryptographicOperations.FixedTimeEquals(
            Encoding.ASCII.GetBytes(expected),
            Encoding.ASCII.GetBytes(signature));
    }
}
