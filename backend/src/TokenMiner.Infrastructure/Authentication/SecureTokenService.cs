using System.Security.Cryptography;
using System.Text;
using TokenMiner.Application.Authentication.Abstractions;

namespace TokenMiner.Infrastructure.Authentication;

/// <summary>
/// Produces opaque, URL-safe tokens and their storage hashes. Tokens carry 256 bits of
/// entropy and are stored only as SHA-256 digests.
/// </summary>
internal sealed class SecureTokenService : ISecureTokenService
{
    private const int TokenSizeBytes = 32;

    public string GenerateToken() => Base64UrlEncode(RandomNumberGenerator.GetBytes(TokenSizeBytes));

    public string Hash(string token) =>
        Convert.ToHexString(SHA256.HashData(Encoding.UTF8.GetBytes(token)));

    private static string Base64UrlEncode(byte[] bytes) =>
        Convert.ToBase64String(bytes).TrimEnd('=').Replace('+', '-').Replace('/', '_');
}
