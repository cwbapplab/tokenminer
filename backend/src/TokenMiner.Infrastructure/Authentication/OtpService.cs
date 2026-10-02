using System.Globalization;
using System.Security.Cryptography;
using System.Text;
using TokenMiner.Application.Authentication;
using TokenMiner.Application.Authentication.Abstractions;

namespace TokenMiner.Infrastructure.Authentication;

/// <summary>
/// Generates numeric codes with a cryptographic RNG and stores only an HMAC of them, so a
/// database leak does not reveal usable codes. Comparison is constant time.
/// </summary>
internal sealed class OtpService : IOtpService
{
    private readonly AuthOptions.OtpOptions _options;
    private readonly byte[] _hmacKey;

    public OtpService(AuthOptions authOptions)
    {
        _options = authOptions.Otp;

        // Fall back to the JWT key so development setups cannot silently run with an empty key.
        var keyMaterial = string.IsNullOrWhiteSpace(_options.HmacKey)
            ? authOptions.Jwt.SigningKey
            : _options.HmacKey;

        _hmacKey = Encoding.UTF8.GetBytes(keyMaterial);
    }

    public string GenerateCode()
    {
        var upperBoundExclusive = (int)Math.Pow(10, _options.Length);
        var value = RandomNumberGenerator.GetInt32(0, upperBoundExclusive);

        return value.ToString(CultureInfo.InvariantCulture).PadLeft(_options.Length, '0');
    }

    public string Hash(string code)
    {
        using var hmac = new HMACSHA256(_hmacKey);
        return Convert.ToHexString(hmac.ComputeHash(Encoding.UTF8.GetBytes(code)));
    }

    public bool Verify(string code, string codeHash) =>
        CryptographicOperations.FixedTimeEquals(
            Encoding.ASCII.GetBytes(Hash(code)),
            Encoding.ASCII.GetBytes(codeHash));
}
