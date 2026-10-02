using System.Text;
using FluentAssertions;
using TokenMiner.Application.Mining.Services;
using Xunit;

namespace TokenMiner.UnitTests.Mining;

/// <summary>
/// Pins the exact bytes of the service signature so the Rust stratum proxy can be held to the
/// same contract. The identical vector is asserted in <c>stratum-proxy/api-client</c>; if either
/// implementation drifts, one of the two tests fails.
/// </summary>
public sealed class ServiceSignatureVectorTests
{
    private const string Secret = "cross-language-secret";
    private const string Method = "POST";
    private const string Path = "/internal/mining/shares";
    private const string Timestamp = "1767225600";
    private const string Nonce = "0123456789abcdef";
    private const string Body = """{"a":1}""";

    private const string ExpectedBodyHash =
        "015ABD7F5CC57A2DD94B7590F04AD8084273905EE33EC5CEBEAE62276A97F862";

    private const string ExpectedSignature =
        "0705551D50F70830D9F337F41476086C07942D3C4B156F128F2067C6952AC8B5";

    [Fact]
    public void BodyHash_MatchesTheCrossLanguageVector()
    {
        var hash = ServiceRequestSignature.ComputeBodyHash(Encoding.UTF8.GetBytes(Body));

        hash.Should().Be(ExpectedBodyHash);
    }

    [Fact]
    public void Signature_MatchesTheCrossLanguageVector()
    {
        var hash = ServiceRequestSignature.ComputeBodyHash(Encoding.UTF8.GetBytes(Body));

        var signature = ServiceRequestSignature.Compute(
            Secret,
            Method,
            Path,
            Timestamp,
            Nonce,
            hash);

        signature.Should().Be(ExpectedSignature);
    }
}
