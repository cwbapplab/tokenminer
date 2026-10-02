using System.Text;
using FluentAssertions;
using TokenMiner.Application.Mining.Services;
using Xunit;

namespace TokenMiner.UnitTests.Mining;

public sealed class ServiceRequestSignatureTests
{
    private const string Secret = "unit-test-shared-secret";
    private const string Method = "POST";
    private const string Path = "/internal/mining/shares";
    private const string Timestamp = "1767225600";
    private const string Nonce = "0123456789abcdef";

    private static string BodyHash(string body = """{"a":1}""") =>
        ServiceRequestSignature.ComputeBodyHash(Encoding.UTF8.GetBytes(body));

    [Fact]
    public void ComputeBodyHash_ProducesStableHexDigests()
    {
        var hash = BodyHash();

        hash.Should().MatchRegex("^[0-9A-F]{64}$");
        hash.Should().Be(BodyHash());
        hash.Should().NotBe(BodyHash("""{"a":2}"""));
    }

    [Fact]
    public void Compute_IsDeterministic()
    {
        var first = ServiceRequestSignature.Compute(Secret, Method, Path, Timestamp, Nonce, BodyHash());
        var second = ServiceRequestSignature.Compute(Secret, Method, Path, Timestamp, Nonce, BodyHash());

        first.Should().Be(second);
        first.Should().MatchRegex("^[0-9A-F]{64}$");
    }

    [Fact]
    public void Compute_IsCaseInsensitiveOnTheMethod()
    {
        var upper = ServiceRequestSignature.Compute(Secret, "POST", Path, Timestamp, Nonce, BodyHash());
        var lower = ServiceRequestSignature.Compute(Secret, "post", Path, Timestamp, Nonce, BodyHash());

        lower.Should().Be(upper);
    }

    [Fact]
    public void Verify_AcceptsAMatchingSignature()
    {
        var signature = ServiceRequestSignature.Compute(Secret, Method, Path, Timestamp, Nonce, BodyHash());

        ServiceRequestSignature.Verify(Secret, Method, Path, Timestamp, Nonce, BodyHash(), signature)
            .Should().BeTrue();
    }

    [Theory]
    [InlineData("other-secret", Method, Path, Timestamp, Nonce, """{"a":1}""")]
    [InlineData(Secret, Method, "/internal/mining/other", Timestamp, Nonce, """{"a":1}""")]
    [InlineData(Secret, Method, Path, "1767225601", Nonce, """{"a":1}""")]
    [InlineData(Secret, Method, Path, Timestamp, "different-nonce", """{"a":1}""")]
    [InlineData(Secret, Method, Path, Timestamp, Nonce, """{"a":2}""")]
    public void Verify_RejectsAnyChangedInput(
        string secret,
        string method,
        string path,
        string timestamp,
        string nonce,
        string body)
    {
        var signature = ServiceRequestSignature.Compute(Secret, Method, Path, Timestamp, Nonce, BodyHash());

        ServiceRequestSignature.Verify(secret, method, path, timestamp, nonce, BodyHash(body), signature)
            .Should().BeFalse();
    }

    [Fact]
    public void Verify_RejectsATruncatedSignature()
    {
        var signature = ServiceRequestSignature.Compute(Secret, Method, Path, Timestamp, Nonce, BodyHash());

        ServiceRequestSignature.Verify(Secret, Method, Path, Timestamp, Nonce, BodyHash(), signature[..32])
            .Should().BeFalse();
    }

    [Fact]
    public void ComputeBodyHash_TreatsMissingBodyAsTheEmptyDigest()
    {
        ServiceRequestSignature.ComputeBodyHash(ReadOnlySpan<byte>.Empty)
            .Should().Be(ServiceRequestSignature.ComputeBodyHash([]));
    }
}
