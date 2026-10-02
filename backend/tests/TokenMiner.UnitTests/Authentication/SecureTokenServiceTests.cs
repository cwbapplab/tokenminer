using FluentAssertions;
using TokenMiner.Infrastructure.Authentication;
using Xunit;

namespace TokenMiner.UnitTests.Authentication;

public sealed class SecureTokenServiceTests
{
    private readonly SecureTokenService _service = new();

    [Fact]
    public void GenerateToken_IsUrlSafeAndSuitablyLong()
    {
        var token = _service.GenerateToken();

        token.Should().MatchRegex("^[A-Za-z0-9_-]+$");
        // 32 random bytes base64url-encoded is 43 characters.
        token.Should().HaveLength(43);
    }

    [Fact]
    public void GenerateToken_IsUnique()
    {
        var tokens = Enumerable.Range(0, 200).Select(_ => _service.GenerateToken()).ToList();

        tokens.Should().OnlyHaveUniqueItems();
    }

    [Fact]
    public void Hash_IsDeterministicHex()
    {
        var hash = _service.Hash("some-token");

        hash.Should().MatchRegex("^[0-9A-F]{64}$");
        hash.Should().Be(_service.Hash("some-token"));
    }

    [Fact]
    public void Hash_DiffersForDifferentInput()
    {
        _service.Hash("token-a").Should().NotBe(_service.Hash("token-b"));
    }
}
