using FluentAssertions;
using TokenMiner.Domain.Users;
using Xunit;

namespace TokenMiner.UnitTests.Users;

public sealed class RefreshTokenTests
{
    private static readonly DateTimeOffset Now = new(2026, 1, 1, 12, 0, 0, TimeSpan.Zero);

    private static RefreshToken CreateToken(TimeSpan? lifetime = null) =>
        new(
            Guid.NewGuid(),
            Guid.NewGuid(),
            "hash",
            Now.Add(lifetime ?? TimeSpan.FromDays(30)),
            Now,
            "127.0.0.1",
            "test-agent");

    [Fact]
    public void NewToken_IsActive()
    {
        var token = CreateToken();

        token.IsActive(Now).Should().BeTrue();
        token.IsRevoked.Should().BeFalse();
    }

    [Fact]
    public void ExpiredToken_IsNotActive()
    {
        var token = CreateToken(TimeSpan.FromMinutes(5));

        token.IsActive(Now.AddMinutes(6)).Should().BeFalse();
        token.IsExpired(Now.AddMinutes(6)).Should().BeTrue();
    }

    [Fact]
    public void Revoke_RecordsReasonAndReplacement()
    {
        var token = CreateToken();
        var replacementId = Guid.NewGuid();

        token.Revoke("rotated", Now, replacementId);

        token.IsRevoked.Should().BeTrue();
        token.RevokedAt.Should().Be(Now);
        token.RevokeReason.Should().Be("rotated");
        token.ReplacedByTokenId.Should().Be(replacementId);
        token.IsActive(Now).Should().BeFalse();
    }

    [Fact]
    public void Revoke_IsIdempotent()
    {
        var token = CreateToken();
        var firstReplacement = Guid.NewGuid();

        token.Revoke("rotated", Now, firstReplacement);
        token.Revoke("reuse_detected", Now.AddMinutes(1), Guid.NewGuid());

        token.RevokeReason.Should().Be("rotated");
        token.ReplacedByTokenId.Should().Be(firstReplacement);
        token.RevokedAt.Should().Be(Now);
    }
}
